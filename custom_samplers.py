"""
Custom Sampler Implementations for llama.cpp direct inference.

Each sampler is a callable that takes (logits, rng) and returns a token ID.
Samplers can optionally specify initialization parameters and have access
to vocab metadata.

Usage:
    from custom_samplers import WhitelistSampler, SecondBestSampler
    
    # Create sampler
    sampler = WhitelistSampler(whitelist_tokens, temperature=0.8)
    
    # Use with LlamaDirect
    output = llm.generate(prompt, custom_sampler=sampler)
"""

import numpy as np
from typing import Set, List, Union, Optional, Protocol, Callable
from pathlib import Path
from abc import ABC, abstractmethod


# ============================================================================
# Base Class / Protocol for Custom Samplers
# ============================================================================

class CustomSampler(ABC):
    """
    Abstract base class for custom samplers.
    
    Subclass this and implement:
      - __call__(logits, rng) -> token_id
    
    Optional:
      - set_vocab_size(vocab_size): called before first use
      - reset(): called between generations if needed
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self._vocab_size: Optional[int] = None
    
    def set_vocab_size(self, vocab_size: int) -> None:
        """Called by LlamaDirect before first use."""
        self._vocab_size = vocab_size
    
    def reset(self) -> None:
        """Optional: reset state between generations."""
        pass
    
    @abstractmethod
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        """
        Sample a token from the logits distribution.
        
        Args:
            logits: Raw logits array of shape (vocab_size,)
            rng: numpy random generator for reproducibility
        
        Returns:
            Selected token ID
        """
        pass
    
    def _apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if self.temperature <= 0:
            return logits  # Will be handled as greedy
        return logits / self.temperature
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute stable softmax probabilities."""
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)


# ============================================================================
# Utility Functions
# ============================================================================

def load_whitelist(filepath: Union[str, Path]) -> Set[int]:
    """Load token IDs from a newline-separated file."""
    whitelist = set()
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Whitelist file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                whitelist.add(int(line))
            except ValueError:
                raise ValueError(f"Invalid token ID on line {line_num}: '{line}'")
    
    return whitelist


# ============================================================================
# Whitelist Sampler - Only sample from allowed tokens
# ============================================================================

class WhitelistSampler(CustomSampler):
    """
    Restricts sampling to a predefined set of allowed tokens.
    
    Efficient implementation: extracts only whitelist logits before softmax,
    so sampling from 50 tokens is O(50), not O(vocab_size).
    
    Args:
        whitelist: Set of allowed token IDs, or path to file with token IDs
        temperature: Sampling temperature (0 = greedy)
    """
    
    def __init__(
        self,
        whitelist: Union[Set[int], List[int], str, Path],
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        
        # Load from file if string/path
        if isinstance(whitelist, (str, Path)):
            self.whitelist = load_whitelist(whitelist)
        else:
            self.whitelist = set(whitelist)
        
        self._whitelist_array: Optional[np.ndarray] = None
    
    def set_vocab_size(self, vocab_size: int) -> None:
        super().set_vocab_size(vocab_size)
        # Filter out invalid tokens and convert to sorted array
        valid_tokens = [t for t in self.whitelist if 0 <= t < vocab_size]
        self._whitelist_array = np.array(sorted(valid_tokens), dtype=np.int32)
    
    def exclude_tokens(self, tokens: Set[int]) -> None:
        """Remove tokens from whitelist (e.g., stop tokens)."""
        self.whitelist -= tokens
        if self._vocab_size is not None:
            valid_tokens = [t for t in self.whitelist if 0 <= t < self._vocab_size]
            self._whitelist_array = np.array(sorted(valid_tokens), dtype=np.int32)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        if self._whitelist_array is None or len(self._whitelist_array) == 0:
            raise ValueError("Whitelist is empty or not initialized. Call set_vocab_size first.")
        
        # Extract only whitelist logits - O(whitelist_size)
        whitelist_logits = logits[self._whitelist_array]
        
        if self.temperature <= 0:
            # Greedy: return token with highest logit
            idx = np.argmax(whitelist_logits)
            return int(self._whitelist_array[idx])
        
        # Apply temperature and softmax
        whitelist_logits = self._apply_temperature(whitelist_logits)
        probs = self._softmax(whitelist_logits)
        
        # Sample from the small distribution
        idx = rng.choice(len(self._whitelist_array), p=probs)
        return int(self._whitelist_array[idx])
    
    def __len__(self) -> int:
        return len(self.whitelist)


# ============================================================================
# Second Best Sampler - Always pick the 2nd most likely token
# ============================================================================

class SecondBestSampler(CustomSampler):
    """
    Always returns the second most likely token (by logit).
    
    This creates interestingly "off" generations that avoid the most
    obvious continuations while staying relatively coherent.
    
    Args:
        temperature: Not used for selection, but included for API consistency
        fallback_to_best: If True, return best token when only 1 valid option
    """
    
    def __init__(self, temperature: float = 1.0, fallback_to_best: bool = True):
        super().__init__(temperature)
        self.fallback_to_best = fallback_to_best
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Get indices that would sort logits in descending order
        sorted_indices = np.argsort(logits)[::-1]
        
        if len(sorted_indices) >= 2:
            return int(sorted_indices[1])  # Second best
        elif self.fallback_to_best and len(sorted_indices) >= 1:
            return int(sorted_indices[0])  # Fallback to best
        else:
            raise ValueError("No valid tokens to sample from")


# ============================================================================
# Nth Best Sampler - Pick the Nth most likely token
# ============================================================================

class NthBestSampler(CustomSampler):
    """
    Returns the Nth most likely token (1-indexed, so n=1 is greedy).
    
    Args:
        n: Which rank to select (1 = best, 2 = second best, etc.)
        temperature: Not used for selection
        fallback: If True, return best available when n > num_valid_tokens
    """
    
    def __init__(self, n: int = 2, temperature: float = 1.0, fallback: bool = True):
        super().__init__(temperature)
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.fallback = fallback
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        sorted_indices = np.argsort(logits)[::-1]
        
        target_idx = self.n - 1  # Convert to 0-indexed
        
        if target_idx < len(sorted_indices):
            return int(sorted_indices[target_idx])
        elif self.fallback and len(sorted_indices) > 0:
            return int(sorted_indices[-1])  # Return lowest available
        else:
            raise ValueError(f"Cannot select rank {self.n} from {len(sorted_indices)} tokens")


# ============================================================================
# Random Top-K Sampler - Uniform random from top K
# ============================================================================

class RandomTopKSampler(CustomSampler):
    """
    Uniformly samples from the top K tokens (ignoring their probabilities).
    
    Unlike regular top-k which samples proportionally to probability,
    this gives equal weight to all tokens in the top K.
    
    Args:
        k: Number of top tokens to consider
        temperature: Not used (uniform sampling)
    """
    
    def __init__(self, k: int = 10, temperature: float = 1.0):
        super().__init__(temperature)
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        sorted_indices = np.argsort(logits)[::-1]
        top_k = sorted_indices[:min(self.k, len(sorted_indices))]
        
        # Uniform random selection from top k
        idx = rng.integers(len(top_k))
        return int(top_k[idx])


# ============================================================================
# Entropy Maximizer - Prefer tokens that increase uncertainty
# ============================================================================

class EntropyAwareSampler(CustomSampler):
    """
    Samples with a bias toward tokens with middle-range probabilities,
    avoiding both the most obvious and the most unlikely tokens.
    
    This encourages more "interesting" or "creative" continuations.
    
    Args:
        temperature: Base temperature for softmax
        entropy_weight: How much to boost middle-probability tokens (0-1)
        top_k: Pre-filter to top K tokens before entropy adjustment
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        entropy_weight: float = 0.5,
        top_k: int = 100
    ):
        super().__init__(temperature)
        self.entropy_weight = np.clip(entropy_weight, 0, 1)
        self.top_k = top_k
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Get top-k indices
        sorted_indices = np.argsort(logits)[::-1]
        top_indices = sorted_indices[:min(self.top_k, len(sorted_indices))]
        top_logits = logits[top_indices]
        
        # Apply temperature
        scaled_logits = self._apply_temperature(top_logits)
        probs = self._softmax(scaled_logits)
        
        # Boost middle-probability tokens
        # Target: probabilities around 1/len(probs) (uniform)
        uniform_prob = 1.0 / len(probs)
        
        # Distance from uniform - tokens close to uniform get boosted
        distance_from_uniform = np.abs(probs - uniform_prob)
        max_distance = max(1 - uniform_prob, uniform_prob)
        
        # Boost factor: 1.0 for uniform, decreasing for extremes
        boost = 1.0 - (distance_from_uniform / max_distance) * self.entropy_weight
        
        # Apply boost and renormalize
        adjusted_probs = probs * boost
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        idx = rng.choice(len(top_indices), p=adjusted_probs)
        return int(top_indices[idx])


# ============================================================================
# Alternating Sampler - Switches between strategies
# ============================================================================

class AlternatingSampler(CustomSampler):
    """
    Alternates between multiple sampling strategies on each token.
    
    Useful for creating varied output patterns.
    
    Args:
        samplers: List of CustomSampler instances to alternate between
        temperature: Fallback temperature (individual samplers may override)
    """
    
    def __init__(
        self,
        samplers: List[CustomSampler],
        temperature: float = 1.0
    ):
        super().__init__(temperature)
        if not samplers:
            raise ValueError("Must provide at least one sampler")
        self.samplers = samplers
        self._step = 0
    
    def set_vocab_size(self, vocab_size: int) -> None:
        super().set_vocab_size(vocab_size)
        for sampler in self.samplers:
            sampler.set_vocab_size(vocab_size)
    
    def reset(self) -> None:
        self._step = 0
        for sampler in self.samplers:
            sampler.reset()
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        sampler = self.samplers[self._step % len(self.samplers)]
        self._step += 1
        return sampler(logits, rng)


# ============================================================================
# Contrastive Sampler - Penalize recent tokens
# ============================================================================

class ContrastiveSampler(CustomSampler):
    """
    Penalizes recently generated tokens to encourage diversity.
    
    Tracks the last N tokens and reduces their logits.
    
    Args:
        temperature: Sampling temperature
        window_size: How many recent tokens to track
        penalty: Logit penalty for recent tokens (positive = penalize more)
        decay: Penalty decay factor for older tokens (1.0 = no decay)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        window_size: int = 32,
        penalty: float = 1.0,
        decay: float = 0.95
    ):
        super().__init__(temperature)
        self.window_size = window_size
        self.penalty = penalty
        self.decay = decay
        self._recent_tokens: List[int] = []
    
    def reset(self) -> None:
        self._recent_tokens = []
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        modified_logits = logits.copy()
        
        # Apply decay-weighted penalties to recent tokens
        for i, token in enumerate(reversed(self._recent_tokens)):
            age = i + 1
            current_penalty = self.penalty * (self.decay ** age)
            if 0 <= token < len(modified_logits):
                modified_logits[token] -= current_penalty
        
        # Sample
        if self.temperature <= 0:
            selected = int(np.argmax(modified_logits))
        else:
            scaled_logits = self._apply_temperature(modified_logits)
            probs = self._softmax(scaled_logits)
            selected = int(rng.choice(len(probs), p=probs))
        
        # Track this token
        self._recent_tokens.append(selected)
        if len(self._recent_tokens) > self.window_size:
            self._recent_tokens.pop(0)
        
        return selected


# ============================================================================
# Weighted Blend Sampler - Combine multiple strategies
# ============================================================================

class WeightedBlendSampler(CustomSampler):
    """
    Blends probabilities from multiple samplers using weighted average.
    
    Note: This requires samplers that can return probability distributions,
    so we use a simplified approach - blend logit modifications.
    
    Args:
        samplers: List of (weight, sampler) tuples
        temperature: Final sampling temperature
    """
    
    def __init__(
        self,
        samplers: List[tuple],  # List of (weight: float, sampler: CustomSampler)
        temperature: float = 1.0
    ):
        super().__init__(temperature)
        if not samplers:
            raise ValueError("Must provide at least one sampler")
        
        total_weight = sum(w for w, _ in samplers)
        self.samplers = [(w / total_weight, s) for w, s in samplers]
    
    def set_vocab_size(self, vocab_size: int) -> None:
        super().set_vocab_size(vocab_size)
        for _, sampler in self.samplers:
            sampler.set_vocab_size(vocab_size)
    
    def reset(self) -> None:
        for _, sampler in self.samplers:
            sampler.reset()
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # For now, just randomly select one sampler weighted by their weights
        weights = [w for w, _ in self.samplers]
        samplers = [s for _, s in self.samplers]
        
        selected_sampler = rng.choice(len(samplers), p=weights)
        return samplers[selected_sampler](logits, rng)


# ============================================================================
# Entropix Sampler - Entropy/Varentropy-aware adaptive sampling
# Based on https://github.com/xjdr-alt/entropix
# ============================================================================

class EntropixSampler(CustomSampler):
    """
    Entropy and Varentropy aware adaptive sampler.
    
    Based on the Entropix project (xjdr-alt/entropix), this sampler dynamically
    adjusts its strategy based on the model's uncertainty:
    
    - LELV (Low Entropy, Low Varentropy): High confidence → Greedy selection
    - HELV (High Entropy, Low Varentropy): Uncertain → Temperature boost + top-p
    - LEHV (Low Entropy, High Varentropy): Multiple good options → Sample from top
    - HEHV (High Entropy, High Varentropy): Very uncertain → Resample with penalty
    
    Args:
        temperature: Base temperature for sampling
        low_ent_thresh: Threshold for low entropy (default 0.3)
        med_ent_thresh: Threshold for medium entropy (default 1.2)
        high_ent_thresh: Threshold for high entropy (default 2.5)
        low_var_thresh: Threshold for low varentropy (default 1.2)
        high_var_thresh: Threshold for high varentropy (default 2.5)
        helv_temp_boost: Temperature multiplier for HELV case (default 1.5)
        top_k: Top-K for filtering in uncertain cases (default 50)
    """
    
    LN_2 = 0.69314718056  # ln(2)
    
    def __init__(
        self,
        temperature: float = 1.0,
        low_ent_thresh: float = 0.3,
        med_ent_thresh: float = 1.2,
        high_ent_thresh: float = 2.5,
        low_var_thresh: float = 1.2,
        high_var_thresh: float = 2.5,
        helv_temp_boost: float = 1.5,
        top_k: int = 50
    ):
        super().__init__(temperature)
        self.low_ent_thresh = low_ent_thresh
        self.med_ent_thresh = med_ent_thresh
        self.high_ent_thresh = high_ent_thresh
        self.low_var_thresh = low_var_thresh
        self.high_var_thresh = high_var_thresh
        self.helv_temp_boost = helv_temp_boost
        self.top_k = top_k
        
        # Stats tracking
        self._case_counts = {"LELV": 0, "HELV": 0, "LEHV": 0, "HEHV": 0, "DEFAULT": 0}
    
    def reset(self) -> None:
        self._case_counts = {"LELV": 0, "HELV": 0, "LEHV": 0, "HEHV": 0, "DEFAULT": 0}
    
    def _compute_entropy_varentropy(self, logits: np.ndarray) -> tuple:
        """Compute entropy and varentropy from logits."""
        # Softmax to get probabilities
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits)
        
        # Log probabilities (add small epsilon for numerical stability)
        log_probs = np.log(probs + 1e-10)
        
        # Entropy: -sum(p * log(p))
        entropy = -np.sum(probs * log_probs)
        
        # Varentropy: variance of log probabilities weighted by probabilities
        # This measures how spread out the uncertainty is
        mean_log_prob = np.sum(probs * log_probs)
        varentropy = np.sum(probs * (log_probs - mean_log_prob) ** 2)
        
        return entropy, varentropy, probs
    
    def _classify_state(self, entropy: float, varentropy: float) -> str:
        """Classify the current state based on entropy and varentropy."""
        low_ent = entropy < self.low_ent_thresh
        high_ent = entropy > self.high_ent_thresh
        med_ent = entropy > self.med_ent_thresh
        low_var = varentropy < self.low_var_thresh
        high_var = varentropy > self.high_var_thresh
        
        if low_ent and low_var:
            return "LELV"  # Confident - greedy
        elif high_ent and low_var:
            return "HELV"  # Uncertain but consistent - boost temperature
        elif not high_ent and high_var:
            return "LEHV"  # Multiple good options - sample from top
        elif med_ent and high_var:
            return "HEHV"  # Very uncertain - resample with penalty
        else:
            return "DEFAULT"
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Compute entropy metrics
        entropy, varentropy, probs = self._compute_entropy_varentropy(logits)
        
        # Classify state
        state = self._classify_state(entropy, varentropy)
        self._case_counts[state] += 1
        
        if state == "LELV":
            # Low entropy, low varentropy: confident → greedy
            return int(np.argmax(logits))
        
        elif state == "HELV":
            # High entropy, low varentropy: uncertain but focused
            # Boost temperature and use top-p sampling
            boosted_temp = self.temperature * self.helv_temp_boost
            scaled_logits = logits / boosted_temp
            boosted_probs = self._softmax(scaled_logits)
            
            # Top-p filtering (nucleus sampling)
            sorted_indices = np.argsort(boosted_probs)[::-1]
            sorted_probs = boosted_probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumsum, 0.9) + 1
            
            top_indices = sorted_indices[:cutoff_idx]
            top_probs = sorted_probs[:cutoff_idx]
            top_probs = top_probs / np.sum(top_probs)
            
            idx = rng.choice(len(top_indices), p=top_probs)
            return int(top_indices[idx])
        
        elif state == "LEHV":
            # Low entropy, high varentropy: multiple viable paths
            # Sample from top-k with regular temperature
            sorted_indices = np.argsort(logits)[::-1]
            top_indices = sorted_indices[:self.top_k]
            top_logits = logits[top_indices]
            
            scaled = self._apply_temperature(top_logits)
            top_probs = self._softmax(scaled)
            
            idx = rng.choice(len(top_indices), p=top_probs)
            return int(top_indices[idx])
        
        elif state == "HEHV":
            # High entropy, high varentropy: very uncertain
            # Resample: penalize the top token and sample again
            best_token = np.argmax(logits)
            penalized_logits = logits.copy()
            penalized_logits[best_token] = float('-inf')
            
            scaled = self._apply_temperature(penalized_logits)
            new_probs = self._softmax(scaled)
            
            return int(rng.choice(len(new_probs), p=new_probs))
        
        else:
            # Default: standard temperature sampling
            if self.temperature <= 0:
                return int(np.argmax(logits))
            
            scaled_logits = self._apply_temperature(logits)
            probs = self._softmax(scaled_logits)
            return int(rng.choice(len(probs), p=probs))
    
    def get_stats(self) -> dict:
        """Return statistics about which sampling strategies were used."""
        return self._case_counts.copy()


# ============================================================================
# CREATIVE / EXPERIMENTAL SAMPLERS
# ============================================================================

# ----------------------------------------------------------------------------
# Chaos Sampler - Pick the LEAST likely token (pure chaos mode)
# ----------------------------------------------------------------------------

class ChaosSampler(CustomSampler):
    """
    Always picks the least likely token. Pure chaos mode.
    
    Creates completely unhinged, incoherent output. Use for entertainment only!
    
    Args:
        nth_worst: Pick the Nth worst token (1 = absolute worst, 2 = second worst)
        temperature: Not used
    """
    
    def __init__(self, nth_worst: int = 1, temperature: float = 1.0):
        super().__init__(temperature)
        self.nth_worst = max(1, nth_worst)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        sorted_indices = np.argsort(logits)  # Ascending order (worst first)
        target_idx = min(self.nth_worst - 1, len(sorted_indices) - 1)
        return int(sorted_indices[target_idx])


# ----------------------------------------------------------------------------
# Stutter Sampler - Sometimes repeats the previous token
# ----------------------------------------------------------------------------

class StutterSampler(CustomSampler):
    """
    Randomly repeats the previous token with some probability.
    
    Creates a stuttering effect: "The the the cat sat sat on the mat mat"
    
    Args:
        stutter_prob: Probability of repeating (0-1)
        max_repeats: Maximum consecutive repeats allowed
        temperature: Sampling temperature when not stuttering
    """
    
    def __init__(
        self,
        stutter_prob: float = 0.3,
        max_repeats: int = 3,
        temperature: float = 1.0
    ):
        super().__init__(temperature)
        self.stutter_prob = np.clip(stutter_prob, 0, 1)
        self.max_repeats = max_repeats
        self._last_token: Optional[int] = None
        self._repeat_count = 0
    
    def reset(self) -> None:
        self._last_token = None
        self._repeat_count = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Check if we should stutter
        if (self._last_token is not None and 
            self._repeat_count < self.max_repeats and
            rng.random() < self.stutter_prob):
            self._repeat_count += 1
            return self._last_token
        
        # Normal sampling
        if self.temperature <= 0:
            selected = int(np.argmax(logits))
        else:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            selected = int(rng.choice(len(probs), p=probs))
        
        # Track for potential stutter
        if selected != self._last_token:
            self._repeat_count = 0
        self._last_token = selected
        
        return selected


# ----------------------------------------------------------------------------
# Temperature Wave Sampler - Oscillates temperature over time
# ----------------------------------------------------------------------------

class TemperatureWaveSampler(CustomSampler):
    """
    Oscillates temperature in a wave pattern over tokens.
    
    Creates output that alternates between focused and creative.
    
    Args:
        min_temp: Minimum temperature (focused)
        max_temp: Maximum temperature (creative)
        period: Number of tokens for one full wave cycle
        wave_type: 'sine', 'triangle', or 'square'
    """
    
    def __init__(
        self,
        min_temp: float = 0.3,
        max_temp: float = 1.5,
        period: int = 20,
        wave_type: str = 'sine'
    ):
        super().__init__(1.0)  # Base temp not used directly
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.period = max(1, period)
        self.wave_type = wave_type
        self._step = 0
    
    def reset(self) -> None:
        self._step = 0
    
    def _get_current_temp(self) -> float:
        phase = (self._step % self.period) / self.period
        
        if self.wave_type == 'sine':
            # Sine wave: smooth oscillation
            wave = (np.sin(2 * np.pi * phase) + 1) / 2
        elif self.wave_type == 'triangle':
            # Triangle wave: linear up then down
            wave = 1 - abs(2 * phase - 1)
        elif self.wave_type == 'square':
            # Square wave: binary flip
            wave = 1.0 if phase < 0.5 else 0.0
        else:
            wave = 0.5
        
        return self.min_temp + wave * (self.max_temp - self.min_temp)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        temp = self._get_current_temp()
        self._step += 1
        
        if temp <= 0:
            return int(np.argmax(logits))
        
        scaled = logits / temp
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Blacklist Sampler - Forbid specific tokens
# ----------------------------------------------------------------------------

class BlacklistSampler(CustomSampler):
    """
    Forbids specific tokens from being generated.
    
    Inverse of WhitelistSampler - blocks specific tokens.
    
    Args:
        blacklist: Set of forbidden token IDs
        temperature: Sampling temperature
    """
    
    def __init__(
        self,
        blacklist: Union[Set[int], List[int]],
        temperature: float = 1.0
    ):
        super().__init__(temperature)
        self.blacklist = set(blacklist)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        modified_logits = logits.copy()
        
        # Set blacklisted tokens to -inf
        for token_id in self.blacklist:
            if 0 <= token_id < len(modified_logits):
                modified_logits[token_id] = float('-inf')
        
        if self.temperature <= 0:
            return int(np.argmax(modified_logits))
        
        scaled = self._apply_temperature(modified_logits)
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Drunken Sampler - Adds random noise to logits
# ----------------------------------------------------------------------------

class DrunkenSampler(CustomSampler):
    """
    Adds random noise to logits before sampling.
    
    Simulates "drunk" text generation - mostly coherent with random mistakes.
    
    Args:
        noise_scale: Standard deviation of noise to add
        temperature: Base sampling temperature
        drunk_level: 0-10 scale, affects noise (overrides noise_scale if set)
    """
    
    def __init__(
        self,
        noise_scale: float = 2.0,
        temperature: float = 1.0,
        drunk_level: int = None
    ):
        super().__init__(temperature)
        if drunk_level is not None:
            # Map 0-10 to reasonable noise scales
            self.noise_scale = drunk_level * 0.5
        else:
            self.noise_scale = noise_scale
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Add Gaussian noise to logits
        noise = rng.normal(0, self.noise_scale, size=logits.shape)
        noisy_logits = logits + noise
        
        if self.temperature <= 0:
            return int(np.argmax(noisy_logits))
        
        scaled = self._apply_temperature(noisy_logits)
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Echo Sampler - Repeats patterns from earlier
# ----------------------------------------------------------------------------

class EchoSampler(CustomSampler):
    """
    With some probability, repeats a token from earlier in the generation.
    
    Creates an echo/callback effect where phrases resurface.
    
    Args:
        echo_prob: Probability of echoing instead of normal sampling
        lookback_range: How far back to look for echo candidates
        temperature: Normal sampling temperature
    """
    
    def __init__(
        self,
        echo_prob: float = 0.15,
        lookback_range: int = 50,
        temperature: float = 1.0
    ):
        super().__init__(temperature)
        self.echo_prob = np.clip(echo_prob, 0, 1)
        self.lookback_range = lookback_range
        self._history: List[int] = []
    
    def reset(self) -> None:
        self._history = []
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Check if we should echo
        if self._history and rng.random() < self.echo_prob:
            # Pick a random token from history
            lookback = min(len(self._history), self.lookback_range)
            history_slice = self._history[-lookback:]
            selected = int(rng.choice(history_slice))
        else:
            # Normal sampling
            if self.temperature <= 0:
                selected = int(np.argmax(logits))
            else:
                scaled = self._apply_temperature(logits)
                probs = self._softmax(scaled)
                selected = int(rng.choice(len(probs), p=probs))
        
        self._history.append(selected)
        return selected


# ----------------------------------------------------------------------------
# Gradual Chaos Sampler - Temperature increases over time
# ----------------------------------------------------------------------------

class GradualChaosSampler(CustomSampler):
    """
    Temperature gradually increases, making output progressively unhinged.
    
    Starts coherent, ends chaotic. Great for "descending into madness" effects.
    
    Args:
        start_temp: Initial temperature
        end_temp: Final temperature
        ramp_tokens: Number of tokens to reach end temperature
        mode: 'linear', 'exponential', or 'sudden' (halfway switch)
    """
    
    def __init__(
        self,
        start_temp: float = 0.3,
        end_temp: float = 2.5,
        ramp_tokens: int = 100,
        mode: str = 'linear'
    ):
        super().__init__(start_temp)
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.ramp_tokens = max(1, ramp_tokens)
        self.mode = mode
        self._step = 0
    
    def reset(self) -> None:
        self._step = 0
    
    def _get_current_temp(self) -> float:
        progress = min(1.0, self._step / self.ramp_tokens)
        
        if self.mode == 'linear':
            return self.start_temp + progress * (self.end_temp - self.start_temp)
        elif self.mode == 'exponential':
            # Exponential ramp
            return self.start_temp * (self.end_temp / self.start_temp) ** progress
        elif self.mode == 'sudden':
            # Sudden switch at halfway
            return self.end_temp if progress > 0.5 else self.start_temp
        else:
            return self.start_temp + progress * (self.end_temp - self.start_temp)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        temp = self._get_current_temp()
        self._step += 1
        
        if temp <= 0:
            return int(np.argmax(logits))
        
        scaled = logits / temp
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Mood Swing Sampler - Randomly changes strategy
# ----------------------------------------------------------------------------

class MoodSwingSampler(CustomSampler):
    """
    Randomly switches between greedy and chaotic on each token.
    
    Creates jarring, unpredictable output with sudden mood swings.
    
    Args:
        greedy_prob: Probability of being greedy (focused)
        chaos_temp: Temperature when in chaotic mode
    """
    
    def __init__(self, greedy_prob: float = 0.5, chaos_temp: float = 2.0):
        super().__init__(1.0)
        self.greedy_prob = np.clip(greedy_prob, 0, 1)
        self.chaos_temp = chaos_temp
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.greedy_prob:
            # Greedy mode
            return int(np.argmax(logits))
        else:
            # Chaos mode
            scaled = logits / self.chaos_temp
            probs = self._softmax(scaled)
            return int(rng.choice(len(probs), p=probs))

# ----------------------------------------------------------------------------
# Lucky Number Sampler - Special behavior on "lucky" steps
# ----------------------------------------------------------------------------

class LuckyNumberSampler(CustomSampler):
    """
    Special sampling behavior on "lucky" token positions.
    
    Args:
        lucky_numbers: List of lucky positions (e.g., [7, 13, 42, 69, 100])
        lucky_behavior: 'greedy', 'chaos', 'second_best', or 'random_top_k'
        normal_temp: Temperature for normal tokens
    """
    
    def __init__(
        self,
        lucky_numbers: List[int] = None,
        lucky_behavior: str = 'chaos',
        normal_temp: float = 0.8
    ):
        super().__init__(normal_temp)
        self.lucky_numbers = set(lucky_numbers or [7, 13, 21, 42, 67, 69, 100])
        self.lucky_behavior = lucky_behavior
        self._step = 0
    
    def reset(self) -> None:
        self._step = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._step += 1
        
        if self._step in self.lucky_numbers:
            # Lucky behavior!
            if self.lucky_behavior == 'greedy':
                return int(np.argmax(logits))
            elif self.lucky_behavior == 'chaos':
                return int(np.argmin(logits))  # Worst token!
            elif self.lucky_behavior == 'second_best':
                sorted_idx = np.argsort(logits)[::-1]
                return int(sorted_idx[1]) if len(sorted_idx) > 1 else int(sorted_idx[0])
            elif self.lucky_behavior == 'random_top_k':
                sorted_idx = np.argsort(logits)[::-1]
                return int(rng.choice(sorted_idx[:10]))
        
        # Normal sampling
        if self.temperature <= 0:
            return int(np.argmax(logits))
        
        scaled = self._apply_temperature(logits)
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Confidence Tracker - Adjusts strategy based on model confidence
# ----------------------------------------------------------------------------

class ConfidenceAdaptiveSampler(CustomSampler):
    """
    Adapts sampling based on model's confidence (max probability).
    
    High confidence → trust the model (greedy or low temp)
    Low confidence → explore more (higher temp)
    
    Args:
        confident_threshold: Max prob above this = confident
        confident_temp: Temperature when confident
        uncertain_temp: Temperature when uncertain
    """
    
    def __init__(
        self,
        confident_threshold: float = 0.7,
        confident_temp: float = 0.3,
        uncertain_temp: float = 1.5
    ):
        super().__init__(1.0)
        self.confident_threshold = confident_threshold
        self.confident_temp = confident_temp
        self.uncertain_temp = uncertain_temp
        self._confidence_history: List[float] = []
    
    def reset(self) -> None:
        self._confidence_history = []
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Get base probabilities
        probs = self._softmax(logits)
        max_prob = np.max(probs)
        self._confidence_history.append(max_prob)
        
        # Choose temperature based on confidence
        if max_prob >= self.confident_threshold:
            temp = self.confident_temp
        else:
            temp = self.uncertain_temp
        
        if temp <= 0:
            return int(np.argmax(logits))
        
        scaled = logits / temp
        scaled_probs = self._softmax(scaled)
        return int(rng.choice(len(scaled_probs), p=scaled_probs))
    
    def get_confidence_history(self) -> List[float]:
        return self._confidence_history.copy()


# ----------------------------------------------------------------------------
# Bracket Sampler - Groups tokens and alternates between group strategies
# ----------------------------------------------------------------------------

class BracketSampler(CustomSampler):
    """
    Alternates between different sampling strategies in "brackets".
    
    Like a tournament: certain portions use different rules.
    
    Args:
        bracket_size: Tokens per bracket
        strategies: List of (temp, top_k) tuples for each bracket type
    """
    
    def __init__(
        self,
        bracket_size: int = 10,
        strategies: List[tuple] = None
    ):
        super().__init__(1.0)
        self.bracket_size = max(1, bracket_size)
        # Default: alternate between careful and exploratory
        self.strategies = strategies or [(0.5, 5), (1.0, 50), (1.5, 100)]
        self._step = 0
    
    def reset(self) -> None:
        self._step = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        bracket = (self._step // self.bracket_size) % len(self.strategies)
        temp, top_k = self.strategies[bracket]
        self._step += 1
        
        modified_logits = logits.copy()
        
        # Apply top-k
        if top_k < len(logits):
            sorted_indices = np.argsort(modified_logits)[::-1]
            mask = np.ones(len(modified_logits), dtype=bool)
            mask[sorted_indices[:top_k]] = False
            modified_logits[mask] = float('-inf')
        
        if temp <= 0:
            return int(np.argmax(modified_logits))
        
        scaled = modified_logits / temp
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ============================================================================
# GASLIGHTING / DEFERRED SAMPLERS
# These let the model "choose" freely, then override the result.
# The model's KV cache reflects the overridden token, not what it wanted.
# This creates deliciously misaligned downstream probabilities!
# ============================================================================

class GaslightSampler(CustomSampler):
    """
    Base gaslighting sampler: model picks normally, then we override.
    
    The model's logits reflect what it WANTED to say, but we substitute
    a different token. The overridden token goes into the KV cache,
    so future predictions are based on a context the model didn't "intend".
    
    This is different from constrained sampling where we filter BEFORE
    sampling - here we let the model express its full preference, then
    gaslight it by pretending it said something else.
    
    Args:
        override_fn: Function(intended_token, logits, rng) -> actual_token
        temperature: Temperature for model's "intended" sampling
    """
    
    def __init__(
        self,
        override_fn: Callable[[int, np.ndarray, np.random.Generator], int] = None,
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.override_fn = override_fn or self._default_override
        self._intended_history: List[int] = []
        self._actual_history: List[int] = []
    
    def _default_override(self, intended: int, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Default: just return what the model wanted (no gaslighting)
        return intended
    
    def reset(self) -> None:
        self._intended_history = []
        self._actual_history = []
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Let the model "choose" freely
        if self.temperature <= 0:
            intended = int(np.argmax(logits))
        else:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            intended = int(rng.choice(len(probs), p=probs))
        
        self._intended_history.append(intended)
        
        # Now gaslight: override with something else
        actual = self.override_fn(intended, logits, rng)
        self._actual_history.append(actual)
        
        return actual
    
    def get_gaslight_stats(self) -> dict:
        """How much did we gaslight the model?"""
        if not self._intended_history:
            return {}
        
        mismatches = sum(1 for i, a in zip(self._intended_history, self._actual_history) if i != a)
        return {
            'total_tokens': len(self._intended_history),
            'gaslighted': mismatches,
            'gaslight_rate': mismatches / len(self._intended_history) if self._intended_history else 0,
            'intended_tokens': self._intended_history[-10:],  # Last 10
            'actual_tokens': self._actual_history[-10:],
        }


# ----------------------------------------------------------------------------
# Deferred Nth-Best: Model picks best, we give it Nth best instead
# ----------------------------------------------------------------------------

class DeferredNthBestSampler(CustomSampler):
    """
    Model samples normally, but we substitute with Nth best token.
    
    The model's KV cache thinks it said the Nth best, but its logits
    were computed expecting the 1st best. Chaos ensues as predictions
    accumulate based on an "unintended" context.
    
    Args:
        n: Which rank to substitute (1 = no change, 2 = second best, etc.)
        sample_first: If True, model samples freely first. If False, greedy then substitute.
        temperature: Temperature for initial sampling (if sample_first=True)
    """
    
    def __init__(self, n: int = 2, sample_first: bool = False, temperature: float = 0.8):
        super().__init__(temperature)
        self.n = max(1, n)
        self.sample_first = sample_first
        self._substitutions = 0
        self._total = 0
    
    def reset(self) -> None:
        self._substitutions = 0
        self._total = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._total += 1
        
        # Get what the model "wants"
        if self.sample_first and self.temperature > 0:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            intended = int(rng.choice(len(probs), p=probs))
        else:
            intended = int(np.argmax(logits))
        
        # Get the Nth best
        sorted_indices = np.argsort(logits)[::-1]
        target_idx = min(self.n - 1, len(sorted_indices) - 1)
        actual = int(sorted_indices[target_idx])
        
        if actual != intended:
            self._substitutions += 1
        
        return actual
    
    def get_stats(self) -> dict:
        return {
            'total': self._total,
            'substitutions': self._substitutions,
            'substitution_rate': self._substitutions / self._total if self._total else 0
        }


# ----------------------------------------------------------------------------
# Deferred Whitelist: Model picks freely, but we force whitelist compliance
# ----------------------------------------------------------------------------

class DeferredWhitelistSampler(CustomSampler):
    """
    Model samples from full vocabulary, then we force whitelist compliance.
    
    Unlike WhitelistSampler which constrains BEFORE sampling, this lets
    the model compute logits for any token, picks normally, THEN checks
    if the result is in the whitelist. If not, substitutes the highest
    probability whitelisted token.
    
    This means the model's internal probability distribution wasn't 
    constrained - it might have been 90% confident in token X, but if
    X isn't whitelisted, we silently replace it. The model then sees
    the replacement in context and gets confused.
    
    Args:
        whitelist: Set of allowed token IDs
        temperature: Sampling temperature
        substitute_strategy: 'highest' (highest logit in whitelist) or 'random'
    """
    
    def __init__(
        self,
        whitelist: Union[Set[int], List[int]],
        temperature: float = 0.8,
        substitute_strategy: str = 'highest'
    ):
        super().__init__(temperature)
        self.whitelist = set(whitelist)
        self.substitute_strategy = substitute_strategy
        self._whitelist_array: Optional[np.ndarray] = None
        
        self._intended_in_whitelist = 0
        self._forced_substitutions = 0
        self._total = 0
    
    def set_vocab_size(self, vocab_size: int) -> None:
        super().set_vocab_size(vocab_size)
        valid = [t for t in self.whitelist if 0 <= t < vocab_size]
        self._whitelist_array = np.array(sorted(valid), dtype=np.int32)
    
    def exclude_tokens(self, tokens: Set[int]) -> None:
        """Remove tokens from whitelist (e.g., stop tokens)."""
        self.whitelist -= tokens
        if self._vocab_size is not None:
            valid = [t for t in self.whitelist if 0 <= t < self._vocab_size]
            self._whitelist_array = np.array(sorted(valid), dtype=np.int32)
    
    def reset(self) -> None:
        self._intended_in_whitelist = 0
        self._forced_substitutions = 0
        self._total = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._total += 1
        
        # Model samples freely from full vocabulary
        if self.temperature <= 0:
            intended = int(np.argmax(logits))
        else:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            intended = int(rng.choice(len(probs), p=probs))
        
        # Check if in whitelist
        if intended in self.whitelist:
            self._intended_in_whitelist += 1
            return intended
        
        # Not in whitelist - substitute!
        self._forced_substitutions += 1
        
        if self._whitelist_array is None or len(self._whitelist_array) == 0:
            # Fallback to intended if whitelist is empty
            return intended
        
        if self.substitute_strategy == 'highest':
            # Pick highest logit from whitelist
            whitelist_logits = logits[self._whitelist_array]
            best_idx = np.argmax(whitelist_logits)
            return int(self._whitelist_array[best_idx])
        else:
            # Random from whitelist
            return int(rng.choice(self._whitelist_array))
    
    def get_stats(self) -> dict:
        return {
            'total': self._total,
            'intended_in_whitelist': self._intended_in_whitelist,
            'forced_substitutions': self._forced_substitutions,
            'compliance_rate': self._intended_in_whitelist / self._total if self._total else 0,
            'gaslight_rate': self._forced_substitutions / self._total if self._total else 0,
        }


# ----------------------------------------------------------------------------
# Substitution Sampler: Replace specific tokens with alternatives
# ----------------------------------------------------------------------------

class SubstitutionSampler(CustomSampler):
    """
    If model picks token X, secretly replace with token Y.
    
    Like a find-replace but at the token level. The model thinks it's
    writing one thing but we're secretly swapping tokens.
    
    Args:
        substitutions: Dict mapping token_id -> replacement_token_id
        temperature: Sampling temperature
    """
    
    def __init__(
        self,
        substitutions: dict = None,
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.substitutions = substitutions or {}
        self._sub_count = 0
        self._total = 0
    
    def reset(self) -> None:
        self._sub_count = 0
        self._total = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._total += 1
        
        # Normal sampling
        if self.temperature <= 0:
            intended = int(np.argmax(logits))
        else:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            intended = int(rng.choice(len(probs), p=probs))
        
        # Check for substitution
        if intended in self.substitutions:
            self._sub_count += 1
            return self.substitutions[intended]
        
        return intended
    
    def get_stats(self) -> dict:
        return {
            'total': self._total,
            'substitutions': self._sub_count,
            'substitution_rate': self._sub_count / self._total if self._total else 0
        }


# ----------------------------------------------------------------------------
# Mismatch Sampler: Always pick something DIFFERENT from what model wanted
# ----------------------------------------------------------------------------

class MismatchSampler(CustomSampler):
    """
    Always returns a different token than what the model intended.
    
    If model's top choice is X, we pick from everything EXCEPT X.
    Maximum cognitive dissonance for the model!
    
    Args:
        mismatch_from: 'top1', 'top3', 'top5' - exclude this many top tokens
        temperature: Temperature for selecting among remaining tokens
    """
    
    def __init__(self, mismatch_from: str = 'top1', temperature: float = 1.0):
        super().__init__(temperature)
        self.mismatch_count = int(mismatch_from.replace('top', '')) if 'top' in mismatch_from else 1
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        sorted_indices = np.argsort(logits)[::-1]
        
        # Exclude top N tokens
        excluded = set(sorted_indices[:self.mismatch_count])
        remaining = [i for i in range(len(logits)) if i not in excluded]
        
        if not remaining:
            # Fallback if we excluded everything
            return int(sorted_indices[-1])
        
        # Sample from remaining
        remaining_logits = logits[remaining]
        
        if self.temperature <= 0:
            best_idx = np.argmax(remaining_logits)
            return remaining[best_idx]
        
        scaled = remaining_logits / self.temperature
        probs = self._softmax(scaled)
        idx = rng.choice(len(remaining), p=probs)
        return remaining[idx]


# ----------------------------------------------------------------------------
# Contrarian Sampler: Opposite of model's confidence
# ----------------------------------------------------------------------------

class ContrarianSampler(CustomSampler):
    """
    The more confident the model is, the more likely we override.
    
    If model is 95% confident in token X, we're 95% likely to pick
    something else. If model is uncertain (50/50), we go with its choice.
    
    Maximum chaos when model is confident, maximum coherence when uncertain.
    
    Args:
        contrarian_scale: How contrarian to be (0 = never override, 1 = full contrarian)
        temperature: Temperature for override sampling
    """
    
    def __init__(self, contrarian_scale: float = 0.8, temperature: float = 1.0):
        super().__init__(temperature)
        self.contrarian_scale = np.clip(contrarian_scale, 0, 1)
        self._overrides = 0
        self._total = 0
    
    def reset(self) -> None:
        self._overrides = 0
        self._total = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._total += 1
        
        probs = self._softmax(logits)
        intended = int(np.argmax(probs))
        max_prob = probs[intended]
        
        # Probability to be contrarian scales with model's confidence
        override_prob = max_prob * self.contrarian_scale
        
        if rng.random() < override_prob:
            # Be contrarian - pick from non-top tokens
            self._overrides += 1
            
            # Zero out best token and resample
            modified_probs = probs.copy()
            modified_probs[intended] = 0
            modified_probs = modified_probs / np.sum(modified_probs)
            
            return int(rng.choice(len(modified_probs), p=modified_probs))
        
        return intended
    
    def get_stats(self) -> dict:
        return {
            'total': self._total,
            'overrides': self._overrides,
            'contrarian_rate': self._overrides / self._total if self._total else 0
        }


# ----------------------------------------------------------------------------
# Breadth Sampler - Pick token that maximizes next-step entropy (keeps options open)
# ----------------------------------------------------------------------------

class BreadthSampler(CustomSampler):
    """
    Picks the token that leads to the LEAST confident NEXT prediction.

    Anti-greedy: instead of committing to high confidence, maximize uncertainty
    in the continuation. Keeps creative doors open.

    For each top-k candidate:
      1. Peek ahead one token (requires forward_fn callback)
      2. Compute entropy of next distribution: H = -sum(p * log(p))
      3. Pick candidate with highest entropy (most uncertain continuation)

    Tradeoff: k forward passes per token. Slow but interesting for creative bursts.

    Args:
        branch_k: How many candidates to evaluate (default 5)
        temperature: Temperature for initial candidate selection
        weight_by_prob: If True, score = p(token) * entropy(next). Avoids picking
                        unlikely tokens just because they lead to chaos.
    """

    def __init__(
        self,
        branch_k: int = 5,
        temperature: float = 0.8,
        weight_by_prob: bool = False,
        top_p: float = 1.0,
        min_p: float = 0.0
    ):
        super().__init__(temperature)
        self.branch_k = branch_k
        self.weight_by_prob = weight_by_prob
        self.top_p = top_p
        self.min_p = min_p
        self._forward_fn: Optional[Callable[[int], np.ndarray]] = None
        self._entropy_history: List[float] = []

    def set_forward_fn(self, fn: Callable[[int], np.ndarray]) -> None:
        """
        Set callback for peeking ahead.

        fn(token_id) -> next_logits: np.ndarray of shape (vocab_size,)
        Should append token_id to context, get logits, then restore context.
        """
        self._forward_fn = fn

    def reset(self) -> None:
        self._entropy_history = []

    def _entropy(self, logits: np.ndarray) -> float:
        """H = -sum(p * log(p))"""
        probs = self._softmax(logits)
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Apply temperature
        scaled_logits = self._apply_temperature(logits)
        probs = self._softmax(scaled_logits)
        
        # 1. Filter candidates using top_p / min_p
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Min P
        if self.min_p > 0:
            p_max = sorted_probs[0]
            thresh = self.min_p * p_max
            valid_mask = sorted_probs >= thresh
            sorted_indices = sorted_indices[valid_mask]
            sorted_probs = sorted_probs[valid_mask]
        
        # Top P
        if self.top_p < 1.0:
            cumsum = np.cumsum(sorted_probs)
            # Use searchsorted to find cutoff index
            cutoff = np.searchsorted(cumsum, self.top_p) + 1
            # Ensure we select at least one
            cutoff = min(max(cutoff, 1), len(sorted_indices))
            sorted_indices = sorted_indices[:cutoff]
            sorted_probs = sorted_probs[:cutoff]
            
        # 2. Pick top-K from the filtered set
        # (BreadthSampler evaluates ALL these candidates)
        candidates = sorted_indices[:self.branch_k]
        candidate_probs = sorted_probs[:self.branch_k]
        
        # Normalize probs for selection if needed
        if candidate_probs.sum() > 0:
            candidate_probs = candidate_probs / candidate_probs.sum()

        # If no forward function, fall back to random from candidates weighted by entropy potential
        # (just pick randomly from top-k as we can't peek ahead)
        if self._forward_fn is None:
            idx = rng.choice(len(candidates), p=candidate_probs / candidate_probs.sum())
            return int(candidates[idx])

        best_token = None
        best_score = -1.0
        best_entropy = 0.0

        for i, cand in enumerate(candidates):
            # Peek ahead: what's the entropy of next token distribution?
            next_logits = self._forward_fn(int(cand))
            h = self._entropy(next_logits)

            if self.weight_by_prob:
                # score = p(token) * H(next) - don't pick garbage tokens
                score = candidate_probs[i] * h
            else:
                score = h

            if score > best_score:
                best_score = score
                best_token = int(cand)
                best_entropy = h

        self._entropy_history.append(best_entropy)
        return best_token

    def get_entropy_history(self) -> List[float]:
        """Returns entropy values of chosen continuations."""
        return self._entropy_history.copy()


# ----------------------------------------------------------------------------
# Delayed Chaos: Normal for N tokens, then chaos
# ----------------------------------------------------------------------------

class DelayedChaosSampler(CustomSampler):
    """
    Samples normally for N tokens, then switches to a chaos strategy.
    
    Lets the model establish a coherent beginning, then pulls the rug.
    
    Args:
        coherent_tokens: How many tokens to be normal
        chaos_strategy: 'worst', 'random', 'second_best', 'mismatch'
        temperature: Temperature for coherent phase
    """
    
    def __init__(
        self,
        coherent_tokens: int = 20,
        chaos_strategy: str = 'second_best',
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.coherent_tokens = coherent_tokens
        self.chaos_strategy = chaos_strategy
        self._step = 0
    
    def reset(self) -> None:
        self._step = 0
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        self._step += 1
        
        if self._step <= self.coherent_tokens:
            # Normal sampling
            if self.temperature <= 0:
                return int(np.argmax(logits))
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            return int(rng.choice(len(probs), p=probs))
        
        # Chaos phase!
        sorted_indices = np.argsort(logits)[::-1]
        
        if self.chaos_strategy == 'worst':
            return int(sorted_indices[-1])
        elif self.chaos_strategy == 'second_best':
            return int(sorted_indices[1]) if len(sorted_indices) > 1 else int(sorted_indices[0])
        elif self.chaos_strategy == 'mismatch':
            # Random from bottom half
            bottom_half = sorted_indices[len(sorted_indices)//2:]
            return int(rng.choice(bottom_half))
        else:  # random
            return int(rng.choice(len(logits)))


# ============================================================================
# ADVANCED / EXPERIMENTAL SAMPLERS
# ============================================================================

# ----------------------------------------------------------------------------
# Lookahead Sampler - Peek ahead before committing
# ----------------------------------------------------------------------------

class LookaheadSampler(CustomSampler):
    """
    Samples multiple candidates, simulates N tokens ahead for each,
    then picks the candidate that leads to highest cumulative probability.
    
    Requires access to the model for forward passes - uses a callback.
    
    Args:
        num_candidates: How many initial candidates to consider
        lookahead_depth: How many tokens to simulate ahead
        child_sampler: Sampler to use for lookahead simulation
        temperature: Temperature for final candidate selection
        scoring: 'log_prob', 'entropy', or 'length'
    """
    
    def __init__(
        self,
        num_candidates: int = 5,
        lookahead_depth: int = 3,
        child_sampler: Optional['CustomSampler'] = None,
        temperature: float = 0.8,
        scoring: str = 'log_prob',
        top_p: float = 1.0,
        min_p: float = 0.0
    ):
        super().__init__(temperature)
        self.num_candidates = num_candidates
        self.lookahead_depth = lookahead_depth
        self.child_sampler = child_sampler
        self.scoring = scoring
        self.top_p = top_p
        self.min_p = min_p
        
        # These will be set by the caller (LlamaDirect)
        self._forward_fn: Optional[Callable] = None
        self._current_logits: Optional[np.ndarray] = None
    
    def set_forward_fn(self, fn: Callable) -> None:
        """Set the forward function for lookahead simulation."""
        self._forward_fn = fn
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Initial candidates: apply temperature and filtering
        scaled_logits = self._apply_temperature(logits)
        probs = self._softmax(scaled_logits)
        
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        # Min P
        if self.min_p > 0:
            p_max = sorted_probs[0]
            thresh = self.min_p * p_max
            valid_mask = sorted_probs >= thresh
            sorted_indices = sorted_indices[valid_mask]
            sorted_probs = sorted_probs[valid_mask]
            
        # Top P
        if self.top_p < 1.0:
            cumsum = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumsum, self.top_p) + 1
            cutoff = min(max(cutoff, 1), len(sorted_indices))
            sorted_indices = sorted_indices[:cutoff]
            sorted_probs = sorted_probs[:cutoff]
            
        # Take up to num_candidates from valid set
        candidates = sorted_indices[:self.num_candidates]
        
        # If no forward function, fall back to simple sampling
        if self._forward_fn is None:
            # Just pick from top candidates by probability
             # (using the already computed sorted_probs)
            candidate_probs = sorted_probs[:len(candidates)]
            
            # Score by log prob (simplified without lookahead)
            candidate_probs = probs[candidates]
            candidate_probs = candidate_probs / np.sum(candidate_probs)
            idx = rng.choice(len(candidates), p=candidate_probs)
            return int(candidates[idx])
        
        # Full lookahead would require model integration
        # For now, return top candidate
        return int(np.argmax(logits))


# ----------------------------------------------------------------------------
# N-Gram Blocking Sampler - Prevent repetitive patterns
# ----------------------------------------------------------------------------

class NGramBlockingSampler(CustomSampler):
    """
    Blocks tokens that would create repeated N-grams.
    
    Much stricter than ContrastiveSampler - completely prevents
    any N-gram from appearing twice.
    
    Args:
        n: N-gram size to block (2 = bigrams, 3 = trigrams, etc.)
        temperature: Sampling temperature
        fallback_to_top_k: If all tokens blocked, sample from top K
    """
    
    def __init__(self, n: int = 3, temperature: float = 0.8, fallback_to_top_k: int = 50):
        super().__init__(temperature)
        self.n = n
        self.fallback_to_top_k = fallback_to_top_k
        self._history: List[int] = []
        self._seen_ngrams: set = set()
    
    def reset(self) -> None:
        self._history = []
        self._seen_ngrams = set()
    
    def _get_ngrams(self, tokens: List[int]) -> set:
        """Extract all N-grams from token list."""
        ngrams = set()
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            ngrams.add(ngram)
        return ngrams
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        modified_logits = logits.copy()
        
        # Check which tokens would create a repeated N-gram
        if len(self._history) >= self.n - 1:
            prefix = tuple(self._history[-(self.n - 1):])
            
            for token_id in range(len(logits)):
                potential_ngram = prefix + (token_id,)
                if potential_ngram in self._seen_ngrams:
                    modified_logits[token_id] = float('-inf')
        
        # Check if all tokens are blocked
        if np.all(np.isinf(modified_logits)):
            # Fallback to top-k from original logits
            sorted_indices = np.argsort(logits)[::-1]
            top_k = sorted_indices[:self.fallback_to_top_k]
            selected = int(rng.choice(top_k))
        else:
            # Normal sampling from non-blocked tokens
            if self.temperature <= 0:
                selected = int(np.argmax(modified_logits))
            else:
                scaled = self._apply_temperature(modified_logits)
                probs = self._softmax(scaled)
                selected = int(rng.choice(len(probs), p=probs))
        
        # Update history and N-grams
        self._history.append(selected)
        if len(self._history) >= self.n:
            new_ngram = tuple(self._history[-self.n:])
            self._seen_ngrams.add(new_ngram)
        
        return selected


# ----------------------------------------------------------------------------
# Reward Guided Sampler - External reward function biases sampling
# ----------------------------------------------------------------------------

class RewardGuidedSampler(CustomSampler):
    """
    Uses an external reward function to bias token selection.
    
    The reward function receives candidate tokens and returns scores.
    These scores are combined with logits to guide sampling.
    
    Args:
        reward_fn: Function(token_ids: List[int], logits: ndarray) -> scores: ndarray
        reward_weight: How much to weight reward vs logits (0-1)
        num_candidates: How many candidates to score
        temperature: Sampling temperature
    """
    
    def __init__(
        self,
        reward_fn: Callable[[List[int], np.ndarray], np.ndarray] = None,
        reward_weight: float = 0.5,
        num_candidates: int = 20,
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.reward_fn = reward_fn
        self.reward_weight = np.clip(reward_weight, 0, 1)
        self.num_candidates = num_candidates
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Get top candidates
        sorted_indices = np.argsort(logits)[::-1]
        candidates = sorted_indices[:self.num_candidates]
        
        if self.reward_fn is not None:
            # Get rewards for candidates
            rewards = self.reward_fn(candidates.tolist(), logits)
            rewards = np.array(rewards)
            
            # Normalize rewards to [0, 1]
            if rewards.max() > rewards.min():
                rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
            else:
                rewards = np.ones_like(rewards) * 0.5
            
            # Combine logits and rewards
            candidate_logits = logits[candidates]
            normalized_logits = self._softmax(candidate_logits)
            
            combined = (1 - self.reward_weight) * normalized_logits + self.reward_weight * rewards
            combined = combined / np.sum(combined)
            
            idx = rng.choice(len(candidates), p=combined)
        else:
            # No reward function - just sample from candidates
            candidate_logits = logits[candidates]
            probs = self._softmax(self._apply_temperature(candidate_logits))
            idx = rng.choice(len(candidates), p=probs)
        
        return int(candidates[idx])


# ----------------------------------------------------------------------------
# CFG Sampler - Classifier-Free Guidance
# ----------------------------------------------------------------------------

class CFGSampler(CustomSampler):
    """
    Classifier-Free Guidance: blend positive and negative logits.
    
    final_logits = negative_logits + scale * (positive_logits - negative_logits)
    
    Requires the caller to provide both positive and negative logits.
    
    Args:
        guidance_scale: How much to amplify the difference (1.0 = no effect, 2.0 = double)
        temperature: Sampling temperature
    """
    
    def __init__(self, guidance_scale: float = 1.5, temperature: float = 0.8):
        super().__init__(temperature)
        self.guidance_scale = guidance_scale
        self._negative_logits: Optional[np.ndarray] = None
    
    def set_negative_logits(self, logits: np.ndarray) -> None:
        """Set the negative (unconditional) logits for CFG."""
        self._negative_logits = logits
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        if self._negative_logits is not None:
            # Apply CFG formula
            cfg_logits = self._negative_logits + self.guidance_scale * (logits - self._negative_logits)
        else:
            cfg_logits = logits
        
        if self.temperature <= 0:
            return int(np.argmax(cfg_logits))
        
        scaled = self._apply_temperature(cfg_logits)
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Unique Tokens Sampler - Never repeat any token
# ----------------------------------------------------------------------------

class UniqueTokensSampler(CustomSampler):
    """
    Never generates the same token twice (within a generation).
    
    Creates very diverse but potentially incoherent output.
    
    Args:
        temperature: Sampling temperature
        exempt_tokens: Set of token IDs that CAN be repeated (e.g., punctuation)
    """
    
    def __init__(self, temperature: float = 0.8, exempt_tokens: Set[int] = None):
        super().__init__(temperature)
        self.exempt_tokens = exempt_tokens or set()
        self._used_tokens: Set[int] = set()
    
    def reset(self) -> None:
        self._used_tokens = set()
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        modified_logits = logits.copy()
        
        # Block all previously used tokens (except exempt ones)
        for token_id in self._used_tokens:
            if token_id not in self.exempt_tokens:
                if 0 <= token_id < len(modified_logits):
                    modified_logits[token_id] = float('-inf')
        
        # Check if we've run out of tokens
        if np.all(np.isinf(modified_logits)):
            # Fallback to any token
            selected = int(np.argmax(logits))
        else:
            if self.temperature <= 0:
                selected = int(np.argmax(modified_logits))
            else:
                scaled = self._apply_temperature(modified_logits)
                probs = self._softmax(scaled)
                selected = int(rng.choice(len(probs), p=probs))
        
        self._used_tokens.add(selected)
        return selected


# ----------------------------------------------------------------------------
# Copy Sampler - Randomly copies tokens from the prompt
# ----------------------------------------------------------------------------

class CopySampler(CustomSampler):
    """
    With some probability, copies a token from the original prompt.
    
    Creates outputs that echo/reference the input.
    
    Args:
        copy_prob: Probability of copying instead of normal sampling
        prompt_tokens: List of token IDs from the prompt (set by caller)
        temperature: Normal sampling temperature
    """
    
    def __init__(
        self,
        copy_prob: float = 0.2,
        prompt_tokens: List[int] = None,
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.copy_prob = np.clip(copy_prob, 0, 1)
        self.prompt_tokens = prompt_tokens or []
    
    def set_prompt_tokens(self, tokens: List[int]) -> None:
        """Set the prompt tokens to copy from."""
        self.prompt_tokens = tokens
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # Check if we should copy
        if self.prompt_tokens and rng.random() < self.copy_prob:
            return int(rng.choice(self.prompt_tokens))
        
        # Normal sampling
        if self.temperature <= 0:
            return int(np.argmax(logits))
        
        scaled = self._apply_temperature(logits)
        probs = self._softmax(scaled)
        return int(rng.choice(len(probs), p=probs))


# ----------------------------------------------------------------------------
# Competing Samplers - Two samplers compete, one wins
# ----------------------------------------------------------------------------

class CompetingSampler(CustomSampler):
    """
    Two samplers each propose a token, then we pick one.
    
    Selection can be random, by probability, or alternating.
    
    Args:
        sampler_a: First sampler
        sampler_b: Second sampler
        selection: 'random', 'prob_weighted', 'alternate', 'a_wins', 'b_wins'
        a_weight: Weight for sampler A (only for prob_weighted)
    """
    
    def __init__(
        self,
        sampler_a: CustomSampler,
        sampler_b: CustomSampler,
        selection: str = 'random',
        a_weight: float = 0.5
    ):
        super().__init__(1.0)
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        self.selection = selection
        self.a_weight = np.clip(a_weight, 0, 1)
        self._step = 0
    
    def set_vocab_size(self, vocab_size: int) -> None:
        super().set_vocab_size(vocab_size)
        self.sampler_a.set_vocab_size(vocab_size)
        self.sampler_b.set_vocab_size(vocab_size)
    
    def reset(self) -> None:
        self._step = 0
        self.sampler_a.reset()
        self.sampler_b.reset()
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        token_a = self.sampler_a(logits, rng)
        token_b = self.sampler_b(logits, rng)
        
        self._step += 1
        
        if self.selection == 'a_wins':
            return token_a
        elif self.selection == 'b_wins':
            return token_b
        elif self.selection == 'alternate':
            return token_a if self._step % 2 == 1 else token_b
        elif self.selection == 'prob_weighted':
            return token_a if rng.random() < self.a_weight else token_b
        else:  # random
            return token_a if rng.random() < 0.5 else token_b


# ----------------------------------------------------------------------------
# Typo Sampler - Introduces intentional typos
# ----------------------------------------------------------------------------

class TypoSampler(CustomSampler):
    """
    With some probability, picks a "nearby" token instead of the intended one.
    
    Simulates typos by picking tokens that are similar (by logit rank).
    
    Args:
        typo_prob: Probability of making a typo
        typo_distance: How far from the intended token to pick (1-10)
        temperature: Normal sampling temperature
    """
    
    def __init__(
        self,
        typo_prob: float = 0.1,
        typo_distance: int = 3,
        temperature: float = 0.8
    ):
        super().__init__(temperature)
        self.typo_prob = np.clip(typo_prob, 0, 1)
        self.typo_distance = max(1, typo_distance)
    
    def __call__(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        # First, determine the "intended" token
        if self.temperature <= 0:
            intended = int(np.argmax(logits))
        else:
            scaled = self._apply_temperature(logits)
            probs = self._softmax(scaled)
            intended = int(rng.choice(len(probs), p=probs))
        
        # Check if we should typo
        if rng.random() < self.typo_prob:
            # Get sorted indices and find where intended is
            sorted_indices = np.argsort(logits)[::-1]
            intended_rank = np.where(sorted_indices == intended)[0]
            
            if len(intended_rank) > 0:
                rank = intended_rank[0]
                # Pick a nearby token
                offset = rng.integers(-self.typo_distance, self.typo_distance + 1)
                new_rank = max(0, min(len(sorted_indices) - 1, rank + offset))
                return int(sorted_indices[new_rank])
        
        return intended


# ============================================================================
# Factory function for easy creation
# ============================================================================



def create_sampler(name: str, **kwargs) -> CustomSampler:
    """
    Factory function to create samplers by name.
    
    Args:
        name: Sampler name (case-insensitive)
        **kwargs: Arguments to pass to sampler constructor
    
    Returns:
        CustomSampler instance
    
    Examples:
        sampler = create_sampler("whitelist", whitelist={1,2,3}, temperature=0.8)
        sampler = create_sampler("chaos")  # Pure chaos mode
        sampler = create_sampler("personality", personality="drunk")
    """
    samplers = {
        # Core samplers
        'whitelist': WhitelistSampler,
        'second_best': SecondBestSampler,
        'nth_best': NthBestSampler,
        'random_top_k': RandomTopKSampler,
        'entropy': EntropyAwareSampler,
        'alternating': AlternatingSampler,
        'contrastive': ContrastiveSampler,
        'entropix': EntropixSampler,
        
        # Creative / Experimental samplers
        'chaos': ChaosSampler,
        'stutter': StutterSampler,
        'temperature_wave': TemperatureWaveSampler,
        'blacklist': BlacklistSampler,
        'drunken': DrunkenSampler,
        'echo': EchoSampler,
        'gradual_chaos': GradualChaosSampler,
        'mood_swing': MoodSwingSampler,
        'lucky': LuckyNumberSampler,
        'confidence': ConfidenceAdaptiveSampler,
        'bracket': BracketSampler,
        
        # Gaslighting / Deferred samplers (post-hoc override)
        'gaslight': GaslightSampler,
        'deferred_nth': DeferredNthBestSampler,
        'deferred_whitelist': DeferredWhitelistSampler,
        'substitution': SubstitutionSampler,
        'mismatch': MismatchSampler,
        'contrarian': ContrarianSampler,
        'delayed_chaos': DelayedChaosSampler,
        'breadth': BreadthSampler,

        # Advanced / Search samplers
        'lookahead': LookaheadSampler,
        'ngram_blocking': NGramBlockingSampler,
        'reward': RewardGuidedSampler,
        'cfg': CFGSampler,
        'unique': UniqueTokensSampler,
        'copy': CopySampler,
        'competing': CompetingSampler,
        'typo': TypoSampler,
    }
    
    name_lower = name.lower().replace('-', '_')
    if name_lower not in samplers:
        available = sorted(set(samplers.keys()))
        raise ValueError(f"Unknown sampler: {name}. Available: {available}")
    
    return samplers[name_lower](**kwargs)


# ============================================================================
# List all samplers for reference
# ============================================================================

ALL_SAMPLERS = [
    # Core
    "whitelist", "second_best", "nth_best", "random_top_k", "entropy_aware",
    "alternating", "contrastive", "entropix",
    # Creative
    "chaos", "stutter", "temperature_wave", "blacklist", "drunken", "echo",
    "gradual_chaos", "mood_swing", "lucky_number", "confidence_adaptive", "bracket",
    # Gaslighting / Deferred
    "gaslight", "deferred_nth_best", "deferred_whitelist", "substitution",
    "mismatch", "contrarian", "delayed_chaos", "breadth",
    # Advanced
    "lookahead", "ngram_blocking", "reward_guided", "cfg", "unique_tokens",
    "copy", "competing", "typo",
]

