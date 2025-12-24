# Custom Samplers Reference

Token samplers for llama.cpp inference. Each takes `(logits, rng)` and returns a token ID.

---

## Core Samplers

### `WhitelistSampler`
Only sample from allowed tokens. O(whitelist_size) not O(vocab).
```python
WhitelistSampler(whitelist={1,2,3}, temperature=0.8)
```

### `SecondBestSampler`
Always picks 2nd most likely token. Creates "off" but coherent text.

### `NthBestSampler`
Pick the Nth most likely token (n=1 is greedy).
```python
NthBestSampler(n=3)  # Always 3rd best
```

### `RandomTopKSampler`
Uniform random from top K (ignores probability weights).
```python
RandomTopKSampler(k=10)
```

### `EntropyAwareSampler`
Boosts middle-probability tokens, avoids extremes.
```python
EntropyAwareSampler(entropy_weight=0.5, top_k=100)
```

### `AlternatingSampler`
Cycles through multiple samplers per token.
```python
AlternatingSampler([sampler_a, sampler_b, sampler_c])
```

### `ContrastiveSampler`
Penalizes recently generated tokens with decay.
```python
ContrastiveSampler(window_size=32, penalty=1.0, decay=0.95)
```

### `EntropixSampler`
Adaptive sampling based on entropy + varentropy:
- **LELV** (confident): greedy
- **HELV** (uncertain): temp boost + top-p
- **LEHV** (multiple good options): sample top-k
- **HEHV** (very uncertain): resample with penalty

```python
EntropixSampler(low_ent_thresh=0.3, high_ent_thresh=2.5)
```

---

## Creative / Experimental

### `ChaosSampler`
Picks the LEAST likely token. Pure chaos.
```python
ChaosSampler(nth_worst=1)  # Absolute worst token
```

### `StutterSampler`
Randomly repeats previous token.
```python
StutterSampler(stutter_prob=0.3, max_repeats=3)
```

### `TemperatureWaveSampler`
Oscillates temperature over time (sine/triangle/square).
```python
TemperatureWaveSampler(min_temp=0.3, max_temp=1.5, period=20, wave_type='sine')
```

### `BlacklistSampler`
Forbids specific tokens.
```python
BlacklistSampler(blacklist={100, 200, 300})
```

### `DrunkenSampler`
Adds Gaussian noise to logits.
```python
DrunkenSampler(noise_scale=2.0)
# or
DrunkenSampler(drunk_level=5)  # 0-10 scale
```

### `EchoSampler`
Randomly echoes tokens from earlier in generation.
```python
EchoSampler(echo_prob=0.15, lookback_range=50)
```

### `GradualChaosSampler`
Temperature ramps up over time. Starts coherent, ends unhinged.
```python
GradualChaosSampler(start_temp=0.3, end_temp=2.5, ramp_tokens=100, mode='linear')
```

### `MoodSwingSampler`
Randomly flips between greedy and chaotic per token.
```python
MoodSwingSampler(greedy_prob=0.5, chaos_temp=2.0)
```

### `LuckyNumberSampler`
Special behavior on "lucky" positions (7, 13, 42...).
```python
LuckyNumberSampler(lucky_numbers=[7,13,42], lucky_behavior='chaos')
```

### `ConfidenceAdaptiveSampler`
High confidence = low temp. Low confidence = explore more.
```python
ConfidenceAdaptiveSampler(confident_threshold=0.7, confident_temp=0.3, uncertain_temp=1.5)
```

### `BracketSampler`
Alternates sampling strategies in fixed-size "brackets".
```python
BracketSampler(bracket_size=10, strategies=[(0.5, 5), (1.0, 50), (1.5, 100)])
```

---

## Gaslighting / Post-Hoc Samplers

Let model pick freely, then override. KV cache sees overridden token.

### `GaslightSampler`
Base class - model samples normally, custom `override_fn` substitutes.
```python
GaslightSampler(override_fn=my_fn, temperature=0.8)
```

### `DeferredNthBestSampler`
Model picks best, we substitute with Nth best.
```python
DeferredNthBestSampler(n=2)  # Always force 2nd best
```

### `DeferredWhitelistSampler`
Model picks freely, we force whitelist compliance after.
```python
DeferredWhitelistSampler(whitelist={...}, substitute_strategy='highest')
```

### `SubstitutionSampler`
Token-level find/replace. If model picks X, secretly swap to Y.
```python
SubstitutionSampler(substitutions={100: 200, 300: 400})
```

### `MismatchSampler`
Always returns DIFFERENT token than model intended.
```python
MismatchSampler(mismatch_from='top3')  # Exclude top 3
```

### `ContrarianSampler`
More confident model is, more likely we override.
```python
ContrarianSampler(contrarian_scale=0.8)
```

### `DelayedChaosSampler`
Normal for N tokens, then chaos.
```python
DelayedChaosSampler(coherent_tokens=20, chaos_strategy='second_best')
```

### `BreadthSampler`
Picks token that leads to HIGHEST entropy continuation.

Anti-greedy: maximize uncertainty in next step. Keeps creative doors open.
Without `forward_fn`, falls back to weighted random from top-k.

```python
sampler = BreadthSampler(branch_k=5, weight_by_prob=True)
sampler.set_forward_fn(my_peek_fn)  # Optional: fn(token_id) -> next_logits
```

---

## Advanced Samplers

### `NGramBlockingSampler`
Completely prevents repeated N-grams.
```python
NGramBlockingSampler(n=3)  # No repeated trigrams
```

### `UniqueTokensSampler`
Never repeats any token (within generation).
```python
UniqueTokensSampler(exempt_tokens={punctuation_ids})
```

### `TypoSampler`
Introduces intentional typos (picks nearby-ranked tokens).
```python
TypoSampler(typo_prob=0.1, typo_distance=3)
```

### `CompetingSampler`
Two samplers propose, one wins.
```python
CompetingSampler(sampler_a, sampler_b, selection='random')
```

### `LookaheadSampler`
Sample candidates, simulate N tokens ahead, pick best path.
Without `forward_fn`, falls back to probability-weighted selection.
```python
LookaheadSampler(num_candidates=5, lookahead_depth=3)
```

### `RewardGuidedSampler`
External reward function biases sampling.
```python
RewardGuidedSampler(reward_fn=my_reward, reward_weight=0.5, num_candidates=20)
```

### `CFGSampler`
Classifier-Free Guidance. Blend positive + negative logits.
```python
sampler = CFGSampler(guidance_scale=1.5)
sampler.set_negative_logits(uncond_logits)
```

### `CopySampler`
Randomly copies tokens from the prompt.
```python
sampler = CopySampler(copy_prob=0.2)
sampler.set_prompt_tokens([...])
```

---

## Factory

```python
from custom_samplers import create_sampler

sampler = create_sampler("chaos")
sampler = create_sampler("whitelist", whitelist={1,2,3})
sampler = create_sampler("breadth", branch_k=5)
```
