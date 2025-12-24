"""
Direct ctypes wrapper for llama.cpp DLLs (b7491+ API).
Bypasses Python bindings to use the native C API directly.

Usage:
    from llama_cpp_direct import LlamaDirect
    from custom_samplers import WhitelistSampler, SecondBestSampler
    
    llm = LlamaDirect(
        dll_path=r"D:\llama-b7491-bin-win-cuda-12.4-x64\llama.dll",
        model_path=r"path\to\model.gguf",
        n_gpu_layers=-1,  # -1 = offload all
        n_ctx=8192
    )
    
    # Generate with custom sampler
    sampler = WhitelistSampler(my_whitelist_set, temperature=0.8)
    output = llm.generate(
        prompt="Hello, world!",
        max_tokens=100,
        custom_sampler=sampler
    )
    
    # Or use the second-best sampler for interesting outputs
    sampler = SecondBestSampler()
    output = llm.generate(prompt="Once upon a time", custom_sampler=sampler)
"""

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_int, c_int8, c_int32, c_uint32, c_float, c_bool, c_size_t,
    POINTER, Structure, byref, cast, CFUNCTYPE
)
import numpy as np
from pathlib import Path
from typing import Optional, Set, List, Union, Callable, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from custom_samplers import CustomSampler

# ============================================================================
# llama.cpp C API Type Definitions (b7491+)
# ============================================================================

# Opaque pointer types
llama_model_p = c_void_p
llama_context_p = c_void_p
llama_sampler_p = c_void_p
llama_vocab_p = c_void_p

# Token type (int32_t in llama.cpp)
llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32


class llama_token_data(Structure):
    """Corresponds to llama_token_data in llama.h"""
    _fields_ = [
        ("id", llama_token),      # token id
        ("logit", c_float),       # log-odds of the token
        ("p", c_float),           # probability of the token
    ]


class llama_token_data_array(Structure):
    """Corresponds to llama_token_data_array in llama.h"""
    _fields_ = [
        ("data", POINTER(llama_token_data)),
        ("size", c_size_t),
        ("selected", c_int32),
        ("sorted", c_bool),
    ]


class llama_batch(Structure):
    """Corresponds to llama_batch in llama.h"""
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(llama_pos)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(llama_seq_id))),
        ("logits", POINTER(c_int8)),
    ]


class llama_model_params(Structure):
    """Model loading parameters - matches current llama.cpp"""
    _fields_ = [
        ("devices", c_void_p),                    # ggml_backend_dev_t *
        ("tensor_buft_overrides", c_void_p),      # llama_model_tensor_buft_override *
        ("n_gpu_layers", c_int32),
        ("split_mode", c_int32),                  # enum llama_split_mode
        ("main_gpu", c_int32),
        ("tensor_split", POINTER(c_float)),
        ("progress_callback", c_void_p),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
    ]


class llama_context_params(Structure):
    """Context parameters - matches current llama.cpp"""
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("rope_scaling_type", c_int32),
        ("pooling_type", c_int32),
        ("attention_type", c_int32),
        ("flash_attn_type", c_int32),           # Changed from flash_attn bool
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", c_void_p),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_int32),
        ("type_v", c_int32),
        ("abort_callback", c_void_p),
        ("abort_callback_data", c_void_p),
        # Bools at the end
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
    ]


class llama_logit_bias(Structure):
    """Logit bias structure for llama_sampler_init_logit_bias"""
    _fields_ = [
        ("token", llama_token),
        ("bias", c_float),
    ]


# ============================================================================
# Main Wrapper Class
# ============================================================================

class LlamaDirect:
    """Direct interface to llama.cpp via ctypes (b7491+ API)."""
    
    def __init__(
        self,
        dll_path: str,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: int = None,
        flash_attn: bool = True,
        verbose: bool = True
    ):
        self.verbose = verbose
        self.model_path = model_path
        self._model = None
        self._ctx = None
        self._vocab = None
        self._lib = None
        self._ggml = None
        self._vocab_size = None
        
        # Add DLL directory to PATH for dependent DLLs
        dll_dir = str(Path(dll_path).parent)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_dir)
        
        # Load ggml.dll first and initialize backends
        ggml_path = Path(dll_path).parent / "ggml.dll"
        self._ggml = ctypes.CDLL(str(ggml_path))
        
        # Use ggml_backend_load_all_from_path to explicitly load backends from the DLL dir
        self._ggml.ggml_backend_load_all_from_path.argtypes = [c_char_p]
        self._ggml.ggml_backend_load_all_from_path.restype = None
        self._ggml.ggml_backend_load_all_from_path(dll_dir.encode('utf-8'))
        
        if verbose:
            print(f"[llama.cpp direct] Backends loaded from {dll_dir}")
        
        # Load the llama library
        self._load_library(dll_path)
        
        # Initialize llama backend
        self._lib.llama_backend_init()
        
        # Load model
        self._load_model(model_path, n_gpu_layers)
        
        # Get vocab
        self._vocab = self._lib.llama_model_get_vocab(self._model)
        
        # Create context
        self._create_context(n_ctx, n_batch, n_threads, flash_attn)
        
        # Get vocab size using vocab pointer
        self._vocab_size = self._lib.llama_vocab_n_tokens(self._vocab)
        
        if verbose:
            print(f"[llama.cpp direct] Model loaded: {Path(model_path).name}")
            print(f"[llama.cpp direct] Vocab size: {self._vocab_size}")
            print(f"[llama.cpp direct] Context size: {n_ctx}")
    
    def _load_library(self, dll_path: str):
        """Load llama.dll and set up function signatures."""
        self._lib = ctypes.CDLL(dll_path)
        
        # ====== Backend functions ======
        self._lib.llama_backend_init.argtypes = []
        self._lib.llama_backend_init.restype = None
        
        self._lib.llama_backend_free.argtypes = []
        self._lib.llama_backend_free.restype = None
        
        # ====== Model functions ======
        self._lib.llama_model_default_params.argtypes = []
        self._lib.llama_model_default_params.restype = llama_model_params
        
        self._lib.llama_load_model_from_file.argtypes = [c_char_p, llama_model_params]
        self._lib.llama_load_model_from_file.restype = llama_model_p
        
        self._lib.llama_free_model.argtypes = [llama_model_p]
        self._lib.llama_free_model.restype = None
        
        self._lib.llama_model_get_vocab.argtypes = [llama_model_p]
        self._lib.llama_model_get_vocab.restype = llama_vocab_p
        
        self._lib.llama_vocab_n_tokens.argtypes = [llama_vocab_p]
        self._lib.llama_vocab_n_tokens.restype = c_int32
        
        self._lib.llama_model_n_ctx_train.argtypes = [llama_model_p]
        self._lib.llama_model_n_ctx_train.restype = c_int32
        
        # Token functions now use vocab
        self._lib.llama_vocab_bos.argtypes = [llama_vocab_p]
        self._lib.llama_vocab_bos.restype = llama_token
        
        self._lib.llama_vocab_eos.argtypes = [llama_vocab_p]
        self._lib.llama_vocab_eos.restype = llama_token
        
        self._lib.llama_vocab_eot.argtypes = [llama_vocab_p]
        self._lib.llama_vocab_eot.restype = llama_token
        
        self._lib.llama_vocab_is_eog.argtypes = [llama_vocab_p, llama_token]
        self._lib.llama_vocab_is_eog.restype = c_bool
        
        # ====== Context functions ======
        self._lib.llama_context_default_params.argtypes = []
        self._lib.llama_context_default_params.restype = llama_context_params
        
        self._lib.llama_init_from_model.argtypes = [llama_model_p, llama_context_params]
        self._lib.llama_init_from_model.restype = llama_context_p
        
        self._lib.llama_free.argtypes = [llama_context_p]
        self._lib.llama_free.restype = None
        
        self._lib.llama_n_ctx.argtypes = [llama_context_p]
        self._lib.llama_n_ctx.restype = c_uint32
        
        # ====== Tokenization - now uses vocab instead of model ======
        self._lib.llama_tokenize.argtypes = [
            llama_vocab_p, c_char_p, c_int32,
            POINTER(llama_token), c_int32, c_bool, c_bool
        ]
        self._lib.llama_tokenize.restype = c_int32
        
        self._lib.llama_detokenize.argtypes = [
            llama_vocab_p, POINTER(llama_token), c_int32,
            c_char_p, c_int32, c_bool, c_bool
        ]
        self._lib.llama_detokenize.restype = c_int32
        
        # ====== Batch functions ======
        self._lib.llama_batch_init.argtypes = [c_int32, c_int32, c_int32]
        self._lib.llama_batch_init.restype = llama_batch
        
        self._lib.llama_batch_free.argtypes = [llama_batch]
        self._lib.llama_batch_free.restype = None
        
        # ====== Decode ======
        self._lib.llama_decode.argtypes = [llama_context_p, llama_batch]
        self._lib.llama_decode.restype = c_int32
        
        # ====== Memory/KV Cache ======
        # New API: get memory handle from context, then operate on it
        self._lib.llama_get_memory.argtypes = [llama_context_p]
        self._lib.llama_get_memory.restype = c_void_p  # llama_memory *
        
        self._lib.llama_memory_clear.argtypes = [c_void_p]  # llama_memory *
        self._lib.llama_memory_clear.restype = None
        
        # ====== Logits ======
        self._lib.llama_get_logits.argtypes = [llama_context_p]
        self._lib.llama_get_logits.restype = POINTER(c_float)
        
        self._lib.llama_get_logits_ith.argtypes = [llama_context_p, c_int32]
        self._lib.llama_get_logits_ith.restype = POINTER(c_float)
        
        # ====== Sampler chain ======
        self._lib.llama_sampler_chain_init.argtypes = [c_void_p]
        self._lib.llama_sampler_chain_init.restype = llama_sampler_p
        
        self._lib.llama_sampler_chain_add.argtypes = [llama_sampler_p, llama_sampler_p]
        self._lib.llama_sampler_chain_add.restype = None
        
        self._lib.llama_sampler_free.argtypes = [llama_sampler_p]
        self._lib.llama_sampler_free.restype = None
        
        # Individual samplers
        self._lib.llama_sampler_init_temp.argtypes = [c_float]
        self._lib.llama_sampler_init_temp.restype = llama_sampler_p
        
        self._lib.llama_sampler_init_top_k.argtypes = [c_int32]
        self._lib.llama_sampler_init_top_k.restype = llama_sampler_p
        
        self._lib.llama_sampler_init_top_p.argtypes = [c_float, c_size_t]
        self._lib.llama_sampler_init_top_p.restype = llama_sampler_p
        
        self._lib.llama_sampler_init_min_p.argtypes = [c_float, c_size_t]
        self._lib.llama_sampler_init_min_p.restype = llama_sampler_p
        
        self._lib.llama_sampler_init_dist.argtypes = [c_uint32]
        self._lib.llama_sampler_init_dist.restype = llama_sampler_p
        
        self._lib.llama_sampler_init_greedy.argtypes = []
        self._lib.llama_sampler_init_greedy.restype = llama_sampler_p
        
        # Logit bias sampler
        self._lib.llama_sampler_init_logit_bias.argtypes = [
            c_int32,  # n_vocab
            c_int32,  # n_logit_bias
            POINTER(llama_logit_bias)  # logit_bias array
        ]
        self._lib.llama_sampler_init_logit_bias.restype = llama_sampler_p
        
        # Apply sampler and sample
        self._lib.llama_sampler_sample.argtypes = [llama_sampler_p, llama_context_p, c_int32]
        self._lib.llama_sampler_sample.restype = llama_token
        
        self._lib.llama_sampler_reset.argtypes = [llama_sampler_p]
        self._lib.llama_sampler_reset.restype = None
    
    def _load_model(self, model_path: str, n_gpu_layers: int):
        """Load the GGUF model."""
        params = self._lib.llama_model_default_params()
        params.n_gpu_layers = n_gpu_layers
        
        model_path_bytes = model_path.encode('utf-8')
        self._model = self._lib.llama_load_model_from_file(model_path_bytes, params)
        
        if not self._model:
            raise RuntimeError(f"Failed to load model: {model_path}")
    
    def _create_context(self, n_ctx: int, n_batch: int, n_threads: int, flash_attn: bool):
        """Create inference context."""
        params = self._lib.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_batch
        params.n_ubatch = n_batch
        # flash_attn_type: -1=auto, 0=disabled, 1=enabled
        params.flash_attn_type = 1 if flash_attn else 0
        
        if n_threads:
            params.n_threads = n_threads
            params.n_threads_batch = n_threads
        
        # Use llama_init_from_model instead of llama_new_context_with_model
        self._ctx = self._lib.llama_init_from_model(self._model, params)
        
        if not self._ctx:
            raise RuntimeError("Failed to create context")
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def bos_token(self) -> int:
        return self._lib.llama_vocab_bos(self._vocab)
    
    @property
    def eos_token(self) -> int:
        return self._lib.llama_vocab_eos(self._vocab)
    
    @property
    def eot_token(self) -> int:
        return self._lib.llama_vocab_eot(self._vocab)
    
    def is_eog(self, token: int) -> bool:
        """Check if token is end-of-generation."""
        return self._lib.llama_vocab_is_eog(self._vocab, token)
    
    def tokenize(self, text: str, add_bos: bool = True) -> List[int]:
        """Tokenize text to token IDs."""
        text_bytes = text.encode('utf-8')
        max_tokens = len(text_bytes) + 16
        
        tokens = (llama_token * max_tokens)()
        n_tokens = self._lib.llama_tokenize(
            self._vocab,  # Use vocab instead of model
            text_bytes,
            len(text_bytes),
            tokens,
            max_tokens,
            add_bos,
            True  # parse_special
        )
        
        if n_tokens < 0:
            max_tokens = -n_tokens
            tokens = (llama_token * max_tokens)()
            n_tokens = self._lib.llama_tokenize(
                self._vocab,
                text_bytes,
                len(text_bytes),
                tokens,
                max_tokens,
                add_bos,
                True
            )
        
        return list(tokens[:n_tokens])
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs to text using llama_detokenize."""
        if not tokens:
            return ""
        
        # Create token array
        n_tokens = len(tokens)
        token_array = (llama_token * n_tokens)(*tokens)
        
        # Allocate buffer for output
        buf_size = n_tokens * 16  # Conservative estimate
        buf = (ctypes.c_char * buf_size)()
        
        # Call llama_detokenize
        n_chars = self._lib.llama_detokenize(
            self._vocab,
            token_array,
            n_tokens,
            buf,
            buf_size,
            False,  # remove_special
            False   # unparse_special
        )
        
        if n_chars < 0:
            # Buffer too small, retry
            buf_size = -n_chars
            buf = (ctypes.c_char * buf_size)()
            n_chars = self._lib.llama_detokenize(
                self._vocab,
                token_array,
                n_tokens,
                buf,
                buf_size,
                False,
                False
            )
        
        return buf[:n_chars].decode('utf-8', errors='replace')
    
    def get_logits(self, idx: int = -1) -> np.ndarray:
        """Get logits for position idx (default: last position)."""
        logits_ptr = self._lib.llama_get_logits_ith(self._ctx, idx)
        return np.ctypeslib.as_array(logits_ptr, shape=(self._vocab_size,)).copy()
    
    # ==========================================================================
    # Custom sampling is now handled by CustomSampler classes in custom_samplers.py
    # See: WhitelistSampler, SecondBestSampler, NthBestSampler, etc.
    # ==========================================================================
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        custom_sampler: Optional["CustomSampler"] = None,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        seed: int = None,
        stop_tokens: Optional[List[int]] = None,
        callback: Optional[Callable[[int, str], bool]] = None,
        # Deprecated - use custom_sampler instead
        whitelist: Optional[Set[int]] = None
    ) -> str:
        """
        Generate text with optional custom sampler.
        
        When custom_sampler is provided, it receives raw logits each step
        and returns the next token ID. This enables creative sampling
        strategies like whitelist constraints, second-best selection, etc.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            custom_sampler: CustomSampler instance (from custom_samplers.py)
            temperature: Sampling temperature (used without custom_sampler)
            top_k: Top-K sampling (used without custom_sampler)
            top_p: Top-P (nucleus) sampling (used without custom_sampler)
            min_p: Min-P sampling (used without custom_sampler)
            seed: Random seed (None = random)
            stop_tokens: Additional stop tokens
            callback: Optional callback(token_id, text) -> bool, return False to stop
            whitelist: DEPRECATED - use WhitelistSampler from custom_samplers instead
        
        Returns:
            Generated text
        """
        import random
        import warnings
        
        # Handle deprecated whitelist parameter
        if whitelist is not None:
            if custom_sampler is not None:
                raise ValueError("Cannot specify both custom_sampler and whitelist")
            warnings.warn(
                "whitelist parameter is deprecated. Use WhitelistSampler from custom_samplers instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Import and create WhitelistSampler for backward compatibility
            from custom_samplers import WhitelistSampler
            custom_sampler = WhitelistSampler(whitelist, temperature=temperature)
        
        # Clear memory (KV cache) - need to get memory handle from context first
        memory = self._lib.llama_get_memory(self._ctx)
        if memory:
            self._lib.llama_memory_clear(memory)
        
        # Tokenize prompt
        tokens = self.tokenize(prompt, add_bos=True)
        
        # Setup stop tokens
        stop_set = {self.eos_token}
        eot = self.eot_token
        if eot >= 0:
            stop_set.add(eot)
        if stop_tokens:
            stop_set.update(stop_tokens)
        
        # Initialize custom sampler if provided
        use_custom_sampling = custom_sampler is not None
        if use_custom_sampling:
            custom_sampler.set_vocab_size(self._vocab_size)
            # If sampler supports excluding tokens (e.g., WhitelistSampler), exclude stop tokens
            if hasattr(custom_sampler, 'exclude_tokens'):
                custom_sampler.exclude_tokens(stop_set)
            custom_sampler.reset()
            if self.verbose:
                print(f"[llama.cpp direct] Using custom sampler: {type(custom_sampler).__name__}")
        
        # Create batch
        batch = self._lib.llama_batch_init(len(tokens) + max_tokens, 0, 1)
        
        # Setup RNG
        if seed is None:
            seed = random.randint(0, 2**63 - 1)
        rng = np.random.default_rng(seed)
        
        # Create sampler chain (only used when no custom sampler)
        sampler = None
        if not use_custom_sampling:
            sampler = self._lib.llama_sampler_chain_init(None)
            if temperature > 0:
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_top_k(top_k))
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_top_p(top_p, 1))
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_min_p(min_p, 1))
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_temp(temperature))
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_dist(seed & 0xFFFFFFFF))
            else:
                self._lib.llama_sampler_chain_add(sampler, self._lib.llama_sampler_init_greedy())
        
        generated_tokens = []
        
        try:
            # Process prompt
            batch.n_tokens = len(tokens)
            for i, tok in enumerate(tokens):
                batch.token[i] = tok
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 0
            batch.logits[len(tokens) - 1] = 1  # Compute logits for last token
            
            if self._lib.llama_decode(self._ctx, batch) != 0:
                raise RuntimeError("Failed to decode prompt")
            
            n_cur = len(tokens)
            
            # Generate tokens
            for step in range(max_tokens):
                if use_custom_sampling:
                    # Custom sampler - get logits and let sampler decide
                    logits = self.get_logits(-1)
                    new_token = custom_sampler(logits, rng)
                else:
                    # Standard sampler chain
                    new_token = self._lib.llama_sampler_sample(sampler, self._ctx, -1)
                
                # Debug: show what was sampled
                if self.verbose and step < 5:
                    print(f"[DEBUG] Step {step}: sampled token {new_token}, is_eog={self.is_eog(new_token)}, in_stop_set={new_token in stop_set}")
                
                # Check for stop
                if new_token in stop_set or self.is_eog(new_token):
                    if self.verbose:
                        print(f"[DEBUG] Stopping: token {new_token}, is_eog={self.is_eog(new_token)}, in_stop_set={new_token in stop_set}")
                    break
                
                generated_tokens.append(new_token)
                
                # Callback
                if callback:
                    token_text = self.detokenize([new_token])
                    if callback(new_token, token_text) is False:
                        break
                
                # Prepare next batch
                batch.n_tokens = 1
                batch.token[0] = new_token
                batch.pos[0] = n_cur
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = 0
                batch.logits[0] = 1
                
                if self._lib.llama_decode(self._ctx, batch) != 0:
                    raise RuntimeError("Failed to decode")
                
                n_cur += 1
                
                # Reset sampler if using standard chain
                if sampler:
                    self._lib.llama_sampler_reset(sampler)
        
        finally:
            if sampler:
                self._lib.llama_sampler_free(sampler)
            self._lib.llama_batch_free(batch)
        
        return self.detokenize(generated_tokens)
    
    def generate_streaming(
        self,
        prompt: str,
        max_tokens: int = 256,
        custom_sampler: Optional["CustomSampler"] = None,
        **kwargs
    ):
        """
        Generator that yields (token_id, text) tuples.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            custom_sampler: Optional CustomSampler instance
            **kwargs: Additional arguments passed to generate()
        
        Yields:
            Tuples of (token_id, decoded_text)
        """
        tokens_generated = []
        
        def collect_callback(token_id, text):
            tokens_generated.append((token_id, text))
            return True
        
        self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            custom_sampler=custom_sampler,
            callback=collect_callback,
            **kwargs
        )
        
        for token_id, text in tokens_generated:
            yield token_id, text
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.llama_free(self._ctx)
        if hasattr(self, '_model') and self._model:
            self._lib.llama_free_model(self._model)
        if hasattr(self, '_lib') and self._lib:
            self._lib.llama_backend_free()
    
    def close(self):
        """Explicitly free resources."""
        self.__del__()
        self._ctx = None
        self._model = None


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    from custom_samplers import WhitelistSampler, SecondBestSampler, NthBestSampler
    
    DLL_PATH = r"D:\llama-b7491-bin-win-cuda-12.4-x64\llama.dll"
    MODEL_PATH = r"D:\trinity-mini-base-pre-anneal-q4_k_m.gguf"
    
    print("Loading llama.cpp directly via ctypes...")
    
    try:
        llm = LlamaDirect(
            dll_path=DLL_PATH,
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096
        )
        
        # Demo 1: Whitelist sampler (only allow top 1000 tokens)
        print("\n--- Generating with WhitelistSampler ---")
        whitelist_sampler = WhitelistSampler(set(range(1000)), temperature=0.7)
        output = llm.generate(
            prompt="The quick brown fox",
            max_tokens=50,
            custom_sampler=whitelist_sampler
        )
        print(f"Output: {output}")
        
        # Demo 2: Second-best sampler (always pick 2nd most likely)
        print("\n--- Generating with SecondBestSampler ---")
        second_best = SecondBestSampler()
        output = llm.generate(
            prompt="Once upon a time",
            max_tokens=50,
            custom_sampler=second_best
        )
        print(f"Output: {output}")
        
        # Demo 3: Third-best sampler
        print("\n--- Generating with NthBestSampler(n=3) ---")
        third_best = NthBestSampler(n=3)
        output = llm.generate(
            prompt="The meaning of life is",
            max_tokens=50,
            custom_sampler=third_best
        )
        print(f"Output: {output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
