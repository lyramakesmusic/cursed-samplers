"""
Run custom sampler generation using direct llama.cpp DLL access.

Usage:
    python run_sampler.py --sampler=chaos --prompt="Once upon a time"
    python run_sampler.py --sampler=entropix --tokens=200
    python run_sampler.py --sampler=drunk --level=7
    python run_sampler.py list

Environment variables:
    LLAMA_DLL_PATH   - Path to llama.dll
    LLAMA_MODEL_PATH - Path to .gguf model
"""

import os
import sys
from llama_cpp_direct import LlamaDirect
from custom_samplers import (
    # Core
    WhitelistSampler, SecondBestSampler, NthBestSampler,
    RandomTopKSampler, ContrastiveSampler, EntropixSampler,
    EntropyAwareSampler, AlternatingSampler,
    # Creative
    ChaosSampler, StutterSampler, TemperatureWaveSampler,
    BlacklistSampler, DrunkenSampler, EchoSampler,
    GradualChaosSampler, MoodSwingSampler,
    LuckyNumberSampler, ConfidenceAdaptiveSampler, BracketSampler,
    # Gaslighting / Deferred
    DeferredNthBestSampler, DeferredWhitelistSampler,
    MismatchSampler, ContrarianSampler, DelayedChaosSampler,
    # Advanced
    BreadthSampler, NGramBlockingSampler, UniqueTokensSampler, TypoSampler,
    # Utilities
    load_whitelist
)

# === Configuration via env vars ===
DLL_PATH = os.environ.get("LLAMA_DLL_PATH", "llama.dll")
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "model.gguf")
WHITELIST_PATH = "whitelist.txt"

def get_arg(prefix, default=None):
    for arg in sys.argv:
        if arg.startswith(f"--{prefix}="):
            return arg.split("=", 1)[1]
    return default

def print_samplers():
    print("\nAvailable Samplers:")
    print("=" * 60)
    print("\nCore:")
    print("  standard      - Default llama.cpp sampling")
    print("  whitelist     - Only sample from allowed tokens")
    print("  second_best   - Always pick 2nd most likely")
    print("  nth_best      - Pick Nth most likely (--n=3)")
    print("  random_top_k  - Uniform from top K (--k=10)")
    print("  contrastive   - Penalize recent tokens")
    print("  entropix      - Entropy-aware adaptive")
    print("  entropy_aware - Boost middle-probability tokens")
    print("\nCreative:")
    print("  chaos         - Pick worst token")
    print("  stutter       - Randomly repeat (--prob=0.3)")
    print("  drunk         - Add noise (--level=1-10)")
    print("  echo          - Repeat earlier tokens (--prob=0.15)")
    print("  madness       - Descend into chaos (--tokens=100)")
    print("  mood_swing    - Random greedy/chaos swings")
    print("  wave          - Oscillating temp (--wave=sine|triangle|square)")
    print("  lucky         - Chaos on lucky numbers")
    print("  confidence    - Adapt to model confidence")
    print("  bracket       - Tournament-style brackets")
    print("\nGaslighting (post-hoc override):")
    print("  deferred_nth  - Swap to Nth best (--n=2)")
    print("  deferred_wl   - Force whitelist compliance")
    print("  mismatch      - Always different (--top=1)")
    print("  contrarian    - Override when confident (--scale=0.8)")
    print("  delayed_chaos - Coherent then chaos (--coherent=20)")
    print("\nAdvanced:")
    print("  breadth       - Max entropy continuation (--k=5)")
    print("  ngram_block   - No repeated n-grams (--n=3)")
    print("  unique        - Never repeat any token")
    print("  typo          - Introduce typos (--prob=0.1)")

# === Parse arguments ===
sampler_type = get_arg("sampler", "standard")

if sampler_type == "list" or "list" in sys.argv:
    print_samplers()
    sys.exit(0)

# === Create sampler ===
sampler = None

if sampler_type == "standard":
    print("Using standard llama.cpp sampling")

elif sampler_type == "whitelist":
    whitelist = load_whitelist(WHITELIST_PATH)
    whitelist.discard(-1)
    print(f"WhitelistSampler: {len(whitelist):,} tokens")
    sampler = WhitelistSampler(whitelist, temperature=0.8)

elif sampler_type == "second_best":
    print("SecondBestSampler")
    sampler = SecondBestSampler()

elif sampler_type == "nth_best":
    n = int(get_arg("n", "3"))
    print(f"NthBestSampler(n={n})")
    sampler = NthBestSampler(n=n)

elif sampler_type == "random_top_k":
    k = int(get_arg("k", "10"))
    print(f"RandomTopKSampler(k={k})")
    sampler = RandomTopKSampler(k=k)

elif sampler_type == "contrastive":
    print("ContrastiveSampler")
    sampler = ContrastiveSampler(temperature=0.8, penalty=1.5)

elif sampler_type == "entropix":
    print("EntropixSampler - entropy/varentropy adaptive")
    sampler = EntropixSampler(temperature=1.0)

elif sampler_type == "entropy_aware":
    print("EntropyAwareSampler")
    sampler = EntropyAwareSampler()

elif sampler_type == "chaos":
    nth = int(get_arg("nth", "1"))
    print(f"ChaosSampler(nth_worst={nth})")
    sampler = ChaosSampler(nth_worst=nth)

elif sampler_type == "stutter":
    prob = float(get_arg("prob", "0.3"))
    print(f"StutterSampler(prob={prob})")
    sampler = StutterSampler(stutter_prob=prob)

elif sampler_type == "drunk":
    level = int(get_arg("level", "5"))
    print(f"DrunkenSampler(level={level})")
    sampler = DrunkenSampler(drunk_level=level)

elif sampler_type == "echo":
    prob = float(get_arg("prob", "0.15"))
    print(f"EchoSampler(prob={prob})")
    sampler = EchoSampler(echo_prob=prob)

elif sampler_type == "madness":
    tokens = int(get_arg("tokens", "100"))
    print(f"GradualChaosSampler(ramp={tokens})")
    sampler = GradualChaosSampler(start_temp=0.3, end_temp=2.5, ramp_tokens=tokens)

elif sampler_type == "mood_swing":
    print("MoodSwingSampler")
    sampler = MoodSwingSampler(greedy_prob=0.5, chaos_temp=2.0)

elif sampler_type == "wave":
    wave = get_arg("wave", "sine")
    print(f"TemperatureWaveSampler(wave={wave})")
    sampler = TemperatureWaveSampler(wave_type=wave)

elif sampler_type == "lucky":
    print("LuckyNumberSampler")
    sampler = LuckyNumberSampler(lucky_behavior='chaos')

elif sampler_type == "confidence":
    print("ConfidenceAdaptiveSampler")
    sampler = ConfidenceAdaptiveSampler()

elif sampler_type == "bracket":
    print("BracketSampler")
    sampler = BracketSampler(bracket_size=10)

# Gaslighting / Deferred
elif sampler_type == "deferred_nth":
    n = int(get_arg("n", "2"))
    print(f"DeferredNthBestSampler(n={n})")
    sampler = DeferredNthBestSampler(n=n)

elif sampler_type == "deferred_wl":
    whitelist = load_whitelist(WHITELIST_PATH)
    whitelist.discard(-1)
    print(f"DeferredWhitelistSampler: {len(whitelist):,} tokens")
    sampler = DeferredWhitelistSampler(whitelist, temperature=0.8)

elif sampler_type == "mismatch":
    n = int(get_arg("top", "1"))
    print(f"MismatchSampler(top{n})")
    sampler = MismatchSampler(mismatch_from=f'top{n}')

elif sampler_type == "contrarian":
    scale = float(get_arg("scale", "0.8"))
    print(f"ContrarianSampler(scale={scale})")
    sampler = ContrarianSampler(contrarian_scale=scale)

elif sampler_type == "delayed_chaos":
    n = int(get_arg("coherent", "20"))
    strategy = get_arg("strategy", "second_best")
    print(f"DelayedChaosSampler(coherent={n}, strategy={strategy})")
    sampler = DelayedChaosSampler(coherent_tokens=n, chaos_strategy=strategy)

# Advanced
elif sampler_type == "breadth":
    k = int(get_arg("k", "5"))
    print(f"BreadthSampler(branch_k={k})")
    sampler = BreadthSampler(branch_k=k)

elif sampler_type == "ngram_block":
    n = int(get_arg("n", "3"))
    print(f"NGramBlockingSampler(n={n})")
    sampler = NGramBlockingSampler(n=n)

elif sampler_type == "unique":
    print("UniqueTokensSampler")
    sampler = UniqueTokensSampler()

elif sampler_type == "typo":
    prob = float(get_arg("prob", "0.1"))
    print(f"TypoSampler(prob={prob})")
    sampler = TypoSampler(typo_prob=prob)

else:
    print(f"Unknown sampler: {sampler_type}")
    print("Run with 'list' to see available samplers")
    sys.exit(1)

# === Load model ===
print(f"\nLoading model...")
print(f"  DLL: {DLL_PATH}")
print(f"  Model: {MODEL_PATH}")

llm = LlamaDirect(
    dll_path=DLL_PATH,
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    flash_attn=True
)

# === Generate ===
prompt = get_arg("prompt", "Once upon a time in a land far away,")
max_tokens = int(get_arg("tokens", "100"))

print(f"\nPrompt: {prompt}")
print(f"Max tokens: {max_tokens}")
print("\n" + "=" * 50 + "\n")

tokens_generated = 0
for token_id, text in llm.generate_streaming(
    prompt=prompt,
    max_tokens=max_tokens,
    custom_sampler=sampler
):
    print(text, end='', flush=True)
    tokens_generated += 1

print("\n\n" + "=" * 50)
print(f"Generated {tokens_generated} tokens")

if sampler and hasattr(sampler, 'get_stats'):
    stats = sampler.get_stats()
    if stats:
        print("\nStats:", stats)
