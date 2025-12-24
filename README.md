# Custom Samplers for llama.cpp

Direct DLL access to llama.cpp with pluggable token samplers. No Python bindings needed.

## Files

- `llama_cpp_direct.py` - Direct ctypes interface to llama.dll
- `custom_samplers.py` - 30+ custom sampling strategies
- `run_sampler.py` - CLI runner
- `SAMPLERS.md` - Full sampler reference

## Setup

```bash
pip install numpy
```

Set environment variables:
```bash
export LLAMA_DLL_PATH=/path/to/llama.dll
export LLAMA_MODEL_PATH=/path/to/model.gguf
```

## Usage

```bash
# List available samplers
python run_sampler.py list

# Run with a sampler
python run_sampler.py --sampler=entropix --prompt="Once upon a time"
python run_sampler.py --sampler=chaos --tokens=50
python run_sampler.py --sampler=drunk --level=7
```

## Sampler Categories

**Core**: whitelist, nth_best, random_top_k, contrastive, entropix, entropy_aware

**Creative**: chaos, stutter, drunk, echo, madness, mood_swing, wave, lucky, confidence, bracket

**Gaslighting** (post-hoc override): deferred_nth, deferred_wl, mismatch, contrarian, delayed_chaos, breadth

**Advanced**: ngram_block, unique, typo, lookahead, reward, cfg, copy, competing

## Programmatic Use

```python
from llama_cpp_direct import LlamaDirect
from custom_samplers import create_sampler

llm = LlamaDirect(dll_path="llama.dll", model_path="model.gguf")
sampler = create_sampler("entropix")

for token_id, text in llm.generate_streaming("Hello", max_tokens=100, custom_sampler=sampler):
    print(text, end='', flush=True)
```

See `SAMPLERS.md` for full documentation.
