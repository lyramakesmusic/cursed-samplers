# custom_samplers

Custom token samplers for llama.cpp via direct DLL access (ctypes, no bindings).

## Setup

```bash
pip install numpy
```

You need:
- `llama.dll` (or `.so`/`.dylib`) from a llama.cpp build
- A `.gguf` model

Set paths via environment variables:
```bash
export LLAMA_DLL_PATH=/path/to/llama.dll
export LLAMA_MODEL_PATH=/path/to/model.gguf
```

## Run

```bash
python run_sampler.py list                    # see all samplers
python run_sampler.py --sampler=entropix      # entropy-aware adaptive
python run_sampler.py --sampler=chaos         # pick worst token
python run_sampler.py --sampler=drunk --level=7
python run_sampler.py --sampler=second_best --prompt="The answer is" --tokens=50
```

## Use in code

```python
from llama_cpp_direct import LlamaDirect
from custom_samplers import create_sampler

llm = LlamaDirect(dll_path="...", model_path="...")
sampler = create_sampler("entropix")

for token_id, text in llm.generate_streaming("Hello", max_tokens=100, custom_sampler=sampler):
    print(text, end='', flush=True)
```

## Samplers

See [SAMPLERS.md](SAMPLERS.md) for the full list.
