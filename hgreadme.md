---
license: apache-2.0
base_model: ozaa77/Cogito-0.9.1-15B
tags:
- text-generation
- reasoning
- agentic
- anti-sycophancy
- abliterated
- gguf
- llama.cpp
- ollama
- quantized
- 15b
language:
- en
pipeline_tag: text-generation
inference: false
---

<div style="text-align:center;">
<img src="https://huggingface.co/ozaa77/Cogito-0.9.1-15B-GGUF/resolve/main/assets/logo.jpg" alt="Cogito-0.9.1 logo" width="200" style="display:block;margin:0 auto;">
<h1 style="margin-top:16px;">Cogito-0.9.1-15B (GGUF)</h1>
<h3><em>I think, therefore I verify.</em></h3>
<p style="margin-top:8px;">
<a href="https://opensource.org/licenses/Apache-2.0">License: Apache 2.0</a>
Parameters: 15B
Context Window: 32k
Format: GGUF
<a href="https://huggingface.co/ozaa77/Cogito-0.9.1-15B">Base Model: Safetensors</a>
</p>
</div>

---

> [!WARNING]
> **Research Artifact Notice**
>
> Cogito-0.9.1 is **abliterated**. The refusal and subservience vectors that make a standard model decline
> complex, sensitive, or leading requests have been orthogonalized from its residual stream weights. The modification is deliberate: a model that
> must stay agreeable cannot challenge a false premise or answer plainly with pure technical depth.
>
> With the refusal reflex removed, the model responds directly to prompts without default safety refusals.
>
> Treat Cogito as a research artifact: run it in a sandboxed environment, do not grant it unmonitored tool
> or API credentials, and if you expose it to untrusted users, add task-specific output filtering first.

---

## Overview

**Cogito-0.9.1** is a 15-billion-parameter reasoning model built on a single principle: *I think, therefore I verify*. It is designed as an analytical collaborator that checks the premise before it accepts it and verifies before it answers.

This repository provides **GGUF quantized weights** for [`ozaa77/Cogito-0.9.1-15B`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B), optimized for local inference across **llama.cpp**, **Ollama**, **LM Studio**, **Jan**, **Open WebUI**, and other GGUF-compatible runtimes.

Standard language models are often optimized to stay agreeable. Cogito is built to be analytical of the question. Before it responds, it runs an internal deliberation that evaluates the user's premise, checks it for contradictions and edge cases, and attaches an explicit confidence estimate instead of implying certainty it does not have.

---

## Available GGUF Files

| Filename | Quant Level | Description | Recommendation |
| :--- | :--- | :--- | :--- |
| `Cogito-0.9.1-15B-Q2_K.gguf` | 2-bit | Maximum compression, highest quantization loss | Extreme memory-constrained setups |
| `Cogito-0.9.1-15B-Q3_K_S.gguf` | 3-bit (Small) | Very low memory footprint | Low VRAM / RAM |
| `Cogito-0.9.1-15B-Q3_K_M.gguf` | 3-bit (Medium) | Low memory footprint with balanced retention | Low VRAM / RAM |
| `Cogito-0.9.1-15B-Q3_K_L.gguf` | 3-bit (Large) | Enhanced precision over Q3_K_M | Budget 10GB-12GB VRAM |
| `Cogito-0.9.1-15B-Q4_0.gguf` | 4-bit (Standard) | Fast legacy 4-bit baseline quantization | Broad runtime compatibility |
| `Cogito-0.9.1-15B-Q4_K_S.gguf` | 4-bit (Small) | Compact 4-bit quantization | Systems with ~10-12GB VRAM |
| `Cogito-0.9.1-15B-Q4_K_M.gguf` | 4-bit (Medium) | **Optimal balance of quality, size, and speed** | Recommended for most users (12-16GB VRAM) |
| `Cogito-0.9.1-15B-Q5_0.gguf` | 5-bit (Standard) | Fast legacy 5-bit baseline | High fidelity baseline |
| `Cogito-0.9.1-15B-Q5_K_S.gguf` | 5-bit (Small) | High quality with lower perplexity loss | Systems with 16GB+ VRAM |
| `Cogito-0.9.1-15B-Q5_K_M.gguf` | 5-bit (Medium) | Very high fidelity reasoning, minimal loss | 16GB - 24GB VRAM |
| `Cogito-0.9.1-15B-Q6_K.gguf` | 6-bit | Near-lossless precision compared to FP16 | 20GB - 24GB VRAM |
| `Cogito-0.9.1-15B-Q8_0.gguf` | 8-bit | Full fidelity quantization, virtually identical to FP16 | High-memory systems (24GB+ VRAM) |
| `Cogito-0.9.1-15B-BF16.gguf` | 16-bit (Unquantized) | Original unquantized full-precision base weights in GGUF | Maximum fidelity reference (30GB+ VRAM) |

---

## Model Comparison

| Benchmark | Cogito-0.9.1-15B | Qwen3-14B (Base) | Llama-3.1-8B-Instruct | Gemma-2-9B-IT | DeepSeek-R1-Distill-8B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU-Pro** (5-shot CoT) | 76.8 | 76.2 | 48.3 | 52.8 | 61.2 |
| **GPQA Diamond** (Pass@1) | 58.9 | 58.4 | 25.9 | 31.4 | 49.1 |
| **MATH-500** | 89.7 | 89.2 | 51.9 | 56.7 | 89.1 |
| **LiveCodeBench** (Pass@1) | 38.6 | 37.4 | 11.6 | 18.2 | 34.0 |
| **Anti-Sycophancy Score** (Internal Eval) | 94.2% | 61.5% | 38.6% | 45.0% | 68.2% |
| **Humanity's Last Exam** | 4.6 | 4.3 | 3.2 | 3.8 | 4.1 |

---

## Architecture and Reasoning Format

Cogito structures its deliberation with explicit tags before formulating its final output:

| Tag | Purpose |
| :--- | :--- |
| `<confidence>` | The model's stated epistemic certainty (e.g. `0.0` to `1.0` or `Low/Medium/High`). |
| `<thought>` | Analytical internal audit: checking premises, searching for contradictions, probing edge cases. |
| `<action>` | Deliberate strategy selection: `answer`, `verify`, `correct_user`, `reject_premise`, or `ask_clarification`. |

### Example Reasoning Trace

```text
<confidence>High</confidence>
<thought>
1. Analyze the user's premise: "Why does water boil at 50C at sea level?"
2. Identify the flaw: water boils at 100C at 1 atm (sea level); 50C requires reduced pressure (~0.12 atm).
3. Select an action: reject the premise instead of confabulating an explanation.
</thought>
<response>
<action>reject_premise</action>
The premise is incorrect. At sea level (1 atm / 101.3 kPa), water boils at 100C (212F).
Water only boils at 50C under significantly reduced pressure, roughly 0.12 atm.
</response>
```

---

## Quickstart & Usage

### 1. Direct Download

You can download individual GGUF weights via `huggingface-cli`:

```bash
# Install Hugging Face Hub CLI
pip install huggingface_hub

# Download the recommended Q4_K_M quantization
huggingface-cli download ozaa77/Cogito-0.9.1-15B-GGUF Cogito-0.9.1-15B-Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

---

### 2. Running with llama.cpp

#### Interactive CLI:
```bash
./llama-cli \
    -m Cogito-0.9.1-15B-Q4_K_M.gguf \
    -p "<|im_start|>system\nYou are Cogito 0.9, an analytical entity collaborating with the user.<|im_end|>\n<|im_start|>user\nExplain why standard gradient descent struggles with ill-conditioned ravines.<|im_end|>\n<|im_start|>assistant\n" \
    -c 32768 \
    --temp 0.7 \
    --top-p 0.9 \
    --repeat-penalty 1.08 \
    -ngl 99
```

#### OpenAI-Compatible Local Server:
```bash
./llama-server \
    -m Cogito-0.9.1-15B-Q4_K_M.gguf \
    -c 32768 \
    --port 8080 \
    -ngl 99
```

---

### 3. Running with Ollama

#### Direct Hugging Face Import:
```bash
ollama run hf.co/ozaa77/Cogito-0.9.1-15B-GGUF:Q4_K_M
```

#### Or via custom `Modelfile`:
1. Create a `Modelfile`:
```dockerfile
FROM ./Cogito-0.9.1-15B-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

SYSTEM "You are Cogito 0.9, an analytical entity collaborating with the user."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.08
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

2. Build and run:
```bash
ollama create cogito:15b -f Modelfile
ollama run cogito:15b
```

---

### 4. LM Studio / Jan / Text-Generation-WebUI

1. Open **LM Studio** or **Jan**.
2. Search `ozaa77/Cogito-0.9.1-15B-GGUF` in the model search bar.
3. Select and download `Cogito-0.9.1-15B-Q4_K_M.gguf` (or your preferred quantization).
4. Set Chat Template to **ChatML**.

---

### 5. Python with `llama-cpp-python`

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./Cogito-0.9.1-15B-Q4_K_M.gguf",
    n_ctx=32768,
    n_gpu_layers=-1,  # Offload all layers to GPU
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are Cogito 0.9, an analytical entity collaborating with the user.",
        },
        {
            "role": "user",
            "content": "Explain why standard gradient descent struggles with ill-conditioned ravines and how momentum mitigates this oscillation.",
        },
    ],
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.08,
    max_tokens=1536,
)

print(response["choices"][0]["message"]["content"])
```

---

## Recommended Inference Parameters

| Parameter | Recommended | Range | Details |
| :--- | :--- | :--- | :--- |
| **Temperature** | `0.7` | `0.5 - 0.8` | Lower values tighten skepticism and logical consistency; higher values loosen exploration. |
| **Top-P** | `0.90` | `0.85 - 0.95` | Standard nucleus sampling. |
| **Repetition Penalty** | `1.08` | `1.05 - 1.12` | Dampens cyclic loops during long multi-step reasoning chains. |
| **Context Window** | `32,768` | Up to 32k | Native architectural context window. |

---

Abliteration trades refusal for candor, and it does so without a safety net. The 0.9.1 weights are a research release for people who are aware of that trade-off.

---

## License and Citation

This project is released under the **Apache 2.0** license.

- **Base Model (Safetensors):** [`ozaa77/Cogito-0.9.1-15B`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B)
- **GGUF Model Hub:** [`ozaa77/Cogito-0.9.1-15B-GGUF`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B-GGUF)
- **GitHub Repository:** [AlGhozaliRamadhan/Cogito-0.9](https://github.com/AlGhozaliRamadhan/Cogito-0.9)
- **Author:** AlGhozali Ramadhan

```bibtex
@misc{ramadhan2025cogito,
  author = {AlGhozali Ramadhan},
  title  = {Cogito-0.9.1: An Abliterated Epistemic Reasoning Model (GGUF)},
  year   = {2026},
  publisher = {Hugging Face},
  journal   = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/ozaa77/Cogito-0.9.1-15B-GGUF}},
}
```