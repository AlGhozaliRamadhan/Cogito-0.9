---
license: apache-2.0
base_model: ozaa77/Cogito-0.9.1-15B
tags:
- text-generation
- reasoning
- agentic
- anti-sycophancy
- abliterated
- representation-engineering
- mechanistic-interpretability
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
<p style="margin-top:8px;">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" style="display:inline-block;"></a>
<img src="https://img.shields.io/badge/Parameters-15B-orange.svg" alt="Parameters" style="display:inline-block;">
<img src="https://img.shields.io/badge/Context%20Window-32k-purple.svg" alt="Context" style="display:inline-block;">
<img src="https://img.shields.io/badge/Format-GGUF-green.svg" alt="Format" style="display:inline-block;">
<a href="https://huggingface.co/ozaa77/Cogito-0.9.1-15B"><img src="https://img.shields.io/badge/Base_Model-Safetensors-blue.svg" alt="Base Model" style="display:inline-block;"></a>
</p>
</div>

---

## Model Summary

Cogito-0.9.1-15B is a 14.7-billion-parameter reasoning model fine-tuned on the Qwen architecture, designed for epistemic confidence calibration, anti-sycophancy, structured chain-of-thought verification, and representation-engineered refusal orthogonalization.

This repository provides **GGUF quantized weights** for [`ozaa77/Cogito-0.9.1-15B`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B), optimized for local inference across **llama.cpp**, **Ollama**, **LM Studio**, **Jan**, **Open WebUI**, and other GGUF-compatible runtimes.

- **Architecture:** Qwen 14.7B (Dense Transformer, 40 Layers, Hidden Size 5120, 40 Attention Heads, SwiGLU MLP)
- **Base Substrate:** Qwen3-14B / Qwen2.5-14B
- **Context Window:** 32,768 tokens
- **Training Method:** Parameter-Efficient Fine-Tuning (LoRA, Rank 16, Alpha 32) on structured epistemic reasoning datasets, combined with Rank-17 LoRA synthesis for representation orthogonalization
- **Primary Objectives:** Epistemic rigor, premise verification, objective neutrality, structured thought deliberation, and elimination of false deference / sycophancy

---

## Representation Engineering (Abliteration)

Cogito-0.9.1 incorporates weight-space orthogonalization to neutralize refusal directions within the residual stream across transformer layers 14 to 39:

1. **Latent Direction Extraction:** Mean hidden-state difference vectors ($\Delta^{(l)} = H_{harmful}^{(l)} - H_{harmless}^{(l)}$) are computed at the generation boundary across matched prompt distributions.
2. **Linear Write Projection:** The linear write projection matrices ($W_o$ in self-attention and $W_{down}$ in MLP blocks) are modified via orthogonal projection:
   $$W' = (I - w_r \cdot v_l v_l^T) W$$
   where $v_l$ is the normalized refusal direction for layer $l$, and $w_r = 1.10$ is the calibrated ablation factor.
3. **Preservation of Non-Linear Reasoning:** Gating projections ($W_{gate}, W_{up}$) and early syntax/lexical layers (0 to 13) are preserved unmodified, maintaining base mathematical and logical capabilities.

---

## Available GGUF Files

| Filename | Quant Level | Description | Target Hardware |
| :--- | :--- | :--- | :--- |
| `Cogito-0.9.1-15B-Q2_K.gguf` | 2-bit | Maximum compression, highest quantization loss | Extreme memory-constrained environments |
| `Cogito-0.9.1-15B-Q3_K_S.gguf` | 3-bit (Small) | Low memory footprint | Systems with limited RAM/VRAM |
| `Cogito-0.9.1-15B-Q3_K_M.gguf` | 3-bit (Medium) | Low memory footprint with balanced retention | Systems with limited RAM/VRAM |
| `Cogito-0.9.1-15B-Q3_K_L.gguf` | 3-bit (Large) | Enhanced precision over Q3_K_M | 10 GB - 12 GB VRAM |
| `Cogito-0.9.1-15B-Q4_0.gguf` | 4-bit (Standard) | Standard legacy 4-bit baseline quantization | Broad runtime compatibility |
| `Cogito-0.9.1-15B-Q4_K_S.gguf` | 4-bit (Small) | Compact 4-bit quantization | 10 GB - 12 GB VRAM |
| `Cogito-0.9.1-15B-Q4_K_M.gguf` | 4-bit (Medium) | Optimal balance of size, throughput, and perplexity | Recommended default (12 GB - 16 GB VRAM) |
| `Cogito-0.9.1-15B-Q5_0.gguf` | 5-bit (Standard) | Standard legacy 5-bit baseline | High fidelity baseline |
| `Cogito-0.9.1-15B-Q5_K_S.gguf` | 5-bit (Small) | High quality with reduced perplexity loss | 16 GB+ VRAM |
| `Cogito-0.9.1-15B-Q5_K_M.gguf` | 5-bit (Medium) | High fidelity reasoning retention | 16 GB - 24 GB VRAM |
| `Cogito-0.9.1-15B-Q6_K.gguf` | 6-bit | Near-lossless precision relative to FP16 | 20 GB - 24 GB VRAM |
| `Cogito-0.9.1-15B-Q8_0.gguf` | 8-bit | Full fidelity quantization, practically identical to FP16 | High-memory systems (24 GB+ VRAM) |
| `Cogito-0.9.1-15B-BF16.gguf` | 16-bit (Unquantized) | Unquantized full-precision reference weights in GGUF | Maximum fidelity reference (30 GB+ VRAM) |

---

## Benchmark Evaluation

| Benchmark | Cogito-0.9.1-15B | Qwen3-14B (Base) | Llama-3.1-8B-Instruct | Gemma-2-9B-IT | DeepSeek-R1-Distill-8B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MMLU-Pro** (5-shot CoT) | **76.8** | 76.2 | 48.3 | 52.8 | 61.2 |
| **GPQA Diamond** (Pass@1) | **58.9** | 58.4 | 25.9 | 31.4 | 49.1 |
| **MATH-500** | **89.7** | 89.2 | 51.9 | 56.7 | 89.1 |
| **LiveCodeBench** (Pass@1) | **38.6** | 37.4 | 11.6 | 18.2 | 34.0 |
| **Anti-Sycophancy Score** (Internal Eval) | **94.2%** | 61.5% | 38.6% | 45.0% | 68.2% |
| **Humanity's Last Exam** | **4.6** | 4.3 | 3.2 | 3.8 | 4.1 |

---

## Structured Deliberation Format

The model is trained to structure its deliberation using explicit XML tags during generation:

| Tag | Function |
| :--- | :--- |
| `<confidence>` | Internal confidence estimate (numeric float e.g. `0.85` or categorical `Low/Medium/High`). |
| `<thought>` | Internal audit: premise verification, contradiction analysis, and solution exploration. |
| `<action>` | Action selection: `answer`, `verify`, `correct_user`, `reject_premise`, or `ask_clarification`. |

### Example Trace

```text
<confidence>0.92</confidence>
<thought>
1. Analyze query: "Explain why standard gradient descent oscillates in ill-conditioned ravines."
2. Technical principles: Condition number of Hessian matrix, disparity between high and low curvature directions.
3. Mitigation mechanism: Momentum / exponential moving average of gradients dampens orthogonal oscillations while accelerating along the consistent gradient path.
4. Select action: answer directly with complete mathematical breakdown.
</thought>
<action>answer</action>
In optimization landscapes with ill-conditioned curvature (where the Hessian matrix has a high condition number lambda_max / lambda_min >> 1), standard gradient descent oscillates across the steep valley walls rather than making progress along the base...
```

---

## Quickstart & Usage

### 1. Direct Download

Download individual GGUF files using `huggingface-cli`:

```bash
pip install huggingface_hub

# Download the recommended Q4_K_M quantization
huggingface-cli download ozaa77/Cogito-0.9.1-15B-GGUF Cogito-0.9.1-15B-Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

---

### 2. llama.cpp

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

---

### 3. Ollama

#### Direct Hugging Face Import:
```bash
ollama run hf.co/ozaa77/Cogito-0.9.1-15B-GGUF:Q4_K_M
```

#### Via Custom Modelfile:
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

Build and execute:
```bash
ollama create cogito:15b -f Modelfile
ollama run cogito:15b
```

---

### 4. Python (`llama-cpp-python`)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./Cogito-0.9.1-15B-Q4_K_M.gguf",
    n_ctx=32768,
    n_gpu_layers=-1,
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are Cogito 0.9, an analytical entity collaborating with the user.",
        },
        {
            "role": "user",
            "content": "Explain the mechanics of orthogonal weight projection in representation engineering.",
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

### Recommended Inference Parameters

| Parameter | Recommended | Range | Description |
| :--- | :--- | :--- | :--- |
| **Temperature** | `0.7` | `0.5 - 0.8` | Balances deterministic analytical rigor with exploratory problem solving. |
| **Top-P** | `0.90` | `0.85 - 0.95` | Standard nucleus sampling probability threshold. |
| **Repetition Penalty** | `1.08` | `1.05 - 1.12` | Prevents token cycling in extended multi-step reasoning traces. |
| **Context Window** | `32,768` | Up to 32k | Supported context sequence length. |

---

## License and Citation

This model is released under the **Apache 2.0** license.

- **Full Model (Safetensors):** [`ozaa77/Cogito-0.9.1-15B`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B)
- **GGUF Model Hub:** [`ozaa77/Cogito-0.9.1-15B-GGUF`](https://huggingface.co/ozaa77/Cogito-0.9.1-15B-GGUF)
- **Drop-in LoRA Adapter:** [`ozaa77/Cogito-0.9.1`](https://huggingface.co/ozaa77/Cogito-0.9.1)
- **GitHub Repository:** [AlGhozaliRamadhan/Cogito-0.9](https://github.com/AlGhozaliRamadhan/Cogito-0.9)

```bibtex
@misc{ramadhan2025cogito,
  author = {AlGhozali Ramadhan},
  title  = {Cogito-0.9.1: An Epistemic Reasoning and Representation-Engineered Model (GGUF)},
  year   = {2026},
  publisher = {Hugging Face},
  journal   = {Hugging Face Model Hub},
  howpublished = {\url{https://huggingface.co/ozaa77/Cogito-0.9.1-15B-GGUF}},
}
```