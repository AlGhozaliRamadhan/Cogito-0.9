"""Central defaults for the fine-tuning stack.

The HauhauCS repo is GGUF-only (no ``config.json``, no safetensors), so it cannot
be LoRA-trained. ``DEFAULT_BASE_MODEL`` is the full-precision safetensors repack
of those same uncensored weights. There is deliberately no substitute base: if
the repack disappears, training fails loudly instead of quietly fine-tuning the
official Qwen checkpoint and losing the uncensored behaviour.
"""

import os

__version__ = "1.0.0"

DEFAULT_BASE_MODEL = "DreamFast/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")
MERGED_DIR = os.path.join(PROJECT_ROOT, "data", "merged")

# Used only when --dataset is omitted. The merged mix (see
# qwen_sft.datasets.merge) is the recommended input; Fable and the Claude Code
# sessions stay opt-in because every sample is a long agentic tool loop.
DEFAULT_DATASETS = ("openthoughts", "astra")

# Measured on the merged mix: median 2.8k tokens, 56% over 2048, 44% over 4096.
# A 4B in 4-bit with gradient checkpointing fits 4096 on a 16 GB T4; drop to
# 2048 only if you actually hit OOM.
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_LIMIT_PER_DATASET = 20000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_SEED = 42

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Qwen3.5 uses a ChatML-derived template; these are the response-only markers.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
