# Architecture

## Scope

Three stages, no more:

```
public HF dataset  ->  normalize  ->  chat JSONL  ->  LoRA SFT  ->  adapter
                                                                      |
                                                        merge -> standalone safetensors
```

## Package map

No `__init__.py` files: `qwen_sft`, `qwen_sft.datasets` and `qwen_sft.finetune`
are implicit namespace packages (PEP 420). Version lives in `config.__version__`.

```
qwen_sft/
├── __main__.py        python -m qwen_sft
├── cli.py             subcommand dispatch (chat | prepare | train | merge)
├── config.py          base model, hyperparameters, paths
├── validation.py      generic chat parse/validate/sanitize
├── runtime.py         interactive chat (base, merged, or base + adapter)
├── datasets/
│   ├── sources.py     Source registry + normalizers + prepare_source()
│   ├── prepare.py     CLI: stream a source, validate, write JSONL
│   └── merge.py       CLI: prepare all, dedupe, shuffle, write one mix
└── finetune/
    ├── train.py       Unsloth + TRL SFTTrainer, response-only loss
    └── merge.py       PEFT merge_and_unload -> safetensors (local or Hub)
```

## Layering

```
runtime  -> validation
finetune -> config, validation, datasets.sources
datasets -> config, validation
```

No module imports `torch`, `unsloth`, `trl` or `datasets` at import time. They are
imported inside the functions that need them so `import qwen_sft` stays cheap and
`--help` works on a CPU-only machine with no GPU libraries installed.

## Merge stage

`qwen_sft.datasets.merge` prepares every requested source (reusing
`data/prepared/<key>.jsonl`), hashes each conversation's messages and keeps the
first occurrence, rejects records that fail `validate_chat`, optionally caps a
source with `--max-per-source`, then shuffles with a fixed seed and writes
`data/merged/qwen_sft_mix.jsonl`. Red-team sets are not in `SOURCES`, so
`--all-sft` can never pull them in.

## Data contract

One line per conversation:

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "...", "reasoning_content": "..."}]}
```

`qwen_sft.validation.parse_messages` accepts the three shapes found in the wild:

| Shape | Example source | Handling |
| --- | --- | --- |
| `messages: [{role, content}]` | Astra, Fable | straightened; JSON-encoded strings decoded |
| `system` + `conversations: [{from, value}]` | OpenThoughts | system turn prepended, `from` aliased |
| `messages: [{role, content, info}]` | AM-DeepSeek | `info.think_content` -> `reasoning_content` |

`validate_chat` requires: >=2 turns, known roles, non-empty content, at least one
assistant turn, final turn is assistant. `sanitize` trims leading non-user and
trailing non-assistant turns before validation, so a ragged record is repaired
instead of discarded. Invalid records are counted and skipped; the run aborts
only when a source yields zero valid conversations.

## Training

1. Resolve `--dataset` values: registry key -> cached `data/prepared/<key>.jsonl`
   (prepared on first use), file path -> used directly, directory -> all `*.jsonl`.
2. Load the base with Unsloth (`FastLanguageModel`, falling back to
   `FastVisionModel` for VL checkpoints such as Qwen3.5-4B), 4-bit by default.
3. Attach LoRA to all attention and MLP projections.
4. Render each conversation with the tokenizer's **own** chat template. No
   hardcoded ChatML: Qwen3.5's template emits ` thinking` blocks and reads
   `reasoning_content`, which is why that field exists in the data contract.
5. `train_on_responses_only` masks prompt tokens; loss is computed on assistant
   turns only, so the model learns answers, not the questions.
6. Save the adapter; checkpoints keep only LoRA weights plus optimizer state.

## Why the old validator is gone

The previous pipeline required a fixed Cogito system prompt, `<confidence>` /
`<thought>` / `<action>` tags on every assistant turn, 3-10 turns, and a
sycophancy word blacklist. Third-party reasoning datasets satisfy none of that,
so the trainer refused every record. Structural validation now only checks what
SFT actually needs.
