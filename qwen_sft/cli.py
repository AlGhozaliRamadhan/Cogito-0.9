"""Single entry point.

    python -m qwen_sft                    chat with the model
    python -m qwen_sft chat --adapter X   same, explicit
    python -m qwen_sft prepare --list     dataset preparation
    python -m qwen_sft train --help      LoRA training
    python -m qwen_sft merge --help      adapter merge
    python -m qwen_sft report --help     training-curve chart
    python -m qwen_sft pretokenize --help  offline Arrow cache for DDP ranks

Bare invocation keeps the old ``python -m qwen_sft --adapter ...`` behaviour.
"""

import sys

COMMANDS = {"chat", "prepare", "pretokenize", "train", "merge", "report"}


def _dispatch(command: str, argv: list[str]) -> int:
    if command == "prepare":
        from qwen_sft.datasets.prepare import main
    elif command == "pretokenize":
        from qwen_sft.finetune.pretokenize import main
    elif command == "train":
        from qwen_sft.finetune.train import main
    elif command == "merge":
        from qwen_sft.finetune.merge import main
    elif command == "report":
        from qwen_sft.report import main
    else:
        from qwen_sft.runtime import main
    return main(argv) or 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return _dispatch(argv[0], argv[1:])
    from qwen_sft.runtime import main as runtime_main

    return runtime_main(argv) or 0


if __name__ == "__main__":
    sys.exit(main())
