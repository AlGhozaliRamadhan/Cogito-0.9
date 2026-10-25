"""Interactive chat against a base model, a merged model, or base + adapter.

    python -m qwen_sft
    python -m qwen_sft --model ./merged_model
    python -m qwen_sft --adapter ./lora_adapter
    python -m qwen_sft --model ./merged_model --system "You are terse."

The old Cogito runtime parsed ``<confidence>/<thought>/<action>`` tags and could
execute shell commands. That is gone: this is a plain chat loop.
"""

import argparse
import sys

from qwen_sft.config import DEFAULT_BASE_MODEL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a Qwen3.5 model + optional LoRA")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base or merged model (repo id or local dir)")
    parser.add_argument("--adapter", default=None, help="LoRA adapter to attach on top of --model")
    parser.add_argument("--system", default="", help="System prompt (empty = none)")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-4bit", action="store_true", help="Load in bf16/fp16 (needs VRAM)")
    return parser.parse_args(argv)


def load(model_name: str, adapter: str | None, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText as _VisionModel
    except ImportError:  # pragma: no cover - transformers < 4.45
        _VisionModel = None

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(dtype=dtype, device_map="auto")
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )

    print(f"[LOAD] {model_name}" + (f" + {adapter}" if adapter else ""))
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (ValueError, KeyError, OSError):
        if _VisionModel is None:
            raise
        model = _VisionModel.from_pretrained(model_name, **kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name, trust_remote_code=True)
    return model, tokenizer


def generate(model, tokenizer, messages, args) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, tokenizer = load(args.model, args.adapter, load_in_4bit=not args.no_4bit)

    history: list[dict] = []
    if args.system.strip():
        history.append({"role": "system", "content": args.system.strip()})

    print("\nChat ready. Commands: /reset, /quit\n")
    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/reset":
            history = [message for message in history if message["role"] == "system"]
            print("[history cleared]\n")
            continue

        history.append({"role": "user", "content": user_input})
        answer = generate(model, tokenizer, history, args)
        history.append({"role": "assistant", "content": answer})
        print(f"\n{answer}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
