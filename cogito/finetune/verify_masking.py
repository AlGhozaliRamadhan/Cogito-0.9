"""Debug helper: verify train_on_responses_only masking on a small model.

Loads a lightweight test model, applies the same response-only
masking as the real training run, and prints which tokens are masked vs
unmasked for the first matching example. Requires the full training stack
(unsloth, trl, transformers, datasets) and a GPU.
"""
import os


def main():
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "cogito_0.9_master_dataset.jsonl")

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-Coder-1.5B",  # Using smaller model to test quickly
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    print("Loading and mapping dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def format_example(example):
        messages = example.get("messages", [])
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": formatted_text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: "Terminal Output" in x["text"]).select([0])

    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir="./temp",
        per_device_train_batch_size=1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=4096,
        dataset_text_field="text",
        packing=False,
    )

    print("Applying train_on_responses_only...")
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    example = trainer.train_dataset[0]
    input_ids = example["input_ids"]
    labels = example["labels"]

    print("\n--- VERIFICATION OUTPUT ---")
    current_state = None
    chunk_tokens = []

    for token_id, label_id in zip(input_ids, labels):
        state = "MASKED" if label_id == -100 else "UNMASKED"
        if current_state is None:
            current_state = state

        if state != current_state:
            text = tokenizer.decode(chunk_tokens)
            print(f"[{current_state}] {repr(text)}")
            chunk_tokens = [token_id]
            current_state = state
        else:
            chunk_tokens.append(token_id)

    if chunk_tokens:
        text = tokenizer.decode(chunk_tokens)
        print(f"[{current_state}] {repr(text)}")
    print("---------------------------\n")


if __name__ == "__main__":
    main()
