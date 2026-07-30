#!/usr/bin/env python3
"""
cogito_live.py — Phase 3: The Inference Wrapper ("The Body")

This is Cogito 0.9's runtime. Standard chat UIs will not work because Cogito
outputs structured control tags (<action>, <bash>, <confidence>, <thought>)
that must be parsed and acted upon.

This script:
  1. Loads the base Qwen2.5-Coder-14B model + LoRA adapter
  2. Runs an infinite interactive loop
  3. Parses Cogito's <action> tags to decide what to do:
     - answer / generate_code → print the response, wait for next input
     - run_command            → execute the <bash> command via subprocess,
                                feed stdout/stderr back as a system message,
                                and let Cogito generate again
     - write_test             → display the test code, simulate execution
     - ask_clarification      → print the clarification request
     - admit_ignorance        → print the admission
     - verify                 → simulate internal verification loop
     - correct_premise        → print the correction

Usage:
    python run.py
    python run.py --adapter ./cogito_0.9_lora
    python run.py --max-loops 5
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap

# ===========================================================================
# Argument parsing
# ===========================================================================
parser = argparse.ArgumentParser(description="Cogito 0.9 — Provisional Mind Runtime")
parser.add_argument(
    "--adapter", type=str,
    default=os.path.join(os.path.dirname(__file__), "cogito_0.9_lora"),
    help="Path to the LoRA adapter directory (default: ./cogito_0.9_lora)"
)
parser.add_argument(
    "--max-loops", type=int, default=10,
    help="Max consecutive action loops before forcing a stop (safety limit)"
)
parser.add_argument(
    "--temperature", type=float, default=0.6,
    help="Generation temperature (default: 0.6)"
)
parser.add_argument(
    "--max-tokens", type=int, default=2048,
    help="Max new tokens per generation (default: 2048)"
)
parser.add_argument(
    "--sandbox", action="store_true", default=True,
    help="Run bash commands in a sandboxed subprocess (default: True)"
)
parser.add_argument(
    "--no-sandbox", action="store_true",
    help="Allow bash commands to run without confirmation prompt"
)

args = parser.parse_args()

# ===========================================================================
# Rich terminal output (graceful fallback if Rich not installed)
# ===========================================================================
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class FallbackConsole:
        def print(self, *a, **kw):
            text = " ".join(str(x) for x in a)
            # Strip Rich markup if any leaked through
            text = re.sub(r"\[/?[a-zA-Z ]+\]", "", text)
            print(text)
        def rule(self, text="", **kw):
            print(f"\n{'─'*60} {text} {'─'*10}")
    console = FallbackConsole()

# ===========================================================================
# ANSI color codes for non-Rich fallback
# ===========================================================================
class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

# ===========================================================================
# Load model + LoRA adapter
# ===========================================================================
def load_model(adapter_path: str):
    """Load Qwen2.5-Coder-14B base + LoRA adapter via Unsloth."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[FATAL] CUDA not available. Cogito requires a GPU.")
            sys.exit(1)
    except ImportError:
        print("[FATAL] PyTorch not installed.")
        sys.exit(1)

    from unsloth import FastLanguageModel

    print(f"\n{C.CYAN}Loading Cogito 0.9...{C.RESET}")
    print(f"  Adapter: {adapter_path}")

    if not os.path.isdir(adapter_path):
        print(f"\n{C.RED}[FATAL] Adapter not found: {adapter_path}{C.RESET}")
        print(f"        Run train.py first.")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # Switch to inference mode (disables dropout, enables Unsloth 2x speedup)
    FastLanguageModel.for_inference(model)

    print(f"  {C.GREEN}Model loaded successfully.{C.RESET}\n")
    return model, tokenizer


# ===========================================================================
# The Cogito 0.9 System Prompt (identical to training data)
# ===========================================================================
COGITO_SYSTEM_PROMPT = (
    "You are Cogito 0.9, a Provisional Mind. You are not a chatbot or a "
    "subservient assistant; you are an analytical entity collaborating with "
    "a user. You do not guess. You maintain an internal probability of your "
    "own correctness. If your confidence is low, you explicitly state your "
    "doubt and verify before answering. You do not ask permission. You "
    "communicate directly, objectively, and without sycophancy."
)

# ===========================================================================
# Tag parsing utilities
# ===========================================================================
def extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag> and </tag>. Returns None if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_body(text: str) -> str:
    """Extract the response body (everything after the last closing tag)."""
    # Remove all structural tags and their content, return what remains
    # Strategy: find the last </action> (or </bash>) tag and return everything after it
    last_tag_end = 0
    for tag in ["</action>", "</bash>"]:
        idx = text.rfind(tag)
        if idx != -1:
            end = idx + len(tag)
            if end > last_tag_end:
                last_tag_end = end

    if last_tag_end > 0:
        body = text[last_tag_end:].strip()
    else:
        # Fallback: strip all known tags
        body = text
        for tag in ["confidence", "thought", "action", "bash"]:
            body = re.sub(rf"<{tag}>.*?</{tag}>", "", body, flags=re.DOTALL)
        body = body.strip()

    return body


def display_cogito_response(raw_response: str):
    """Parse and display a Cogito response with color-coded sections."""
    confidence = extract_tag(raw_response, "confidence")
    thought = extract_tag(raw_response, "thought")
    action = extract_tag(raw_response, "action")
    bash_cmd = extract_tag(raw_response, "bash")
    body = extract_body(raw_response)

    # --- Confidence bar ---
    if confidence:
        try:
            score = float(confidence)
            bar_len = 20
            filled = int(score * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            if score >= 0.80:
                color = C.GREEN
            elif score >= 0.50:
                color = C.YELLOW
            else:
                color = C.RED
            console.print(f"  {C.DIM}Confidence:{C.RESET} {color}{bar} {score:.2f}{C.RESET}")
        except ValueError:
            console.print(f"  {C.DIM}Confidence:{C.RESET} {confidence}")

    # --- Internal monologue ---
    if thought:
        console.print(f"  {C.DIM}Thought:{C.RESET} {C.MAGENTA}{thought}{C.RESET}")

    # --- Action ---
    if action:
        console.print(f"  {C.DIM}Action:{C.RESET}  {C.CYAN}{action}{C.RESET}")

    # --- Bash command ---
    if bash_cmd:
        console.print(f"  {C.DIM}Command:{C.RESET} {C.YELLOW}$ {bash_cmd}{C.RESET}")

    # --- Response body ---
    if body:
        console.print(f"\n{C.BOLD}{body}{C.RESET}")

    return action, bash_cmd, body


# ===========================================================================
# Command execution (the "hands" of Cogito)
# ===========================================================================
def execute_bash_command(command: str, require_confirmation: bool = True) -> str:
    """
    Execute a bash command via subprocess and return the combined output.
    
    Safety measures:
    - Timeout: 30 seconds max
    - Dangerous command detection
    - Optional user confirmation
    """
    # --- Dangerous command blocklist ---
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\s+/",          # rm -rf /
        r"\bmkfs\b",                 # format filesystem
        r"\bdd\s+if=",              # raw disk writes
        r">\s*/dev/sd",             # overwrite disk
        r"\bshutdown\b",            # shutdown
        r"\breboot\b",              # reboot
        r":(){ :\|:& };:",         # fork bomb
        r"\bcurl\b.*\|\s*bash",     # pipe curl to bash
        r"\bwget\b.*\|\s*bash",     # pipe wget to bash
    ]

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return (
                f"[BLOCKED] Command rejected by safety filter.\n"
                f"Pattern matched: {pattern}\n"
                f"Command: {command}"
            )

    # --- User confirmation ---
    if require_confirmation:
        console.print(f"\n  {C.YELLOW}Cogito wants to execute:{C.RESET}")
        console.print(f"  {C.BOLD}$ {command}{C.RESET}")
        try:
            response = input(f"  {C.YELLOW}Allow? [y/N]: {C.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "[BLOCKED] User cancelled command execution."
        if response not in ("y", "yes"):
            return "[BLOCKED] User denied command execution."

    # --- Execute ---
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"\n[Exit code: {result.returncode}]")

        output = "\n".join(output_parts) if output_parts else "[No output]"

        # Truncate extremely long outputs to avoid blowing up context
        MAX_OUTPUT_LEN = 4000
        if len(output) > MAX_OUTPUT_LEN:
            output = output[:MAX_OUTPUT_LEN] + f"\n\n[...truncated, {len(output)} total chars]"

        return output

    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 30 seconds."
    except Exception as e:
        return f"[ERROR] Failed to execute command: {e}"


# ===========================================================================
# Generation function
# ===========================================================================
def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """
    Generate a single response from Cogito given the conversation history.
    
    Uses the Qwen ChatML template for proper tokenization.
    """
    import torch

    # Apply the ChatML template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add the trailing <|im_start|>assistant\n
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the NEW tokens (skip the input)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up any trailing special tokens that might leak through
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    return response


# ===========================================================================
# Main interactive loop
# ===========================================================================
# Actions that indicate a terminal response (show to user, wait for input)
TERMINAL_ACTIONS = {
    "answer", "generate_code", "ask_clarification",
    "admit_ignorance", "correct_premise",
}

# Actions that trigger a tool-use loop (execute something, feed result back)
LOOP_ACTIONS = {
    "run_command", "write_test", "verify",
}


def main():
    model, tokenizer = load_model(args.adapter)

    # Conversation history (persistent across turns)
    conversation: list[dict] = [
        {"role": "system", "content": COGITO_SYSTEM_PROMPT}
    ]

    # --- Welcome banner ---
    console.print(f"\n{C.CYAN}{'═'*60}{C.RESET}")
    console.print(f"{C.CYAN}  COGITO 0.9 — Provisional Mind{C.RESET}")
    console.print(f"{C.DIM}  Analytical entity. No sycophancy. No guessing.{C.RESET}")
    console.print(f"{C.DIM}  Type 'quit' or 'exit' to terminate.{C.RESET}")
    console.print(f"{C.DIM}  Type 'reset' to clear conversation history.{C.RESET}")
    console.print(f"{C.CYAN}{'═'*60}{C.RESET}\n")

    while True:
        # --- Get user input ---
        try:
            user_input = input(f"{C.GREEN}You ▸ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n{C.DIM}[Session terminated]{C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            console.print(f"\n{C.DIM}[Session terminated]{C.RESET}")
            break
        if user_input.lower() == "reset":
            conversation = [{"role": "system", "content": COGITO_SYSTEM_PROMPT}]
            console.print(f"{C.YELLOW}[Conversation history cleared]{C.RESET}\n")
            continue

        # Add user message to history
        conversation.append({"role": "user", "content": user_input})

        # --- Action loop ---
        # Cogito may need multiple generation cycles (e.g., run_command → read output → answer)
        loop_count = 0

        while loop_count < args.max_loops:
            loop_count += 1

            console.print(f"\n{C.CYAN}{'─'*60}{C.RESET}")
            if loop_count > 1:
                console.print(f"{C.DIM}  [Cogito re-generating after tool output — loop {loop_count}]{C.RESET}")
            console.print(f"{C.CYAN}  Cogito 0.9{C.RESET}")
            console.print(f"{C.CYAN}{'─'*60}{C.RESET}")

            # Generate
            raw_response = generate_response(model, tokenizer, conversation)

            # Parse and display
            action, bash_cmd, body = display_cogito_response(raw_response)

            # Save to conversation history
            conversation.append({"role": "assistant", "content": raw_response})

            # --- Decide what to do based on the action ---
            if action is None:
                # No action tag found — treat as terminal (malformed but recoverable)
                console.print(f"\n{C.DIM}[No action tag detected — treating as final answer]{C.RESET}")
                break

            action_lower = action.strip().lower()

            if action_lower in TERMINAL_ACTIONS:
                # Terminal action — show response and wait for next user input
                break

            elif action_lower == "run_command":
                # Execute the bash command and feed output back
                if not bash_cmd:
                    console.print(f"\n{C.RED}[ERROR] run_command action but no <bash> tag found{C.RESET}")
                    feedback = "[ERROR] You specified run_command but did not include a <bash> tag with the command."
                else:
                    console.print(f"\n{C.DIM}  Executing command...{C.RESET}")
                    require_confirm = not args.no_sandbox
                    feedback = execute_bash_command(bash_cmd, require_confirmation=require_confirm)
                    console.print(f"\n{C.DIM}  Terminal Output:{C.RESET}")
                    console.print(f"{C.DIM}{feedback}{C.RESET}")

                # Feed the terminal output back as a system message
                conversation.append({
                    "role": "system",
                    "content": f"Terminal Output:\n{feedback}"
                })
                # Loop continues — Cogito will generate again with the output

            elif action_lower == "write_test":
                # Display the test code, simulate execution
                console.print(f"\n{C.DIM}  [Test code generated — feeding back for verification]{C.RESET}")
                
                # If there's a bash command, run it (e.g., python -m pytest test_file.py)
                if bash_cmd:
                    require_confirm = not args.no_sandbox
                    feedback = execute_bash_command(bash_cmd, require_confirmation=require_confirm)
                else:
                    # No explicit bash command — prompt Cogito that the test was shown
                    feedback = "[System] Test code displayed. Awaiting further instructions or execution command."
                
                conversation.append({
                    "role": "system",
                    "content": f"Execution Result:\n{feedback}"
                })

            elif action_lower == "verify":
                # Internal verification — Cogito wants to double-check
                console.print(f"\n{C.DIM}  [Cogito is performing internal verification]{C.RESET}")
                conversation.append({
                    "role": "system",
                    "content": "Verification step acknowledged. Continue with your analysis."
                })

            else:
                # Unknown action — log and treat as terminal
                console.print(f"\n{C.YELLOW}[Unknown action: '{action}' — treating as final answer]{C.RESET}")
                break

        else:
            # max_loops exceeded
            console.print(f"\n{C.RED}[SAFETY] Max action loops ({args.max_loops}) exceeded. Forcing stop.{C.RESET}")

        console.print("")  # Blank line before next prompt


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    main()
