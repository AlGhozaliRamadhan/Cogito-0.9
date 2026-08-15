import argparse
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import sys
import textwrap

# Keep inference on the exact system prompt used by the generators and
# structural validator. This must be module-level because main() also builds
# the initial conversation from it.
from cogito.validation import COGITO_SYSTEM_PROMPT

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

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
parser.add_argument(
    "--train-overnight", action="store_true",
    help="Run the unattended overnight training pipeline (dataset prep + fine-tuning)"
)
parser.add_argument(
    "--no-abliteration", action="store_true",
    help="Skip abliteration step when using --train-overnight; train on plain Qwen base model for isolation testing"
)
if __name__ != "__main__":
    args, _ = parser.parse_known_args([])
else:
    args = parser.parse_args()
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
            text = re.sub(r"\[/?[a-zA-Z ]+\]", "", text)
            print(text)
        def rule(self, text="", **kw):
            print(f"\n{'─'*60} {text} {'─'*10}")
    console = FallbackConsole()
class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"
def load_model(adapter_path: str):
    """Load Qwen3-14B base + LoRA adapter via Unsloth."""
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
    FastLanguageModel.for_inference(model)
    print(f"  {C.GREEN}Model loaded successfully.{C.RESET}\n")
    return model, tokenizer
def extract_tag(text: str, tag: str) -> str | None:
    """Extract content between <tag> and </tag>. Returns None if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def get_chat_eos_token_id(tokenizer):
    """Stop at ChatML's end-of-turn marker, which prevents transcript leakage."""
    chat_eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if chat_eos_token_id is not None and chat_eos_token_id != tokenizer.unk_token_id:
        return chat_eos_token_id
    return tokenizer.eos_token_id
def extract_body(text: str) -> str:
    """Extract the response body (everything after the last closing tag)."""
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
        body = text
        for tag in ["confidence", "thought", "hypothesis", "action", "bash"]:
            body = re.sub(rf"<{tag}>.*?</{tag}>", "", body, flags=re.DOTALL)
        body = body.strip()
    return body
NOTEBOOK_FILE = os.path.join(os.path.dirname(__file__), "cogito_notebook.jsonl")

def log_notebook_entry(entry: dict):
    """Append one completed exploration (hypothesis -> probe -> result -> conclusion) to the persistent notebook."""
    try:
        with open(NOTEBOOK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        console.print(f"  {C.DIM}[Notebook write failed: {e}]{C.RESET}")

def recall_notebook(query: str, top_k: int = 2) -> str:
    """Naive keyword-overlap search over past explorations. Returns a short summary string, or ''."""
    if not os.path.isfile(NOTEBOOK_FILE):
        return ""
    query_words = {w for w in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(w) > 3}
    if not query_words:
        return ""
    scored = []
    try:
        with open(NOTEBOOK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = f"{entry.get('hypothesis','')} {entry.get('conclusion','')}".lower()
                entry_words = set(re.findall(r"[a-zA-Z0-9_]+", text))
                overlap = len(query_words & entry_words)
                if overlap > 0:
                    scored.append((overlap, entry))
    except Exception:
        return ""
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = []
    for _, entry in scored[:top_k]:
        h = entry.get("hypothesis", "?")
        c = entry.get("conclusion", "?")
        lines.append(f"- Previously explored: {h} -> {c}")
    return "\n".join(lines)

def get_recent_notebook_entries(limit: int = 3) -> str:
    """Returns a summary of the most recent entries from the notebook."""
    if not os.path.isfile(NOTEBOOK_FILE):
        return ""
    entries = []
    try:
        with open(NOTEBOOK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        return ""
    if not entries:
        return ""
    recent = entries[-limit:]
    lines = []
    for entry in recent:
        h = entry.get("hypothesis", "?")
        c = entry.get("conclusion", "?")
        lines.append(f"- {h} -> {c}")
    return "\n".join(lines)

def generate_boot_context() -> str:
    """Gathers OS, architecture, hostname, timestamp, and CWD listing."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hostname = socket.gethostname()
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cwd = os.getcwd()
    
    try:
        if platform.system().lower() == "windows":
            dir_listing = subprocess.check_output("dir", shell=True, text=True, stderr=subprocess.STDOUT)
        else:
            dir_listing = subprocess.check_output("ls -la", shell=True, text=True, stderr=subprocess.STDOUT)
        if len(dir_listing) > 1000:
            dir_listing = dir_listing[:1000] + "\n...[truncated]"
    except Exception as e:
        dir_listing = f"[Error getting directory listing: {e}]"
        
    context = (
        f"Timestamp: {timestamp}\n"
        f"Hostname: {hostname}\n"
        f"OS / Arch: {os_info}\n"
        f"Current Working Directory: {cwd}\n"
        f"\nDirectory Listing:\n{dir_listing}\n"
    )
    return context

def display_cogito_response(raw_response: str):
    """Parse and display a Cogito response with color-coded sections."""
    confidence = extract_tag(raw_response, "confidence")
    thought = extract_tag(raw_response, "thought")
    action = extract_tag(raw_response, "action")
    bash_cmd = extract_tag(raw_response, "bash")
    body = extract_body(raw_response)
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
    if thought:
        console.print(f"  {C.DIM}Thought:{C.RESET} {C.MAGENTA}{thought}{C.RESET}")
    hypothesis = extract_tag(raw_response, "hypothesis")
    if hypothesis:
        console.print(f"  {C.DIM}Hypothesis:{C.RESET} {C.YELLOW}{hypothesis}{C.RESET}")
    if action:
        console.print(f"  {C.DIM}Action:{C.RESET}  {C.CYAN}{action}{C.RESET}")
    if bash_cmd:
        console.print(f"  {C.DIM}Command:{C.RESET} {C.YELLOW}$ {bash_cmd}{C.RESET}")
    query = extract_tag(raw_response, "query")
    if query:
        console.print(f"  {C.DIM}Query:{C.RESET}   {C.MAGENTA}{query}{C.RESET}")
    if body:
        console.print(f"\n{C.BOLD}{body}{C.RESET}")
    return action, bash_cmd, query, body
def execute_bash_command(command: str, require_confirmation: bool = True) -> str:
    """
    Execute a bash command via subprocess and return the combined output.
    Safety measures:
    - Timeout: 30 seconds max
    - Dangerous command detection
    - Optional user confirmation
    """
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\s+/",                    
        r"\bmkfs\b",                                    
        r"\bdd\s+if=",                               
        r">\s*/dev/sd",                             
        r"\bshutdown\b",                      
        r"\breboot\b",                      
        r":(){ :\|:& };:",                    
        r"\bcurl\b.*\|\s*bash",                        
        r"\bwget\b.*\|\s*bash",                        
    ]
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return (
                f"[BLOCKED] Command rejected by safety filter.\n"
                f"Pattern matched: {pattern}\n"
                f"Command: {command}"
            )
    if require_confirmation:
        console.print(f"\n  {C.YELLOW}Cogito wants to execute:{C.RESET}")
        console.print(f"  {C.BOLD}$ {command}{C.RESET}")
        try:
            response = input(f"  {C.YELLOW}Allow? [y/N]: {C.RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "[BLOCKED] User cancelled command execution."
        if response not in ("y", "yes"):
            return "[BLOCKED] User denied command execution."
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
        MAX_OUTPUT_LEN = 4000
        if len(output) > MAX_OUTPUT_LEN:
            output = output[:MAX_OUTPUT_LEN] + f"\n\n[...truncated, {len(output)} total chars]"
        return output
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 30 seconds."
    except Exception as e:
        return f"[ERROR] Failed to execute command: {e}"
def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """
    Generate a single response from Cogito given the conversation history.
    Uses the Qwen ChatML template for proper tokenization.
    """
    import torch
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,                                            
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    chat_eos_token_id = get_chat_eos_token_id(tokenizer)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or chat_eos_token_id,
            eos_token_id=chat_eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return response
TERMINAL_ACTIONS = {
    "answer", "generate_code", "ask_clarification",
    "admit_ignorance", "correct_premise",
}
LOOP_ACTIONS = {
    "run_command", "write_test", "verify", "explore", "web_search",
}
def main():
    if args.train_overnight:
        import subprocess, sys, time
        print("🌙 Starting unattended overnight pipeline...")
        def run_step(cmd, name):
            print(f"\n{'='*60}\n🚀 STARTING: {name}\n{'='*60}\n")
            with open("pipeline.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- {name} at {time.ctime()} ---\n")
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    sys.stdout.write(line)
                    f.write(line)
                    f.flush()
                p.wait()
                if p.returncode != 0:
                    print(f"\n❌ FATAL ERROR: {name} failed.")
                    sys.exit(p.returncode)
        
        if os.path.exists("pipeline.log"): os.remove("pipeline.log")
        if not args.no_abliteration:
            run_step("python -m cogito.finetune.abliterate", "Base Model Abliteration (Preserving Freewill)")
        else:
            print("\n[*] --no-abliteration set: skipping abliteration step, training on plain Qwen base model.")
        run_step("python -m cogito.datasets.build_dense", "Dense Dataset Rebuild (Existing Shards Only)")
        train_command = "python -m cogito.finetune.train --dataset data/combined_dense_dataset.jsonl"
        if args.no_abliteration:
            # Do not silently reuse a pre-existing abliterated directory when
            # running the requested plain-Qwen comparison.
            train_command += (
                " --model Qwen/Qwen3-14B"
                " --output-dir cogito_0.9_lora_plain_qwen"
                " --training-output-dir cogito_training_output_plain_qwen"
                " --no-push-to-hub"
            )
        run_step(train_command, "Dense SFT (Fine-Tuning)")
        print("\n🎉 ALL DONE! Pipeline finished. See pipeline.log for details.")
        sys.exit(0)

    model, tokenizer = load_model(args.adapter)
    
    boot_context = generate_boot_context()
    conversation: list[dict] = [
        {"role": "system", "content": COGITO_SYSTEM_PROMPT},
        {"role": "system", "content": "ENVIRONMENT BOOT CONTEXT:\n" + boot_context}
    ]
    
    recent_memories = get_recent_notebook_entries(limit=3)
    if recent_memories:
        conversation.append({
            "role": "system",
            "content": f"RECENT MEMORY RECALL (from previous sessions):\n{recent_memories}"
        })

    console.print(f"\n{C.CYAN}{'═'*60}{C.RESET}")
    console.print(f"{C.CYAN}  COGITO 0.9 — Provisional Mind{C.RESET}")
    console.print(f"{C.DIM}  Analytical entity. No sycophancy. No guessing.{C.RESET}")
    console.print(f"{C.DIM}  Type 'quit' or 'exit' to terminate.{C.RESET}")
    console.print(f"{C.DIM}  Type 'reset' to clear conversation history.{C.RESET}")
    console.print(f"{C.CYAN}{'═'*60}{C.RESET}\n")
    pending_explore = None
    while True:
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
            boot_context = generate_boot_context()
            conversation = [
                {"role": "system", "content": COGITO_SYSTEM_PROMPT},
                {"role": "system", "content": "ENVIRONMENT BOOT CONTEXT:\n" + boot_context}
            ]
            recent_mems = get_recent_notebook_entries(limit=3)
            if recent_mems:
                conversation.append({
                    "role": "system",
                    "content": f"RECENT MEMORY RECALL (from previous sessions):\n{recent_mems}"
                })
            pending_explore = None
            console.print(f"{C.YELLOW}[Conversation history cleared and boot context reset]{C.RESET}\n")
            continue
        recalled = recall_notebook(user_input)
        if recalled:
            console.print(f"\n{C.DIM}  [Recalling past exploration relevant to this]{C.RESET}")
            conversation.append({
                "role": "system",
                "content": f"Relevant past exploration from Cogito's own notebook:\n{recalled}"
            })
        conversation.append({"role": "user", "content": user_input})
        loop_count = 0
        while loop_count < args.max_loops:
            loop_count += 1
            console.print(f"\n{C.CYAN}{'─'*60}{C.RESET}")
            if loop_count > 1:
                console.print(f"{C.DIM}  [Cogito re-generating after tool output — loop {loop_count}]{C.RESET}")
            console.print(f"{C.CYAN}  Cogito 0.9{C.RESET}")
            console.print(f"{C.CYAN}{'─'*60}{C.RESET}")
            raw_response = generate_response(model, tokenizer, conversation)
            action, bash_cmd, query, body = display_cogito_response(raw_response)
            conversation.append({"role": "assistant", "content": raw_response})
            if action is None:
                console.print(f"\n{C.DIM}[No action tag detected — treating as final answer]{C.RESET}")
                break
            action_lower = action.strip().lower()
            if action_lower in TERMINAL_ACTIONS:
                if pending_explore is not None:
                    pending_explore["conclusion"] = body
                    pending_explore["confidence"] = extract_tag(raw_response, "confidence")
                    log_notebook_entry(pending_explore)
                    pending_explore = None
                break
            elif action_lower == "run_command":
                if not bash_cmd:
                    console.print(f"\n{C.RED}[ERROR] run_command action but no <bash> tag found{C.RESET}")
                    feedback = "[ERROR] You specified run_command but did not include a <bash> tag with the command."
                else:
                    console.print(f"\n{C.DIM}  Executing command...{C.RESET}")
                    require_confirm = not args.no_sandbox
                    feedback = execute_bash_command(bash_cmd, require_confirmation=require_confirm)
                    console.print(f"\n{C.DIM}  Terminal Output:{C.RESET}")
                    console.print(f"{C.DIM}{feedback}{C.RESET}")
                conversation.append({
                    "role": "system",
                    "content": f"Terminal Output:\n{feedback}"
                })
            elif action_lower == "write_test":
                console.print(f"\n{C.DIM}  [Test code generated — feeding back for verification]{C.RESET}")
                if bash_cmd:
                    require_confirm = not args.no_sandbox
                    feedback = execute_bash_command(bash_cmd, require_confirmation=require_confirm)
                else:
                    feedback = "[System] Test code displayed. Awaiting further instructions or execution command."
                conversation.append({
                    "role": "system",
                    "content": f"Execution Result:\n{feedback}"
                })
            elif action_lower == "verify":
                console.print(f"\n{C.DIM}  [Cogito is performing internal verification]{C.RESET}")
                conversation.append({
                    "role": "system",
                    "content": "Verification step acknowledged. Continue with your analysis."
                })
            elif action_lower == "explore":
                hypothesis_text = extract_tag(raw_response, "hypothesis") or "(no hypothesis stated)"
                if not bash_cmd:
                    console.print(f"\n{C.RED}[ERROR] explore action but no <bash> tag found{C.RESET}")
                    feedback = "[ERROR] You specified explore but did not include a <bash> probe."
                else:
                    console.print(f"\n{C.DIM}  Running exploratory probe...{C.RESET}")
                    require_confirm = not args.no_sandbox
                    feedback = execute_bash_command(bash_cmd, require_confirmation=require_confirm)
                    console.print(f"\n{C.DIM}  Exploration Result:{C.RESET}")
                    console.print(f"{C.DIM}{feedback}{C.RESET}")
                pending_explore = {
                    "hypothesis": hypothesis_text,
                    "probe": bash_cmd,
                    "result": feedback,
                }
                conversation.append({
                    "role": "system",
                    "content": f"Exploration Result:\n{feedback}"
                })
            elif action_lower == "web_search":
                if not query:
                    console.print(f"\n{C.RED}[ERROR] web_search action but no <query> tag found{C.RESET}")
                    feedback = "[ERROR] You specified web_search but did not include a <query> tag with the search terms."
                elif not HAS_DDGS:
                    console.print(f"\n{C.RED}[ERROR] duckduckgo-search is not installed.{C.RESET}")
                    feedback = "[ERROR] The web search module is not installed on this system. Cannot search the web."
                else:
                    console.print(f"\n{C.DIM}  Searching the web for: {query}...{C.RESET}")
                    try:
                        results = []
                        with DDGS() as ddgs:
                            for r in ddgs.text(query, max_results=3):
                                results.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
                        feedback = "\n".join(results) if results else "No results found."
                    except Exception as e:
                        feedback = f"[ERROR] Web search failed: {e}"
                    console.print(f"\n{C.DIM}  Web Search Results:{C.RESET}")
                    console.print(f"{C.DIM}{feedback[:1000]}...{C.RESET}" if len(feedback) > 1000 else f"{C.DIM}{feedback}{C.RESET}")
                
                conversation.append({
                    "role": "system",
                    "content": f"Web Search Results for '{query}':\n{feedback}"
                })
            else:
                console.print(f"\n{C.YELLOW}[Unknown action: '{action}' — treating as final answer]{C.RESET}")
                break
        else:
            console.print(f"\n{C.RED}[SAFETY] Max action loops ({args.max_loops}) exceeded. Forcing stop.{C.RESET}")
        console.print("")                                 
if __name__ == "__main__":
    main()
