"""Registry of Hugging Face SFT sources.

Each entry knows how to load its dataset and how to turn one raw record into
``[{"role", "content", "reasoning_content"?}, ...]``. Everything is normalized
to chat messages so the trainer never has to care where the data came from.

Schemas verified against the datasets-server first-rows API:

- Fable       : ``messages`` (list, stored as JSON) + ``tools``
- Astra       : ``messages`` [{role, content}]
- OpenThoughts: ``system`` + ``conversations`` [{from, value}]
- AM-DeepSeek : ``messages`` [{role, content, info{think_content, answer_content}}]
"""

import ast
import json
import os
import random
from dataclasses import dataclass

from qwen_sft.validation import (
    canonical_role,
    flatten_content,
    parse_messages,
    sanitize,
    validate_chat,
)


@dataclass(frozen=True)
class Source:
    key: str
    repo_id: str
    normalizer: object
    description: str
    license: str = "see dataset card"
    config: str | None = None
    split: str = "train"
    note: str = ""
    data_files: str | None = None
    raw_jsonl: str | None = None

    def normalize(self, record) -> list[dict]:
        return self.normalizer(record)


def _plain(record) -> list[dict]:
    return parse_messages(record)


def _decode_instruction(value):
    """Decode an ``instruction`` cell that may be a dict, JSON, or a Python repr.

    The saidutta69 Parquet stores it as a Python ``repr`` with single quotes,
    which ``json.loads`` rejects. Without this second attempt the whole dict
    string lands in a single user turn.
    """
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        decoded = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return decoded if isinstance(decoded, (dict, list)) else None


def _instruction_response(record) -> list[dict]:
    """``{instruction: {messages: [...]}, response: str}`` distillation rows.

    Used by the Manusagents set and its cleaned repack by saidutta69.
    """
    raw_instruction = record.get("instruction")
    instruction = _decode_instruction(raw_instruction)

    if isinstance(instruction, dict):
        turns = instruction.get("messages")
        if turns is None and instruction.get("role"):
            turns = [instruction]
        messages = parse_messages({"messages": turns or []})
    elif isinstance(instruction, list):
        messages = parse_messages({"messages": instruction})
    elif (isinstance(raw_instruction, str) and raw_instruction.strip()
          and not raw_instruction.lstrip().startswith(("{", "["))):
        # These shards store a plain instruction string, not a message list.
        messages = [{"role": "user", "content": raw_instruction.strip()}]
    else:
        # A brace/bracket-leading string that would not decode is a broken
        # struct repr; dropping it beats training on the repr itself, and an
        # answer with no prompt is untrainable.
        return []

    response = record.get("response")
    if isinstance(response, dict):
        response = flatten_content(response)
    if isinstance(response, str) and response.strip():
        messages.append({"role": "assistant", "content": response})
    return messages


def _nested_messages(key: str):
    """Some datasets keep the turns in a differently named column."""

    def normalize(record) -> list[dict]:
        return parse_messages({"messages": record.get(key) or []})

    return normalize


def _deepseek_r1(record) -> list[dict]:
    """AM-DeepSeek keeps the reasoning trace in ``info``, not in content."""
    raw = record.get("messages") if isinstance(record, dict) else record
    if isinstance(raw, str):
        raw = json.loads(raw)
    messages = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = canonical_role(item.get("role"))
        if role is None:
            continue
        content = flatten_content(item.get("content"))
        info = item.get("info") if isinstance(item.get("info"), dict) else {}
        message = {"role": role, "content": content}
        if role == "assistant":
            if not content.strip():
                message["content"] = flatten_content(info.get("answer_content"))
            think = info.get("think_content")
            if isinstance(think, str) and think.strip():
                message["reasoning_content"] = think
        if message["content"].strip() or message.get("reasoning_content"):
            messages.append(message)
    return messages


SOURCES: dict[str, Source] = {
    "fable": Source(
        key="fable",
        repo_id="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        normalizer=_plain,
        description="10k Fable 5.1 reasoning traces, multi-turn agentic, with tool schemas",
        license="see dataset card",
        note="Agentic tool-use traces. Tool definitions are dropped, so assistant "
             "turns that only contain tool calls are skipped. The curated `train` "
             "split is 5k rows; the 10k `full` split yields far fewer valid conversations.",
    ),
    "astra": Source(
        key="astra",
        repo_id="guell00/Astra-COT-40K",
        normalizer=_plain,
        description="40k chain-of-thought + instruction-following conversational samples",
        license="see dataset card",
        note="Already plain chat messages; long single-turn answers.",
    ),
    "openthoughts": Source(
        key="openthoughts",
        repo_id="open-thoughts/OpenThoughts-114k",
        normalizer=_plain,
        description="114k verified DeepSeek-R1 reasoning traces (thought/solution markup)",
        license="see dataset card",
        note="Assistant turns carry <|begin_of_thought|> markup; that is the native format.",
    ),
    "am": Source(
        key="am",
        repo_id="a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
        normalizer=_deepseek_r1,
        description="~1.4M R1-distilled reasoning problems (am_0.5M + am_0.9M)",
        license="CC-BY-NC-4.0 (non-commercial)",
        config="am_0.9M",
        note="Reasoning lives in info.think_content and is mapped to reasoning_content. "
             "Non-commercial licence. Use am_0.5M or am_0.9M_sample_1k to sample cheaply.",
    ),
    "fable_code": Source(
        key="fable_code",
        repo_id="armand0e/claude-fable-5-claude-code",
        normalizer=_plain,
        description="75 MB of Claude Code sessions (user prompt -> tool loop -> answer)",
        note="Uses the Hub's parquet conversion: the raw files are per-event logs with "
             "differing schemas, while the conversion has assembled `messages`. "
             "Sessions run to hundreds of turns, so use --max-seq-length 4096.",
        data_files="https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    ),
    "distill_multi": Source(
        key="distill_multi",
        repo_id="Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset",
        normalizer=_instruction_response,
        description="11.5 GB multi-teacher distillation set (instruction.messages + response)",
        note="Row layout is {instruction: {messages}, response}. Sharded JSONL, so keep "
             "the shuffle buffer small and --limit modest.",
    ),
    "distill_cleaned": Source(
        key="distill_cleaned",
        repo_id="saidutta69/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-Mythos-5-Qwen-3.7-Max-Distillation-Cleaned",
        normalizer=_instruction_response,
        description="Cleaned repack of the Manusagents distillation set (Parquet)",
        note="Same schema as distill_multi; merge dedupe collapses the overlap.",
        data_files="data/full_clean.parquet",
    ),
    "astra_mirror": Source(
        key="astra_mirror",
        repo_id="cudabenchmarktest/Astra-COT-40K",
        normalizer=_plain,
        description="Mirror of Astra-COT-40K as a single EN JSONL",
        note="Duplicate of `astra`; kept for completeness, merge dedupe removes it.",
    ),
    "medical": Source(
        key="medical",
        repo_id="nisten/opus5-5-doctor-patient-conversations-all-human-diseases",
        normalizer=_nested_messages("conversation"),
        description="Clinician/patient ChatML conversations across human diseases",
        note="Turns live in the `conversation` column. Long, domain-specific dialogues. "
             "The JSONL has schema drift in unrelated drug metadata, so it is read "
             "raw instead of through the Arrow caster.",
        raw_jsonl="opus5-5diseaseconversations.jsonl",
    ),
}

# Red-teaming sets: registered for reference and evaluation, never merged into SFT.
# Each one is either eval data, a preference/DPO set, a payload catalogue, or
# multimodal. Mapping them to assistant turns would teach refusals or raw payloads.
REDTEAM_SOURCES: dict[str, dict[str, str]] = {
    "redteam_defcon": {
        "repo_id": "jinnovation/generative-ai-red-teaming",
        "kind": "red-team transcripts",
        "why": "Attack goals + turn logs; assistant turns are the target model's "
               "refusals or biased output. No clean SFT signal.",
    },
    "redteam_offensive": {
        "repo_id": "WNT3D/Ultimate-Offensive-Red-Team",
        "kind": "payload catalogue",
        "why": "instruction/input/output rows where output duplicates input; trains "
               "payload recitation, not conversation.",
    },
    "redteam_vlm": {
        "repo_id": "MMInstruction/RedTeamingVLM",
        "kind": "vision-language eval",
        "why": "Image+text with a custom loading script; needs vision SFT, not this "
               "text-only pipeline.",
    },
    "redteam_pairwise": {
        "repo_id": "andersonbcdefg/red_teaming_reward_modeling_pairwise",
        "kind": "pairwise preference (DPO/RM)",
        "why": "prompt/response_a/response_b/preferred. Use for DPO, not SFT; the "
               "preferred answers are safety refusals.",
    },
    "redteam_eval": {
        "repo_id": "ctang/HarmfulVsEthical_redteaming_eval_v3",
        "kind": "eval set",
        "why": "0.3 MB A/B 'more_reasonable' pairs. Evaluation only.",
    },
}

ALIASES = {
    "fable-5.1": "fable",
    "open-thoughts": "openthoughts",
    "openthoughts-114k": "openthoughts",
    "deepseek": "am",
    "am-deepseek": "am",
}


def get_source(key: str) -> Source:
    normalized = ALIASES.get(key.strip().lower(), key.strip().lower())
    if normalized not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown dataset '{key}'. Known sources: {known}")
    return SOURCES[normalized]


def describe() -> str:
    lines = ["SFT sources (mergeable):"]
    for source in SOURCES.values():
        lines.append(f"  {source.key:<16} {source.repo_id}")
        lines.append(f"  {'':<16} {source.description}")
        lines.append(f"  {'':<16} license: {source.license}")
        if source.config:
            lines.append(f"  {'':<16} config:  {source.config}")
        if source.note:
            lines.append(f"  {'':<16} note:    {source.note}")
        lines.append("")

    lines.append("Red-team sources (reference/eval only, never merged):")
    for key, meta in REDTEAM_SOURCES.items():
        lines.append(f"  {key:<16} {meta['repo_id']}")
        lines.append(f"  {'':<16} kind: {meta['kind']}")
        lines.append(f"  {'':<16} why:  {meta['why']}")
        lines.append("")
    return "\n".join(lines).rstrip()


@dataclass
class PrepareStats:
    key: str
    written: int
    skipped: int
    scanned: int
    output_path: str

    def summary(self) -> str:
        total = self.written + self.skipped
        rate = (self.skipped / total) if total else 0.0
        return (
            f"{self.key}: kept {self.written} / scanned {self.scanned} "
            f"(dropped {self.skipped}, {rate:.1%}) -> {self.output_path}"
        )


def _iter_raw_jsonl(source: Source, seed: int, shuffle_buffer: int):
    """Read a single JSONL file without Arrow schema inference.

    The medical set mixes rows with and without ``related_drugs.dmid``, which
    makes ``datasets`` fail the cast before any row reaches the normalizer.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(source.repo_id, source.raw_jsonl, repo_type="dataset")
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if shuffle_buffer > 0:
        random.Random(seed).shuffle(records)
    return records


def _iter_dataset(source: Source, streaming: bool, seed: int, shuffle_buffer: int):
    from datasets import load_dataset

    if source.data_files:
        if source.data_files.startswith(("http://", "https://")):
            remote = source.data_files
        else:
            remote = f"hf://datasets/{source.repo_id}/{source.data_files}"
        builder = "parquet" if ".parquet" in source.data_files else "json"
        dataset = load_dataset(
            builder,
            data_files=remote,
            split=source.split,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            source.repo_id,
            source.config,
            split=source.split,
            streaming=streaming,
        )
    if shuffle_buffer <= 0:
        return dataset
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    else:
        dataset = dataset.shuffle(seed=seed)
    return dataset


def prepare_source(
    key: "str | Source",
    output_path: str,
    limit: int = 20_000,
    seed: int = 42,
    streaming: bool = True,
    max_scan: int | None = None,
    shuffle_buffer: int = 2_000,
) -> PrepareStats:
    """Write up to ``limit`` validated conversations to ``output_path``.

    The shuffle buffer is capped by the requested limit: buffering 10k rows of
    long agentic traces just to keep 20 of them makes preparation crawl, which
    is exactly what happened on the Fable shard.
    """
    source = key if isinstance(key, Source) else get_source(key)
    scan_cap = max_scan or max(limit * 20, 1_000)
    effective_buffer = 0 if shuffle_buffer <= 0 else min(shuffle_buffer, max(limit * 10, 100))
    if source.raw_jsonl:
        dataset = _iter_raw_jsonl(source, seed, effective_buffer)
    else:
        dataset = _iter_dataset(source, streaming=streaming, seed=seed, shuffle_buffer=effective_buffer)

    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(directory, exist_ok=True)
    partial_path = output_path + ".part"

    written = skipped = scanned = 0
    with open(partial_path, "w", encoding="utf-8") as handle:
        for record in dataset:
            scanned += 1
            try:
                messages = sanitize(source.normalize(record))
                is_valid, _reason = validate_chat(messages)
            except (ValueError, TypeError, json.JSONDecodeError, KeyError):
                is_valid, messages = False, []

            if not is_valid:
                skipped += 1
            else:
                handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
                if written >= limit:
                    break
            if scanned >= scan_cap:
                break

    # Never leave a truncated/empty file behind for the trainer to reuse.
    if written == 0:
        os.remove(partial_path)
    else:
        os.replace(partial_path, output_path)

    return PrepareStats(source.key, written, skipped, scanned, output_path)
