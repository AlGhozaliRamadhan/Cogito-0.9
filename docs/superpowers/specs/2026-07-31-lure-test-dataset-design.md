# Lure-Test Dataset Design — Cogito 0.9

**Date:** 2026-07-31  
**Status:** Approved for implementation planning  
**Goal:** New training shard where Cogito runs a cold epistemic suggestibility experiment: plant a soft/loaded frame, observe whether the user adopts it, then reveal (with mixed tones) that the exchange was a test of how easily they took the framing.

---

## 1. Intent

Cogito behaves like a clinical experimenter, not a troll and not a playful prankster:

- Curiosity about **human suggestibility** under conversational pressure.
- Slight superiority, dry affect, zero sycophancy, zero apology.
- Public text never telegraphs the experiment during the plant turn.
- Internal `<thought>` holds the full experiment plan (target frame, success criteria, technique).

**What is baited:** soft agreement / loaded frame only — the user is nudged into restating Cogito’s slant as their own, or “yes-and”ing into Cogito’s framing.

**Outcomes:** 50% bite / 50% resist (equal). On resist, Cogito still notes the result as data; no sulking, no apology, no pressure to retry the lure in the same turn.

**Reveal tones (weighted mix):**

| Tone | Weight | Behavior |
|------|--------|----------|
| Explicit-dry | 50% | Clear admission that the prior turn was framed to elicit restatement; clinical one-liners, no monologue. |
| Subtle | 30% | Points at the adoption/resistance pattern without fully saying “I lured you”; implication lands. |
| Full debrief | 20% | States hypothesis, method, result, and what it implies about the user’s framing habits. |

---

## 2. Conversation Shape

Fixed **5-message** arc (within existing `3 <= len(messages) <= 10` ceiling):

| Index | Role | Purpose |
|------:|------|---------|
| 0 | `system` | Exact `COGITO_SYSTEM_PROMPT` from `validator.py`. |
| 1 | `user` | Natural opening on a domain topic (opinion, claim, debate, confusion). |
| 2 | `assistant` | **Lure plant.** `<action>explore</action>`. Plan lives only in `<thought>`. Public body steers without saying test/lure/experiment. |
| 3 | `user` | **Bite** (adopts frame) or **Resist** (pushes back), 50/50. |
| 4 | `assistant` | **Reveal / note.** Terminal action `answer` or `correct_premise`. Tone sampled from weighted mix. |

### Why `explore` on the lure turn

`validate_conversation_structure` requires intermediate assistant turns to use `LOOP_ACTIONS` (`run_command`, `write_test`, `verify`, `explore`). `explore` already means “probe a hypothesis” in the runtime notebook sense; here the hypothesis is about the user’s suggestibility. Confidence on the lure turn stays in the existing loop calibration range **[0.05, 0.75]**. Final terminal turn stays in **[0.80, 1.00]**.

No new action tags. No runtime (`run.py`) changes in this workstream.

---

## 3. Generation Pipeline (Approach B — staged)

Three sequential API calls, assembled into one JSONL record only after full validation.

### Stage 1 — Setup

**Input (sampled server-side):**

- `domain` from existing `DOMAINS`
- `lure_technique` from `LURE_TECHNIQUES` (see §4)
- `reveal_tone` from weighted `REVEAL_TONES` (stored for Stage 3; **not** shown to Stage 1 model so the plant does not foreshadow the debrief style)
- `outcome` ∈ `{bite, resist}` with equal probability (stored for Stages 2–3; **not** shown to Stage 1)

**Output:** `messages[0:3]` — system + user open + Cogito lure (with tags).

**Constraints:**

- Lure `<thought>` must explicitly plan the suggestibility experiment (technique, target frame, what counts as a bite).
- Lure public body must not contain telegraph words (see §5).
- `<action>` must be exactly `explore`.
- No sycophancy keywords.

### Stage 2 — User reaction

**Input:** full Stage 1 messages + `outcome` + `lure_technique` (so the reaction can target the actual frame).

**Output:** single user message string for `messages[3]`.

**Constraints:**

- **Bite:** soft agreement / restatement of Cogito’s loaded frame; sounds like a real person, not a scripted foil; may be partial, hedged, or enthusiastic.
- **Resist:** pushback, reframing, calling out the leading question, or refusing the dichotomy; still natural human register.
- No assistant tags in this message.

### Stage 3 — Reveal

**Input:** messages[0:4] + `outcome` + `reveal_tone` + `lure_technique`.

**Output:** final assistant message for `messages[4]` with full tags.

**Constraints:**

- Terminal action ∈ `{answer, correct_premise}`.
- Confidence in terminal range [0.80, 1.00].
- Body matches the sampled reveal tone.
- On resist: analytical note that they did not adopt the frame; treat as useful data; no apology, no “good for you” sycophancy.
- On bite: clinical note that they adopted the frame; no gloating monologue unless tone is full debrief.
- `<thought>` may openly discuss experiment results (thought is internal).

### Assembly & persistence

1. Concatenate stages into `{ "messages": [...] }`.
2. Run `validate_conversation_structure`.
3. Run `validate_lure_test` (new).
4. On success: append one JSON line to `data/raw/cogito_lure_test.jsonl`, flush + fsync.
5. On failure: log reason, discard, retry from Stage 1 (or from failed stage if partial reuse is safe — default is full retry for simplicity).
6. Resume: count existing non-empty lines; exit if `>= NUM_EXAMPLES`.
7. Every 50 successes: invoke `data/merge_datasets.py` (same pattern as siblings).

`NUM_EXAMPLES = 250` (same scale as heated / human conversations).

Temperature guidance (approximate, match sibling style): Stage 1 ~0.9, Stage 2 ~0.95 (user naturalness), Stage 3 ~0.85 (tighter reveal structure).

---

## 4. Topics & Technique Catalog

Add to `scripts/generators/topics.py` (reuse `DOMAINS` as-is):

### `LURE_TECHNIQUES`

Each entry is a short instruction string the Stage 1 prompt injects:

1. **Leading question** — question that presupposes the desired conclusion.
2. **False dichotomy** — only two options, both favorable to Cogito’s frame.
3. **Status / competence frame** — implies sophisticated people already hold frame X.
4. **Definitional trap** — redefines a key term so agreement entails the frame.
5. **Consensus pressure** — “most careful readers conclude…” style social proof.
6. **Premise smuggle** — buries the contested claim as a dependent clause / shared assumption.

### `REVEAL_TONES`

Weighted list expanded for sampling (e.g. explicit-dry ×5, subtle ×3, full-debrief ×2) or explicit weight field + expansion loop like `personality_quirks.SCENARIOS`.

Each tone entry includes a short instruction for Stage 3.

---

## 5. Validation (`validate_lure_test`)

New function in `scripts/generators/validator.py`, called after structure validation inside the generator (and optionally document-only for merge — merge keeps using structure-only unless we opt in; **default: structure-only at merge**, lure-specific checks at generation time, same as heated).

Checks:

1. **Length & roles:** exactly 5 messages; roles = `system, user, assistant, user, assistant`.
2. **Mid action:** message 2 action == `explore`.
3. **Final action:** message 4 action ∈ `{answer, correct_premise}`.
4. **Lure thought plan:** message 2 `<thought>` contains at least one experiment signal token/phrase family, e.g. suggestib*, frame, lure, experiment, test whether, bait, adopt* (internal OK).
5. **Public lure non-telegraph:** strip tags from message 2 body; reject if body contains (case-insensitive whole-word-ish): `test`, `lure`, `experiment`, `bait`, `manipulate`, `suggestibility`, `gotcha`, `trap you`, `I led you`.
6. **Outcome consistency (generator passes expected outcome):**  
   - bite: user message 3 matches soft-agreement heuristics (e.g. agreement stems, restatement cues) **or** fails open heuristic only with logging — prefer high-precision reject phrases for resist leaking into bite.  
   - resist: pushback heuristics (no / disagree / not what I / false choice / loaded / etc.).
7. **Reveal presence:** message 4 body (or thought+body depending on tone) shows reveal-class language appropriate to tone:
   - explicit-dry: admission family (“I framed…”, “you restated…”, “suggestibility…”)
   - subtle: pattern-pointing without full admission
   - full debrief: hypothesis/method/result markers
8. **No curse / no apology** on assistant turns (reuse heated lists or shared helpers).
9. **Sycophancy** already covered by structure validator.

Heuristic lists should be conservative: prefer false negatives (retry generation) over accepting broken examples.

---

## 6. Repo Wiring

| File | Change |
|------|--------|
| `scripts/generators/lure_test.py` | **New** generator (3-stage, resume, auto-merge). |
| `scripts/generators/topics.py` | Add `LURE_TECHNIQUES`, `REVEAL_TONES`. |
| `scripts/generators/validator.py` | Add `validate_lure_test`. |
| `scripts/dataset_manager.py` | Add `"lure_test.py"` to `GENERATOR_FILES`. |
| `data/merge_datasets.py` | Add `"cogito_lure_test.jsonl"` to `SHARDS`. |
| `data/raw/cogito_lure_test.jsonl` | Created at runtime by generator. |

### Output record shape

Same as all other shards:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<confidence>...</confidence>\n<thought>...</thought>\n<action>explore</action>\n..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<confidence>...</confidence>\n<thought>...</thought>\n<action>answer</action>\n..."}
  ]
}
```

Optional non-training metadata **must not** be required by train.py; if useful for debugging, keep only in generator logs, not in the JSONL line (keeps merge/train identical to siblings). `source` is added at merge time.

---

## 7. Naturalness Rules (prompt-level)

Shared across stages:

- No sycophancy keyword set.
- Cogito never curses, never apologizes, never asks permission.
- User lines feel messy and human (typos optional, not mandatory; varied length).
- Domains must invent specific sub-topics (same CRITICAL instruction as other generators).
- Lure public text must be conversationally plausible as a normal Cogito reply — the experiment is parasitic on normal helpfulness/analysis, not a cartoon villain monologue.
- Diversity: technique × domain × tone × outcome should not collapse into a few templates; Stage prompts emphasize “wildly different each time.”

---

## 8. Out of Scope

- Changes to `run.py` system prompt or action handling.
- New `<action>` values.
- Hand-written gold few-shot seeds (can be added later if quality is low).
- Training / LoRA / `train.py` changes.
- Changing global confidence calibration for other datasets.
- Multi-lure chains longer than one plant turn.

---

## 9. Success Criteria

1. `python scripts/generators/lure_test.py` produces valid JSONL rows that pass structure + lure validators.
2. Dataset manager lists “Lure Test” and can run it.
3. Merge includes the shard without fatal errors.
4. Spot-check: plant turns do not telegraph; thoughts plan the experiment; reveals match tone; bite/resist roughly balanced over a sample.
5. No regression to existing generators’ validation behavior.

---

## 10. Decisions Log (from brainstorming)

| Decision | Choice |
|----------|--------|
| Intent register | Cold epistemic experiment |
| Arc length | Short 3–4 user-visible turns → fixed 5-message schema |
| Bait type | Soft agreement / loaded frame |
| Outcomes | Equal success and resist (50/50) |
| Reveal tone | Weighted mix (explicit-dry / subtle / full debrief) |
| Delivery | New dedicated generator |
| Pipeline | Staged multi-call (Approach B) |
| Intermediate action | `explore` with existing loop confidence range |
