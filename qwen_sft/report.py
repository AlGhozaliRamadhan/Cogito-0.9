"""Research-grade training curve from a run's saved metrics.

    python -m qwen_sft report                       # plots training_output/train_log.jsonl (or trainer_state.json)
    python -m qwen_sft report --log path/train_log.jsonl --out chart.png
    python -m qwen_sft report --state path/trainer_state.json

The training run appends one JSON row per log event to
``training_output/train_log.jsonl`` (loss, learning_rate, grad_norm, epoch,
step). Rows from resumed sessions are appended again, so duplicates collapse
to their last occurrence before plotting. ``trainer_state.json`` — which every
checkpoint push carries — is accepted as a fallback source, so the curve can
be rebuilt from the Hub even when a session vanished mid-run.
"""

import argparse
import json
import os
import sys

from qwen_sft.config import PROJECT_ROOT

DEFAULT_LOG = os.path.join(PROJECT_ROOT, "training_output", "train_log.jsonl")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "loss_chart.png")


def read_rows(path) -> list[dict]:
    """Parse a .jsonl log file into rows, skipping blank/broken lines."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def rows_from_state(path) -> list[dict]:
    """Extract log_history from a trainer_state.json checkpoint body."""
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    history = state.get("log_history") if isinstance(state, dict) else None
    return [row for row in history if isinstance(row, dict)] if isinstance(history, list) else []


def dedupe(rows) -> list[dict]:
    """Keep the last occurrence per global_step, oldest first.

    A resumed session re-logs steps it inherited from trainer_state; the last
    append wins, which is the value the trainer itself was holding.
    """
    by_step = {}
    for row in rows:
        step = row.get("global_step")
        if isinstance(step, int):
            by_step[step] = row
        elif step is not None:
            try:
                by_step[int(step)] = row
            except (TypeError, ValueError):
                continue
    return [by_step[step] for step in sorted(by_step)]


def collect(log_dir: str) -> tuple[list[dict], str]:
    """Rows from train_log.jsonl, falling back to trainer_state.json."""
    log_path = os.path.join(log_dir, "train_log.jsonl")
    rows = dedupe(read_rows(log_path))
    if rows:
        return rows, log_path
    state_rows = dedupe(rows_from_state(os.path.join(log_dir, "trainer_state.json")))
    if state_rows:
        return state_rows, os.path.join(log_dir, "trainer_state.json")
    return [], log_path


def plot(rows, out_path: str, title: str = "SFT training loss") -> str:
    """Render loss (raw + EMA) over the cosine LR schedule; return the path.

    Two stacked panels share the optimizer-step axis; a secondary axis on top
    shows the epoch fraction reached. Style is deliberately restrained: thin
    raw line, thick median trend, light grid, no chartjunk.
    """
    import matplotlib
    import numpy

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_rows = [row for row in rows
                 if isinstance(row.get("loss"), (int, float))
                 and isinstance(row.get("global_step"), int)]
    if len(loss_rows) < 2:
        raise ValueError(f"only {len(loss_rows)} loss points; nothing to plot")

    steps = numpy.array([row["global_step"] for row in loss_rows])
    loss = numpy.array([row["loss"] for row in loss_rows])
    window = max(3, len(loss) // 60)
    ema = numpy.empty_like(loss)
    ema[0] = loss[0]
    alpha = 2.0 / (window + 1.0)
    for index in range(1, len(loss)):
        ema[index] = alpha * loss[index] + (1 - alpha) * ema[index - 1]

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(9, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_loss.plot(steps, loss, color="#4C72B0", lw=0.8, alpha=0.35)
    ax_loss.plot(steps, ema, color="#2F5587", lw=2.0, label="trend (EMA)")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(axis="y", alpha=0.3, ls="--", lw=0.6)
    ax_loss.spines["right"].set_visible(False)
    ax_loss.text(0.99, 0.03, f"final loss {loss[-1]:.3f}   |   {len(loss)} log points",
                 transform=ax_loss.transAxes, ha="right", va="bottom",
                 fontsize=9, color="#404040")
    ax_loss.set_title(title, loc="left", fontsize=12)

    epochs = [row.get("epoch") for row in loss_rows]
    if all(isinstance(value, (int, float)) for value in epochs) and epochs[-1]:
        steps_per_epoch = steps[-1] / epochs[-1]
        top = ax_loss.secondary_xaxis(
            "top", functions=(lambda s: s / steps_per_epoch,
                              (lambda e: e * steps_per_epoch)))
        top.set_xlabel("epoch")
        top.spines["top"].set_visible(True)

    lr_rows = [row for row in rows
               if isinstance(row.get("learning_rate"), (int, float))
               and isinstance(row.get("global_step"), int)]
    if len(lr_rows) >= 2:
        lr_steps = numpy.array([row["global_step"] for row in lr_rows])
        lr = numpy.array([row["learning_rate"] for row in lr_rows])
        ax_lr.plot(lr_steps, lr, color="#55A868", lw=1.3)
        ax_lr.set_ylabel("learning\nrate")
    ax_lr.set_xlabel("optimizer step")
    ax_lr.grid(axis="y", alpha=0.3, ls="--", lw=0.6)
    ax_lr.spines[["right", "top"]].set_visible(False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_chart(log_dir: str, out_path: str = DEFAULT_OUT,
                 title: str = "SFT training loss") -> tuple[str, str] | None:
    """Find metrics under a training_output dir and plot them.

    Returns (path, source_used) or None when there is nothing to plot yet.
    """
    rows, source = collect(log_dir)
    if not rows:
        return None
    out = plot(rows, out_path, title=title)
    return out, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the SFT training curve")
    parser.add_argument("--log", default=DEFAULT_LOG, help="train_log.jsonl path")
    parser.add_argument("--state", default=None, help="trainer_state.json fallback source")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output PNG path")
    parser.add_argument("--title", default="SFT training loss")
    args = parser.parse_args(argv)

    rows = dedupe(read_rows(args.log))
    source = args.log
    if not rows and args.state:
        rows, source = dedupe(rows_from_state(args.state)), args.state
    elif not rows and os.path.basename(args.log) == "train_log.jsonl":
        state = os.path.join(os.path.dirname(args.log), "trainer_state.json")
        if os.path.isfile(state):
            rows, source = dedupe(rows_from_state(state)), state
    if not rows:
        print(f"[REPORT] nothing to plot: no rows in {args.log}"
              + (" or " + os.path.join(os.path.dirname(args.log), "trainer_state.json")
                 if args.state is None else ""))
        return 1

    out = plot(rows, args.out, title=args.title)
    print(f"[REPORT] {len(rows)} rows from {source} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
