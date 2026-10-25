"""Unsloth's fast kernels apply only if it is imported before
trl/transformers/peft — a plain warning at runtime, a real VRAM/perf loss on a
16 GB T4. main() must `import unsloth` before anything heavy, while module
level stays dependency-free so `--help` and tests run without GPU libraries.
"""

import ast
import os

TRAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "qwen_sft", "finetune", "train.py")

HEAVY = ("unsloth", "torch", "trl", "transformers", "peft", "datasets")


def _parsed():
    with open(TRAIN, encoding="utf-8") as handle:
        return ast.parse(handle.read())


def _direct_imports(body) -> list[tuple[int, str]]:
    """(line, top-level module name) of the Import nodes directly in *body*."""
    pairs = []
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                pairs.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            pairs.append((node.lineno, (node.module or "").split(".")[0]))
    return pairs


def _subtree_imports(node) -> list[tuple[int, str]]:
    """Same shape, but every import anywhere inside *node*."""
    return _direct_imports([sub for sub in ast.walk(node)])


def _first_lines(pairs) -> dict[str, int]:
    first = {}
    for line, name in pairs:
        if name not in first or line < first[name]:
            first[name] = line
    return first


def _main_fn(tree):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main")


def test_main_imports_unsloth_before_everything_it_patches():
    first = _first_lines(_subtree_imports(_main_fn(_parsed())))
    for name in ("unsloth", "trl", "datasets"):
        assert name in first, f"main() no longer imports {name}"
    assert first["unsloth"] < first["trl"], (
        "`import unsloth` must precede the trl/datasets imports in main():"
        " otherwise Unsloth's kernels are skipped (slower, more VRAM)")
    assert first["unsloth"] < first["datasets"]


def test_module_level_stays_dependency_free():
    """Only functions may import the GPU stack; module scope imports stdlib/config."""
    roots = [name for _, name in _direct_imports(_parsed().body)]
    offenders = [name for name in HEAVY if name in roots]
    assert not offenders, f"module level imports GPU-only packages: {offenders}"


def test_lazy_imports_live_inside_functions():
    """unsloth/trl/peft exist as lazy imports somewhere (the design), and the
    module-level ban above is what keeps `--help` dependency-free."""
    lazy_users = [node.name for node in _parsed().body
                  if isinstance(node, ast.FunctionDef)
                  and any(name in ("unsloth", "trl", "peft")
                          for _, name in _subtree_imports(node))]
    assert lazy_users, "expected load_base/attach_lora-style lazy GPU imports"
