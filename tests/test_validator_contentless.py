"""Regression test: record 478 crash.

A system feedback turn whose dict lacks a ``content`` key (only ``role`` and
``=``) slipped through audit and crashed format_example with KeyError('content').
The validator must reject any conversation containing a content-less message,
so this malformed shape never reaches training.

Imports come from the canonical ``cogito.validation`` module; the legacy
``cogito.generators.validator`` alias must expose the exact same objects.
"""
import pytest

from cogito.validation import COGITO_SYSTEM_PROMPT, validate_conversation_structure


def _assistant_turn(action, text, confidence):
    return {
        "role": "assistant",
        "content": (
            f"<confidence>{confidence}</confidence><thought>{action} step</thought>"
            f"<action>{action}</action>\n{text}"
        ),
    }


def _valid_conv():
    """A structurally valid conversation (loop -> feedback -> terminal answer)."""
    return [
        {"role": "system", "content": COGITO_SYSTEM_PROMPT},
        {"role": "user", "content": "Diagnose and fix the failing build."},
        _assistant_turn("run_command", "Running a quick diagnostic.", 0.50),
        {"role": "system", "content": "Proceeding."},
        _assistant_turn("answer", "Here is the final answer.", 0.95),
    ]


def test_valid_baseline_passes():
    assert validate_conversation_structure(_valid_conv())


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_missing_content_key_rejected(idx):
    conv = _valid_conv()
    conv[idx] = {"role": conv[idx]["role"]}
    ok, reason = validate_conversation_structure(conv)
    assert not ok, f"conversation with content-less message {idx} must be rejected"
    assert "Message" in reason or "Turn" in reason


def test_missing_content_system_feedback_rejected():
    """Mirror of the malformed msg-3 in record 478 (role-only, no content)."""
    messages = _valid_conv()
    messages[3] = {"role": "system"}
    ok, reason = validate_conversation_structure(messages)
    assert not ok, "content-less system feedback turn must be rejected"


def test_generators_validator_alias_is_identical():
    """Legacy alias cogito.generators.validator must re-export the canonical API.

    Any code that imported the old scripts/generators/validator.py path keeps
    resolving to the same objects (not copies).
    """
    import cogito.generators.validator as alias
    import cogito.validation as canonical

    for name in (
        "COGITO_SYSTEM_PROMPT",
        "SYCOPHANCY_KEYWORDS",
        "REQUIRED_TAGS",
        "TERMINAL_ACTIONS",
        "LOOP_ACTIONS",
        "STYLE_CLICHES",
        "ValidationResult",
        "canonicalize_system_prompt",
        "check_sycophancy",
        "check_writing_style",
        "validate_assistant_tags",
        "validate_confidence_value",
        "validate_all_assistant_messages",
        "validate_confidence_calibration",
        "validate_conversation_structure",
        "validate_heated_conversation",
    ):
        assert getattr(alias, name) is getattr(canonical, name), f"alias.{name} differs"

    # Underscore internals are re-exported explicitly (star-import skips them).
    assert alias._strip_assistant_tags is canonical._strip_assistant_tags
    assert alias._extract_tag_inner is canonical._extract_tag_inner
