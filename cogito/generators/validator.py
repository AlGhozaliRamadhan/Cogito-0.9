"""Backwards-compat alias: the canonical validator lives in cogito.validation.

Kept so ``from cogito.generators.validator import ...`` (and any code that
imported the old ``scripts/generators/validator.py`` path) keeps resolving.
"""
from cogito.validation import *  # noqa: F401,F403
from cogito.validation import (  # noqa: F401  (explicit re-export; underscore names are not covered by *)
    COGITO_SYSTEM_PROMPT,
    SYCOPHANCY_KEYWORDS,
    REQUIRED_TAGS,
    TERMINAL_ACTIONS,
    LOOP_ACTIONS,
    STYLE_CLICHES,
    ValidationResult,
    canonicalize_system_prompt,
    check_sycophancy,
    check_writing_style,
    validate_assistant_tags,
    validate_confidence_value,
    validate_all_assistant_messages,
    validate_confidence_calibration,
    validate_conversation_structure,
    validate_heated_conversation,
    _strip_assistant_tags,
    _extract_tag_inner,
)
