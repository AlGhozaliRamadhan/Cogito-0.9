"""Shared OpenAI-compatible API client with runtime model fallback.

All generators import from here instead of duplicating setup.

Import safety: importing this module NEVER requires an API key and never
touches the network. Missing credentials surface as ``CogitoAPIKeyError``
at call time (``get_client`` / ``get_working_model``), so any module can
import the generators on a machine without NVIDIA_API_KEY configured.
"""
import os
import time
from openai import OpenAI

_client = None
_current_model = None
_blacklisted_models = set()

BASE_URL = "https://ri4ckkb.abc-tunnel.us/v1"

MODELS = [
    "oc/deepseek-v4-flash-free",
    "nvidia/nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nvidia/nemotron-3-super-120b-a12b",
    "openrouter/poolside/laguna-s-2.1:free",
    "xkiro/minimax/minimax-m3",
    "xkiro/deepseek/deepseek-v4-pro",
    "ollama/minimax-m3",
    "gcli/grok-4.5",
    "mistral/mistral-large-latest",
]


class CogitoAPIKeyError(RuntimeError):
    """Raised when an API call is attempted without NVIDIA_API_KEY configured."""


def _get_api_key():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("NVIDIA_API_KEY")
    except Exception:
        return os.environ.get("NVIDIA_API_KEY")


def _require_api_key() -> str:
    """Return the API key or raise CogitoAPIKeyError (call-time, not import-time)."""
    api_key = _get_api_key()
    if not api_key:
        raise CogitoAPIKeyError(
            "NVIDIA_API_KEY not found in Kaggle Secrets or environment."
        )
    return api_key


def get_working_model():
    """Try models in order until one responds. Called lazily at first use."""
    global _current_model
    if _current_model is not None:
        return _current_model

    api_key = _require_api_key()

    for model in MODELS:
        if model in _blacklisted_models:
            continue
        try:
            client = OpenAI(base_url=BASE_URL, api_key=api_key)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=10,
            )
            _current_model = model
            print(f"[INFO] Using model: {model}")
            return model
        except Exception as e:
            print(f"[WARN] Model {model} failed: {e}")
            continue

    raise RuntimeError("[FATAL] All models failed")


def get_client():
    """Return (and cache) the OpenAI client for the current model."""
    global _client
    if _client is not None:
        return _client

    api_key = _require_api_key()

    model = get_working_model()
    _client = OpenAI(base_url=BASE_URL, api_key=api_key)
    return _client


def get_current_model():
    """Return the currently active model name (may change mid-run after fallback)."""
    return _current_model


def _looks_like_rate_limit(error: Exception) -> bool:
    s = str(error).lower()
    return any(
        keyword in s
        for keyword in ("429", "rate limit", "rate_limit", "quota", "too many requests")
    )


def handle_api_failure(error: Exception) -> bool:
    """Called mid-generation when the current model fails.

    Blacklists the failing model, invalidates the client cache, probes for the
    next working model, and returns True when a new model is ready.

    Never returns False — when all models are exhausted, it waits for rate
    limits to reset, clears the blacklist, and retries indefinitely until a
    model becomes available.
    """
    global _client, _current_model

    tag = "RATE LIMIT" if _looks_like_rate_limit(error) else "MODEL FAIL"
    print(f"\n[{tag}] {_current_model} -> {str(error)[:200]}")

    _blacklisted_models.add(_current_model)
    _current_model = None
    _client = None

    api_key = _require_api_key()

    while True:
        for model in MODELS:
            if model in _blacklisted_models:
                continue
            try:
                test_client = OpenAI(base_url=BASE_URL, api_key=api_key)
                test_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    timeout=10,
                )
                _current_model = model
                _client = test_client
                print(f"[SWITCHED] Now using model: {model}\n")
                return True
            except Exception as e:
                print(f"[WARN] Fallback model {model} also failed: {e}")
                _blacklisted_models.add(model)
                continue

        # All models exhausted — wait for rate limits to reset, then retry
        wait_seconds = 90
        print(f"[WAIT] All models exhausted. Clearing blacklist and retrying in {wait_seconds}s...")
        time.sleep(wait_seconds)
        _blacklisted_models.clear()
        print("[RETRY] Blacklist cleared. Retrying model chain...")


# NOTE: no eager get_working_model() call here. Probing models requires the
# API key and hits the network, so it must happen lazily on first use.
