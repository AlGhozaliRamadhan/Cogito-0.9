"""The eval must not report an untrained adapter as a measured one."""

from qwen_sft.eval.bench import adapter_norm


class _Param:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return self

    def sum(self):
        return self.value


class _Model:
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def named_parameters(self):
        return iter(self._pairs)


def test_adapter_norm_sums_only_the_b_matrix():
    model = _Model([
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight", _Param(9.0)),
        ("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight", _Param(0.5)),
        ("base_model.model.layers.1.self_attn.q_proj.lora_B.default.weight", _Param(0.25)),
    ])
    assert adapter_norm(model) == 0.75


def test_adapter_norm_is_none_without_an_adapter():
    model = _Model([("base_model.model.layers.0.self_attn.q_proj.weight", _Param(3.0))])
    assert adapter_norm(model) is None


def test_adapter_norm_reports_zero_for_an_untrained_adapter():
    # PEFT zero-initialises lora_B: this is the no-op case the guard catches.
    model = _Model([("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight", _Param(0.0))])
    assert adapter_norm(model) == 0.0
