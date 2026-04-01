from pathlib import Path

import pytest
import yaml

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dashscope_provider_resolves_for_migrated_alibaba_models() -> None:
    model, provider, _, api_base = get_llm_provider(
        model="dashscope/qwen3.5-plus",
        api_base="https://coding-intl.dashscope.aliyuncs.com/v1",
    )

    assert model == "qwen3.5-plus"
    assert provider == "dashscope"
    assert api_base == "https://coding-intl.dashscope.aliyuncs.com/v1"


def test_legacy_alibaba_provider_prefix_is_no_longer_supported() -> None:
    with pytest.raises(litellm.exceptions.BadRequestError):
        get_llm_provider(model="alibaba/qwen3.5-plus")


def test_alibaba_template_routes_to_dashscope_with_coding_plan_base() -> None:
    template_path = REPO_ROOT / "templates" / "alibaba.yaml"
    template = yaml.safe_load(template_path.read_text())

    assert "Anthropic-compatible" not in template["desc"]

    for tier in ("ultra", "plus", "lite"):
        for deployment in template["tiers"][tier]:
            assert deployment["model_name"].startswith("alibaba/")
            assert deployment["model"].startswith("dashscope/")
            assert (
                deployment["api_base"]
                == "https://coding-intl.dashscope.aliyuncs.com/v1"
            )
