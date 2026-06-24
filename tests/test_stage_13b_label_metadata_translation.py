import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_13b_label_metadata_translation as stage_13b  # noqa: E402


def test_observed_operators_are_grouped_by_effective_label() -> None:
    candidates = pd.DataFrame(
        [
            {"label_ja": "117系", "operator_jp": "JR西日本", "operator_en": "JR West"},
            {"label_ja": "117系", "operator_jp": "JR東海", "operator_en": "JR Central"},
            {
                "label_ja": "117系(West Express 銀河)",
                "operator_jp": "JR西日本",
                "operator_en": "JR West",
            },
        ]
    )
    translations = {
        "JR West": ("JR西日本", "JR西日本"),
        "JR Central": ("JR東海", "JR东海"),
    }

    lookup = stage_13b.build_observed_operator_lookup(candidates, translations)

    assert lookup["117系"] == {
        "ja": ["JR東海", "JR西日本"],
        "en": ["JR Central", "JR West"],
        "zh": ["JR东海", "JR西日本"],
    }
    assert lookup["117系(West Express 銀河)"] == {
        "ja": ["JR西日本"],
        "en": ["JR West"],
        "zh": ["JR西日本"],
    }


def test_canonical_pair_rejects_mismatched_stage_06_japanese_name() -> None:
    candidates = pd.DataFrame(
        [
            {"label_ja": "新形式", "operator_jp": "JR東日本", "operator_en": "JR West"},
            {"label_ja": "新形式", "operator_jp": "JR西日本", "operator_en": "JR West"},
        ]
    )

    lookup = stage_13b.build_observed_operator_lookup(
        candidates,
        {"JR West": ("JR西日本", "JR西日本")},
    )

    assert lookup["新形式"] == {
        "ja": ["JR西日本"],
        "en": ["JR West"],
        "zh": ["JR西日本"],
    }


def test_unknown_operator_keeps_blank_chinese_review_slot() -> None:
    candidates = pd.DataFrame(
        [
            {
                "label_ja": "新形式",
                "operator_jp": "新鉄道",
                "operator_en": "New Railway",
            }
        ]
    )

    lookup = stage_13b.build_observed_operator_lookup(candidates, {})

    assert lookup["新形式"] == {
        "ja": ["新鉄道"],
        "en": ["New Railway"],
        "zh": [""],
    }
