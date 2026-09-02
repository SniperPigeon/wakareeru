import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_02_model_fixing as stage_02  # noqa: E402


def test_manual_catalog_row_serializes_stage_06_metadata() -> None:
    value = stage_02.build_manual_metadata_json(
        pd.Series(
            {
                "source_series": "ET127系",
                "entry_kind": "merge",
                "operator_jp": ["えちごトキめき鉄道"],
                "operator_en": ["Echigo Tokimeki Railway"],
                "submodel": "ET127",
            }
        )
    )

    assert value == (
        '{"operator_en": "Echigo Tokimeki Railway", '
        '"operator_jp": "えちごトキめき鉄道", "submodel": "ET127"}'
    )


def test_multi_operator_catalog_row_does_not_create_scalar_operator_lock() -> None:
    value = stage_02.build_manual_metadata_json(
        pd.Series(
            {
                "source_series": "shared",
                "entry_kind": "merge",
                "operator_jp": ["JR東日本", "青い森鉄道"],
                "operator_en": ["JR East", "Aoimori Railway"],
            }
        )
    )

    assert value == "{}"


def test_find_root_ignores_empty_exact_category(monkeypatch) -> None:
    candidates = ["JNR Kiha 185", "JNR Kiha 185 series"]
    monkeypatch.setattr(
        stage_02,
        "fetch_commons_categories",
        lambda prefix, limit: candidates,
    )
    monkeypatch.setattr(
        stage_02,
        "fetch_commons_category_info",
        lambda categories: {
            "JNR Kiha 185": {"files": 0, "subcats": 0},
            "JNR Kiha 185 series": {"files": 16, "subcats": 2},
        },
    )

    result = stage_02._find_root_from_prefixes(
        "キハ185系",
        ["JNR Kiha 185"],
    )

    assert result["commons_root_category"] == "JNR Kiha 185 series"
    assert result["needs_review"] is False
    assert "忽略空分类：JNR Kiha 185" in result["commons_root_decision"]


def test_find_root_marks_all_empty_candidates_for_review(monkeypatch) -> None:
    candidates = ["JNR Kiha 185", "JNR Kiha 185 series"]
    monkeypatch.setattr(
        stage_02,
        "fetch_commons_categories",
        lambda prefix, limit: candidates,
    )
    monkeypatch.setattr(
        stage_02,
        "fetch_commons_category_info",
        lambda categories: {
            category: {"files": 0, "subcats": 0} for category in categories
        },
    )

    result = stage_02._find_root_from_prefixes(
        "キハ185系",
        ["JNR Kiha 185"],
    )

    assert result["commons_root_category"] is None
    assert result["needs_review"] is True
    assert result["commons_root_decision"] == "候选分类均为空（无文件且无子分类）"


def test_empty_cached_root_is_refetched() -> None:
    row = {"series": "キハ185系", "wiki_title": "国鉄キハ185系気動車"}
    cache = {
        ("キハ185系", "国鉄キハ185系気動車"): {
            "commons_root_category": "JNR Kiha 185",
            "commons_root_decision": "精确匹配；未确认系列父子关系",
        }
    }

    assert stage_02._cached_commons_result(row, cache) is not None
    assert (
        stage_02._cached_commons_result(
            row,
            cache,
            empty_cached_roots={"JNR Kiha 185"},
        )
        is None
    )


def test_fixed_catalog_root_bypasses_discovery() -> None:
    row = pd.Series(
        {
            "commons_root_category": "Aoimori Railway 701 series",
            "operator_en": ["Aoimori Railway", "JR East"],
        }
    )

    result = stage_02._fixed_commons_result(row)

    assert result == {
        "commons_prefix": "Aoimori Railway 701 series",
        "commons_root_category": "Aoimori Railway 701 series",
        "commons_root_decision": "人工车型目录指定",
        "commons_operator_roots": {
            "Aoimori Railway": "Aoimori Railway 701 series",
            "JR East": "Aoimori Railway 701 series",
        },
        "commons_candidates": [],
        "needs_review": False,
    }


def test_same_canonical_series_and_root_merges_operators() -> None:
    rows = pd.DataFrame(
        [
            {
                "series": "701系",
                "wiki_title": "JR東日本701系電車",
                "operator_page_title": ["JR東日本の車両形式"],
                "operator_jp": ["JR東日本"],
                "operator_en": ["JR East"],
                "commons_root_category": "JR East 701 series",
                "commons_root_decision": "前缀精确匹配",
                "commons_operator_roots": {"JR East": "JR East 701 series"},
                "commons_candidates": ["JR East 701 series"],
                "needs_review": False,
            },
            {
                "series": "701系",
                "wiki_title": "青い森鉄道701系電車",
                "operator_page_title": [],
                "operator_jp": ["青い森鉄道"],
                "operator_en": ["Aoimori Railway"],
                "commons_root_category": "JR East 701 series",
                "commons_root_decision": "人工车型目录指定",
                "commons_operator_roots": {
                    "Aoimori Railway": "JR East 701 series"
                },
                "commons_candidates": [],
                "needs_review": False,
            },
        ]
    )

    result = stage_02.consolidate_series_roots(rows)

    assert len(result) == 1
    assert result.iloc[0]["operator_jp"] == ["JR東日本", "青い森鉄道"]
    assert result.iloc[0]["operator_en"] == ["JR East", "Aoimori Railway"]
    assert result.iloc[0]["commons_operator_roots"] == {
        "JR East": "JR East 701 series",
        "Aoimori Railway": "JR East 701 series",
    }


def test_same_canonical_series_with_different_roots_keeps_both_entries() -> None:
    rows = pd.DataFrame(
        [
            {
                "series": "701系",
                "operator_page_title": [],
                "operator_jp": ["JR東日本"],
                "operator_en": ["JR East"],
                "commons_root_category": "JR East 701 series",
                "commons_root_decision": "人工车型目录指定",
                "commons_operator_roots": {"JR East": "JR East 701 series"},
                "commons_candidates": [],
                "needs_review": False,
            },
            {
                "series": "701系",
                "operator_page_title": [],
                "operator_jp": ["第三部门鉄道"],
                "operator_en": ["Third Sector Railway"],
                "commons_root_category": "Third Sector Railway 701 series",
                "commons_root_decision": "人工车型目录指定",
                "commons_operator_roots": {
                    "Third Sector Railway": "Third Sector Railway 701 series"
                },
                "commons_candidates": [],
                "needs_review": False,
            },
        ]
    )

    result = stage_02.consolidate_series_roots(rows)

    assert result["commons_root_category"].tolist() == [
        "JR East 701 series",
        "Third Sector Railway 701 series",
    ]
