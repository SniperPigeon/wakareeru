import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_02_model_fixing as stage_02  # noqa: E402


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
