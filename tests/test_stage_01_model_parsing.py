import sys
from pathlib import Path

import pandas as pd
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_01_model_parsing as stage_01  # noqa: E402


JR_CENTRAL_WIKITEXT = """
== 新幹線車両 ==
=== 現有車両 ===
全て電車。
; [[新幹線N700系電車|N700系]]
: 説明中の[[新幹線700系電車|700系]]は車両一覧として扱わない。
:; [[新幹線N700系電車|N700A]]
; [[新幹線N700S系電車|N700S]]（N700S系）
=== 廃形式 ===
==== 電車 ====
; [[新幹線0系電車|0系]]
==== その他 ====
; [[新幹線911形ディーゼル機関車|911形]]
; [[国鉄ホキ800形貨車#新幹線931形貨車|931形]]
== 在来線現有車両 ==
=== 電車 ===
* '''近郊形'''
** [[JR東海313系電車|313系]]
""".strip()


def test_parse_jr_central_definition_lists() -> None:
    rows = stage_01.parse_vehicle_wikitext(JR_CENTRAL_WIKITEXT.splitlines())

    assert [row["series"] for row in rows] == [
        "N700系",
        "N700S系",
        "0系",
        "911形",
        "931形",
        "313系",
    ]
    assert rows[0] == {
        "series": "N700系",
        "wiki_title": "新幹線N700系電車",
        "status": "現役",
        "type": "新幹線電車",
        "subtype": "",
    }
    assert rows[1]["status"] == "現役"
    assert rows[1]["type"] == "新幹線電車"
    assert rows[2]["status"] == "廃止"
    assert rows[2]["type"] == "新幹線電車"
    assert rows[3]["status"] == "廃止"
    assert rows[3]["type"] == "その他新幹線車両"
    assert rows[4]["type"] == "その他新幹線車両"
    assert rows[5]["status"] == "現役"
    assert rows[5]["type"] == "電車"
    assert rows[5]["subtype"] == "近郊形"


def test_manual_catalog_supports_alias_merge_and_operator_lists(tmp_path: Path) -> None:
    catalog_path = tmp_path / "manual_series.csv"
    catalog_path.write_text(
        "source_series,series,entry_kind,wiki_title,full_name,status,type,subtype,"
        "operator_jp,operator_en,commons_root_category\n"
        "青い森701系,701系,merge,青い森鉄道701系電車,,現役,電車,,"
        "青い森鉄道|JR東日本,Aoimori Railway|JR East,Aoimori Railway 701 series\n",
        encoding="utf-8",
    )

    catalog = stage_01.load_manual_series_catalog(catalog_path)

    assert catalog.iloc[0]["source_series"] == "青い森701系"
    assert catalog.iloc[0]["series"] == "701系"
    assert catalog.iloc[0]["full_name"] == "青い森鉄道701系電車"
    assert catalog.iloc[0]["operator_jp"] == ["青い森鉄道", "JR東日本"]
    assert catalog.iloc[0]["operator_en"] == ["Aoimori Railway", "JR East"]


def test_manual_new_collision_requires_explicit_merge() -> None:
    parsed = pd.DataFrame([{"series": "701系"}])
    manual = pd.DataFrame(
        [
            {
                "source_series": "第三部门701系",
                "series": "701系",
                "entry_kind": "new",
            }
        ]
    )

    with pytest.raises(ValueError, match="同车/别名请改用 merge"):
        stage_01.append_manual_series_catalog(parsed, manual)


def test_manual_same_name_different_vehicle_uses_qualified_series() -> None:
    parsed = pd.DataFrame([{"series": "1000形"}])
    manual = pd.DataFrame(
        [
            {
                "source_series": "1000形",
                "series": "架空鉄道1000形",
                "entry_kind": "new",
            }
        ]
    )

    result = stage_01.append_manual_series_catalog(parsed, manual)

    assert result["series"].tolist() == ["1000形", "架空鉄道1000形"]
