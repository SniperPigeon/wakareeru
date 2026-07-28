import sys
from pathlib import Path


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
