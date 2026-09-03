import sys
import sqlite3
from pathlib import Path

import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_06_llm_metadata_labeling as stage_06  # noqa: E402
from pipeline import utils  # noqa: E402


def test_locked_manual_metadata_overrides_only_configured_fields() -> None:
    detail, applied = stage_06.overlay_locked_manual_metadata(
        {
            "operator_jp": "JR東日本",
            "operator_en": "JR East",
            "submodel": "E127",
        },
        '{"operator_jp":"えちごトキめき鉄道",'
        '"operator_en":"Echigo Tokimeki Railway","submodel":"ET127"}',
        ["operator_jp", "operator_en"],
    )

    assert detail == {
        "operator_jp": "えちごトキめき鉄道",
        "operator_en": "Echigo Tokimeki Railway",
        "submodel": "E127",
    }
    assert applied == ["operator_jp", "operator_en"]


def test_missing_manual_value_leaves_llm_result() -> None:
    detail, applied = stage_06.overlay_locked_manual_metadata(
        {"operator_jp": "JR西日本", "operator_en": "JR West"},
        "{}",
        ["operator_jp", "operator_en"],
    )

    assert detail["operator_en"] == "JR West"
    assert applied == []


def test_locked_manual_metadata_config_rejects_non_stage_06_column() -> None:
    with pytest.raises(ValueError, match="非 Stage 06 字段"):
        stage_06.validate_locked_manual_metadata_columns(["operator_jp", "series"])


def test_enforce_manual_metadata_locks_updates_already_processed_image() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            operator_jp TEXT,
            operator_en TEXT,
            manual_metadata_json TEXT NOT NULL DEFAULT '{}',
            llm_metadata_processed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        INSERT INTO images (
            id, operator_jp, operator_en, manual_metadata_json,
            llm_metadata_processed
        ) VALUES (1, 'JR東日本', 'JR East', ?, 1)
        """,
        (
            '{"operator_jp":"青い森鉄道",'
            '"operator_en":"Aoimori Railway"}',
        ),
    )

    counts = stage_06.enforce_manual_metadata_locks(
        conn, ["operator_jp", "operator_en"]
    )

    assert counts == (1, 2)
    assert conn.execute(
        "SELECT operator_jp, operator_en, llm_metadata_processed FROM images"
    ).fetchone() == ("青い森鉄道", "Aoimori Railway", 1)


def test_migration_015_adds_manual_metadata_to_version_14_database() -> None:
    project_root = Path(__file__).resolve().parents[1]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 14")

    utils.apply_migrations(conn, project_root / "config" / "migrations")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    assert "manual_metadata_json" in columns
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 15
