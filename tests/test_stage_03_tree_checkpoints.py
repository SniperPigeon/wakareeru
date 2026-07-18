import sqlite3
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_03_manifest_crawling as stage_03  # noqa: E402


def _checkpoint_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE category_tree_checkpoints (
            series TEXT NOT NULL,
            root_category TEXT NOT NULL,
            category TEXT NOT NULL,
            remaining_depth INTEGER NOT NULL,
            completed_at TEXT NOT NULL,
            PRIMARY KEY (series, root_category, category)
        )
        """
    )
    return conn


def test_subtree_depth_coverage_respects_finite_and_unlimited_depth() -> None:
    assert stage_03.subtree_depth_covers(3, 2)
    assert not stage_03.subtree_depth_covers(2, 3)
    assert stage_03.subtree_depth_covers(-1, 3)
    assert stage_03.subtree_depth_covers(-1, -1)
    assert not stage_03.subtree_depth_covers(3, -1)


def test_completed_root_subtree_skips_remote_discovery(monkeypatch) -> None:
    conn = _checkpoint_conn()
    stage_03.mark_subtree_checkpoint(conn, "0系", "Shinkansen 0", "Shinkansen 0", 5)
    conn.commit()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed subtree must not access Commons")

    monkeypatch.setattr(stage_03, "fetch_subcategories", fail_if_called)
    monkeypatch.setattr(stage_03, "build_image_records", fail_if_called)

    records, complete = stage_03.crawl_category_records_with_checkpoint(
        conn=conn,
        row=pd.Series({"series": "0系", "commons_root_category": "Shinkansen 0"}),
        category="Shinkansen 0",
        path=["Shinkansen 0"],
        depth=0,
        max_depth=5,
        max_files_per_category=50,
        visited_categories={},
    )

    assert records == []
    assert complete is True


def test_subtree_checkpoint_keeps_deepest_completed_depth() -> None:
    conn = _checkpoint_conn()
    stage_03.mark_subtree_checkpoint(conn, "0系", "Shinkansen 0", "Shinkansen 0", 2)
    stage_03.mark_subtree_checkpoint(conn, "0系", "Shinkansen 0", "Shinkansen 0", 5)
    stage_03.mark_subtree_checkpoint(conn, "0系", "Shinkansen 0", "Shinkansen 0", 3)

    assert stage_03.get_subtree_checkpoint_depth(
        conn, "0系", "Shinkansen 0", "Shinkansen 0"
    ) == 5

    stage_03.mark_subtree_checkpoint(conn, "0系", "Shinkansen 0", "Shinkansen 0", -1)
    assert stage_03.get_subtree_checkpoint_depth(
        conn, "0系", "Shinkansen 0", "Shinkansen 0"
    ) == -1
