import sys
import sqlite3
from pathlib import Path

import pandas as pd


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_13c_crop_duplicate_detection as duplicate_stage  # noqa: E402


def _row(
    crop_id: int,
    *,
    label: str,
    box: tuple[float, float, float, float],
    image_id: int | None = None,
) -> dict:
    return {
        "crop_id": crop_id,
        "image_id": image_id or crop_id,
        "source_sha1": "same-sha1",
        "detector_model": "gdino",
        "nms_iou_threshold": 0.5,
        "detector_score": 0.9,
        "noise_review_label": None,
        "manual_corrected_label": None,
        "effective_label": label,
        "downloaded_path": f"img/{crop_id}.jpg",
        "box_x1": box[0],
        "box_y1": box[1],
        "box_x2": box[2],
        "box_y2": box[3],
    }


def test_build_duplicate_groups_keeps_distinct_vehicles_separate() -> None:
    rows = pd.DataFrame(
        [
            _row(1, label="E259", box=(0, 0, 100, 100)),
            _row(2, label="E261", box=(0.2, 0.1, 100.1, 100.0)),
            _row(3, label="E259", box=(200, 0, 300, 100)),
            _row(4, label="E261", box=(200.1, 0, 300, 100)),
        ]
    )

    groups = duplicate_stage.build_duplicate_groups(rows, iou_threshold=0.99)

    assert len(groups) == 2
    assert groups[0]["member_crop_ids"] == [1, 2]
    assert groups[1]["member_crop_ids"] == [3, 4]
    assert groups[0]["candidate_labels"] == ["E259", "E261"]
    assert groups[0]["review_status"] == duplicate_stage.REVIEW_STATUS_PENDING


def test_same_label_duplicate_group_is_auto_resolved() -> None:
    rows = pd.DataFrame(
        [
            _row(10, label="E259", box=(0, 0, 100, 100)),
            _row(11, label="E259", box=(0, 0, 100, 100)),
        ]
    )

    groups = duplicate_stage.build_duplicate_groups(rows, iou_threshold=0.99)

    assert len(groups) == 1
    assert groups[0]["review_status"] == duplicate_stage.REVIEW_STATUS_AUTO_RESOLVED
    assert groups[0]["resolved_label"] == "E259"


def test_matching_manual_review_is_preserved_only_for_unchanged_members() -> None:
    rows = pd.DataFrame(
        [
            _row(20, label="E259", box=(0, 0, 100, 100)),
            _row(21, label="E261", box=(0, 0, 100, 100)),
        ]
    )
    groups = duplicate_stage.build_duplicate_groups(rows, iou_threshold=0.99)
    group = groups[0]
    reviews = {
        group["group_key"]: {
            "member_crop_ids_json": "[20, 21]",
            "candidate_labels_json": '["E259", "E261"]',
            "review_status": "confirmed",
            "resolved_label": "E259",
            "review_note": "checked",
            "reviewed_at": "2026-01-01 00:00:00",
        }
    }

    duplicate_stage.preserve_matching_manual_reviews(groups, reviews)

    assert group["review_status"] == "confirmed"
    assert group["resolved_label"] == "E259"


def test_replace_duplicate_groups_writes_group_payload() -> None:
    rows = pd.DataFrame(
        [
            _row(30, label="E259", box=(0, 0, 100, 100)),
            _row(31, label="E261", box=(0, 0, 100, 100)),
        ]
    )
    groups = duplicate_stage.build_duplicate_groups(rows, iou_threshold=0.99)
    group = groups[0]
    group.update(
        {
            "global_top1_label": "E259",
            "global_top1_prob": 0.8,
            "global_top_k": [{"label": "E259", "probability": 0.8}],
            "proposed_label": "E259",
            "proposed_candidate_prob": 0.9,
            "proposed_candidate_margin": 0.8,
            "candidate_scores": [{"label": "E259", "probability": 0.9}],
            "proposal_model": "test-model",
        }
    )
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "migrations"
        / "013_add_crop_duplicate_groups.sql"
    )

    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE crops (id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO crops(id) VALUES (?)", [(30,), (31,)])
        conn.executescript(migration_path.read_text(encoding="utf-8"))
        duplicate_stage.replace_duplicate_groups(conn, groups)
        stored = conn.execute(
            """
            SELECT member_count, candidate_label_count, proposed_label, review_status
            FROM crop_duplicate_groups
            """
        ).fetchone()

    assert stored == (2, 2, "E259", "pending")
