from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tools import crop_duplicate_review_gradio as review_tool


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "config" / "migrations"


def _create_review_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                series TEXT,
                fine_grained_series TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE crops (
                id INTEGER PRIMARY KEY,
                image_id INTEGER NOT NULL,
                detector_score REAL,
                box_x1 REAL,
                box_y1 REAL,
                box_x2 REAL,
                box_y2 REAL,
                noise_review_label TEXT,
                noise_review_note TEXT,
                noise_reviewed_at TEXT,
                noise_review_score_col TEXT,
                manual_corrected_label TEXT,
                manual_corrected_at TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO images(id, series, fine_grained_series)
            VALUES (?, ?, ?)
            """,
            [
                (1, "A", "A"),
                (2, "B", "B"),
                (3, "B", "B"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO crops(
                id, image_id, detector_score, box_x1, box_y1, box_x2, box_y2,
                manual_corrected_label, manual_corrected_at
            ) VALUES (?, ?, 0.9, 0, 0, 100, 100, ?, ?)
            """,
            [
                (1, 1, "old-correction", "2026-01-01"),
                (2, 2, None, None),
                (3, 3, None, None),
            ],
        )
        conn.executescript(
            (MIGRATIONS_DIR / "013_add_crop_duplicate_groups.sql").read_text(
                encoding="utf-8"
            )
        )
        conn.executescript(
            (MIGRATIONS_DIR / "014_add_crop_duplicate_exclusion_reason.sql").read_text(
                encoding="utf-8"
            )
        )
        conn.execute(
            """
            INSERT INTO crop_duplicate_groups(
                group_key, source_sha1, detector_model, nms_iou_threshold,
                representative_crop_id, member_crop_ids_json,
                candidate_labels_json, member_count, candidate_label_count,
                box_x1, box_y1, box_x2, box_y2
            ) VALUES (
                'group-1', 'sha1', 'gdino', 0.5, 1, ?, ?, 3, 2,
                0, 0, 100, 100
            )
            """,
            (json.dumps([1, 2, 3]), json.dumps(["A", "B"])),
        )
        conn.commit()


def _configure_tool(path: Path) -> None:
    review_tool.DB_PATH = path
    review_tool.CONFIG = {
        "crops_storage": {
            "label_column": "fine_grained_series",
        }
    }


def test_group_exclusion_writes_structured_reason_to_all_crops(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    _create_review_db(db_path)
    _configure_tool(db_path)

    review_tool.save_resolution(
        group_id=1,
        status=review_tool.REVIEW_STATUS_EXCLUDED,
        resolved_label=None,
        exclusion_reason="bad_crop",
        note="whole group is unusable",
    )

    with sqlite3.connect(db_path) as conn:
        group = conn.execute(
            """
            SELECT review_status, exclusion_reason, review_note
            FROM crop_duplicate_groups
            WHERE id = 1
            """
        ).fetchone()
        crops = conn.execute(
            """
            SELECT noise_review_label, noise_review_note,
                   noise_review_score_col, manual_corrected_label,
                   manual_corrected_at
            FROM crops
            ORDER BY id
            """
        ).fetchall()

    assert group == ("excluded", "bad_crop", "whole group is unusable")
    assert crops == [
        ("bad_crop", "whole group is unusable", "crop_duplicate_review", None, None),
        ("bad_crop", "whole group is unusable", "crop_duplicate_review", None, None),
        ("bad_crop", "whole group is unusable", "crop_duplicate_review", None, None),
    ]


def test_member_exclusion_reconciles_or_removes_group(tmp_path: Path) -> None:
    db_path = tmp_path / "review.sqlite"
    _create_review_db(db_path)
    _configure_tool(db_path)

    review_tool.exclude_member(
        group_id=1,
        crop_id=1,
        exclusion_reason="out_of_label_space",
        note="not a train",
    )

    with sqlite3.connect(db_path) as conn:
        group = conn.execute(
            """
            SELECT member_crop_ids_json, candidate_labels_json, member_count,
                   candidate_label_count, review_status, resolved_label
            FROM crop_duplicate_groups
            WHERE id = 1
            """
        ).fetchone()
        excluded_crop = conn.execute(
            """
            SELECT noise_review_label, noise_review_score_col,
                   manual_corrected_label
            FROM crops
            WHERE id = 1
            """
        ).fetchone()

    assert json.loads(group[0]) == [2, 3]
    assert json.loads(group[1]) == ["B"]
    assert group[2:] == (2, 1, "auto_resolved", "B")
    assert excluded_crop == ("out_of_label_space", "crop_duplicate_review", None)

    review_tool.exclude_member(
        group_id=1,
        crop_id=2,
        exclusion_reason="bad_crop",
        note="bad framing",
    )

    with sqlite3.connect(db_path) as conn:
        remaining_group_count = conn.execute(
            "SELECT COUNT(*) FROM crop_duplicate_groups"
        ).fetchone()[0]

    assert remaining_group_count == 0
