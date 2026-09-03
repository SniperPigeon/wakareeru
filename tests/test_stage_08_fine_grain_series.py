import sqlite3
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from pipeline import stage_08_fine_grain_series as stage_08  # noqa: E402


def test_private_railway_family_rules_merge_near_identical_subseries() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rules_path = project_root / "config" / "manual_fine_grained_series.csv"
    series_to_expected = {
        "東武50000型": "東武50000系",
        "東武50050型": "東武50000系",
        "東武50070型": "東武50000系",
        "東武50090型": "東武50000系",
        "東武9000型": "東武9000・9050型",
        "東武9050型": "東武9000・9050型",
        "東武70000型": "東武70000系",
        "東武70090型": "東武70000系",
        "東急2020系": "東急2020・3020・6020系",
        "東急3020系": "東急2020・3020・6020系",
        "東急6020系": "東急2020・3020・6020系",
    }

    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            """
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                series TEXT NOT NULL,
                submodel TEXT,
                bandai TEXT,
                special_formation TEXT,
                special_livery TEXT,
                operator_en TEXT,
                fine_grained_series TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO images(id, series) VALUES (?, ?)",
            enumerate(series_to_expected, start=1),
        )

        stage_08.apply_fine_grained_labels(conn, rules_path)

        actual = dict(conn.execute("SELECT series, fine_grained_series FROM images"))

    assert actual == series_to_expected
