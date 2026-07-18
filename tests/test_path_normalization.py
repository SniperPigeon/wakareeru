import os
import unicodedata
from pathlib import Path

import pytest

from pipeline.path_normalization import collect_rename_plan, validate_rename_plan


def _nfd_name() -> str:
    return unicodedata.normalize("NFD", "ばんえつ.jpg")


def _nfc_name() -> str:
    return unicodedata.normalize("NFC", "ばんえつ.jpg")


def test_collect_rename_plan_keeps_real_nfd_to_nfc_rename(tmp_path: Path) -> None:
    source = tmp_path / _nfd_name()
    source.write_bytes(b"image")
    if (tmp_path / _nfc_name()).exists():
        pytest.skip("filesystem resolves NFC and NFD names to the same entry")

    plans = collect_rename_plan(tmp_path)

    assert len(plans) == 1
    assert plans[0].source == source
    assert plans[0].target == tmp_path / _nfc_name()


def test_collect_rename_plan_skips_same_inode_alias(tmp_path: Path) -> None:
    source = tmp_path / _nfd_name()
    target = tmp_path / _nfc_name()
    source.write_bytes(b"image")
    if not target.exists():
        os.link(source, target)

    assert source.samefile(target)

    assert collect_rename_plan(tmp_path) == []


def test_validate_rename_plan_rejects_different_existing_target(tmp_path: Path) -> None:
    source = tmp_path / _nfd_name()
    target = tmp_path / _nfc_name()
    source.write_bytes(b"source")
    if target.exists():
        pytest.skip("filesystem resolves NFC and NFD names to the same entry")
    target.write_bytes(b"target")

    plans = collect_rename_plan(tmp_path)
    with pytest.raises(FileExistsError, match="normalization collision"):
        validate_rename_plan(plans)
