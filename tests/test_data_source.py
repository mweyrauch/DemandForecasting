import sys
import types

import pytest

from dfi.data import resolve_data_dir


def test_local_source(tmp_path):
    assert resolve_data_dir({"source": "local", "dir": str(tmp_path)}) == tmp_path


def test_kagglehub_source_finds_nested_files(tmp_path, monkeypatch):
    nested = tmp_path / "sub" / "m5"
    nested.mkdir(parents=True)
    (nested / "calendar.csv").write_text("d\n")
    fake = types.SimpleNamespace(competition_download=lambda handle: str(tmp_path))
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    cfg = {"source": "kagglehub", "kagglehub": {"kind": "competition", "handle": "x"}}
    assert resolve_data_dir(cfg) == nested


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        resolve_data_dir({"source": "foo"})
