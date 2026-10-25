import pytest

from traffic_agent import config


@pytest.fixture(autouse=True)
def _isolated_project_paths(tmp_path, monkeypatch):
    """Relative config paths (evidence/, data/) resolve into a temp dir, so tests
    never write challans or snapshots into the real project folders."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
