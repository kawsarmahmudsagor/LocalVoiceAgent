import tomllib
from pathlib import Path


def test_multipart_is_a_core_dependency():
    # The upload route (UploadFile) is always registered, and FastAPI raises at registration
    # without python-multipart: the bot must start without the optional `rag` extra.
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert any(d.startswith("python-multipart") for d in project["dependencies"])
    assert not any(d.startswith("python-multipart") for d in project["optional-dependencies"]["rag"])
