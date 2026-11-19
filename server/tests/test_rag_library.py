import pytest

from voicebot.rag.library import Library, RagSettings


def test_settings_validation():
    s = RagSettings().updated({"top_k": 5, "enabled": True}); assert (s.top_k, s.enabled) == (5, True)
    with pytest.raises(ValueError) as e: RagSettings().updated({"top_k": 0, "min_score": 2, "bogus": 1})
    assert "top_k" in str(e.value) and "min_score" in str(e.value) and "bogus" in str(e.value)
    with pytest.raises(ValueError, match="chunk_overlap"): RagSettings().updated({"chunk_size": 200, "chunk_overlap": 101})
    with pytest.raises(ValueError, match="embedder"): RagSettings().updated({"embedder": "openai:x"})


def test_settings_rejects_wrong_types_and_empty_embedder():
    for patch in ({"top_k": True}, {"top_k": "3"}, {"enabled": 1}, {"min_score": "x"}, {"embedder": "hf:"}, {"embedder": "ollama:"}):
        with pytest.raises(ValueError):
            RagSettings().updated(patch)
    assert RagSettings().updated({"embedder": "ollama:nomic-embed-text"}).embedder == "ollama:nomic-embed-text"
    assert RagSettings().updated({"min_score": 1}).min_score == 1.0


def test_add_dedupes_by_content(tmp_path):              # Review Focus 2
    lib = Library(tmp_path)
    a, new = lib.add_file("a.txt", b"same"); b, again = lib.add_file("copy.txt", b"same")
    assert new and not again and b.doc_id == a.doc_id and len(lib.documents()) == 1
    assert lib.path(a.doc_id).read_bytes() == b"same" and a.state == "queued"


def test_manifest_and_settings_survive_restart(tmp_path):
    lib = Library(tmp_path); d, _ = lib.add_file("a.md", b"# hi"); lib.set_state(d.doc_id, "ready", chunks=1)
    lib.save_settings(RagSettings(top_k=7))
    lib2 = Library(tmp_path)
    assert lib2.get(d.doc_id).state == "ready" and lib2.get(d.doc_id).chunks == 1
    assert lib2.load_settings(RagSettings()).top_k == 7


def test_corrupt_settings_fall_back_and_remove(tmp_path):
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    lib = Library(tmp_path); assert lib.load_settings(RagSettings()) == RagSettings()
    d, _ = lib.add_file("a.txt", b"x"); assert lib.remove(d.doc_id) and not lib.path(d.doc_id).exists()
    assert not lib.remove("nope")
