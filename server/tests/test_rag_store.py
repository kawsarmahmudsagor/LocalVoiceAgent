import pytest

from voicebot.rag.store import ChunkStore, collection_name


def test_collection_name():
    assert collection_name("hf:BAAI/bge-small-en-v1.5") == "docs-hf-baai-bge-small-en-v1-5"


def test_add_query_delete(tmp_path):
    s = ChunkStore(tmp_path); s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    s.add("hf:x", "d1", "a.txt", ["north", "east"], [[1.0, 0.0], [0.0, 1.0]])
    hits = s.query("hf:x", [1.0, 0.0], 2)
    assert [(h.text, h.chunk, h.name) for h in hits] == [("north", 0, "a.txt"), ("east", 1, "a.txt")]
    assert hits[0].score == pytest.approx(1.0) and hits[1].score == pytest.approx(0.0, abs=1e-6)
    s.delete_doc("hf:x", "d1"); assert s.count("hf:x") == 0


def test_one_collection_per_embedder_and_metadata(tmp_path):
    s = ChunkStore(tmp_path); s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    assert s.info("hf:x") == {"embedder": "hf:x", "dim": 2, "chunk_size": 800, "chunk_overlap": 120}
    assert s.info("ollama:y") is None and s.count("ollama:y") == 0


def test_persists_across_instances(tmp_path):
    s = ChunkStore(tmp_path); s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    s.add("hf:x", "d1", "a.txt", ["north"], [[1.0, 0.0]])
    assert ChunkStore(tmp_path).count("hf:x") == 1


def test_doc_ids(tmp_path):
    s = ChunkStore(tmp_path)
    assert s.doc_ids("hf:x") == set()
    s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    s.add("hf:x", "d1", "a.txt", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    s.add("hf:x", "d2", "b.txt", ["c"], [[1.0, 0.0]])
    assert s.doc_ids("hf:x") == {"d1", "d2"}
    s.delete_doc("hf:x", "d1"); assert s.doc_ids("hf:x") == {"d2"}


def test_add_slices_by_client_max_batch(tmp_path, monkeypatch):
    s = ChunkStore(tmp_path); s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    monkeypatch.setattr(s._client, "get_max_batch_size", lambda: 3)
    from chromadb.api.models.Collection import Collection
    sizes, real_add = [], Collection.add

    def spy(self, *a, **kw):
        sizes.append(len(kw["ids"])); return real_add(self, *a, **kw)
    monkeypatch.setattr(Collection, "add", spy)
    texts = [f"t{n}" for n in range(8)]
    s.add("hf:x", "d1", "a.txt", texts, [[1.0, 0.0]] * 8)
    assert sizes == [3, 3, 2] and s.count("hf:x") == 8
    hits = s.query("hf:x", [1.0, 0.0], 8)
    assert sorted(h.chunk for h in hits) == list(range(8)) and {h.text for h in hits} == set(texts)


def test_add_with_start_offset(tmp_path):
    s = ChunkStore(tmp_path); s.reset("hf:x", dim=2, chunk_size=800, chunk_overlap=120)
    s.add("hf:x", "d1", "a.txt", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
    s.add("hf:x", "d1", "a.txt", ["c"], [[1.0, 0.0]], start=2)
    assert s.count("hf:x") == 3
    assert sorted(h.chunk for h in s.query("hf:x", [1.0, 0.0], 3)) == [0, 1, 2]
