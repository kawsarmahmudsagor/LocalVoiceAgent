import pytest

from voicebot.rag.embedders import EmbedderError
from voicebot.rag.library import Library, RagSettings
from voicebot.rag.service import EMPTY, RagError, RagService

from rag_fakes import FakeEmbedder


@pytest.fixture
def svc(tmp_path):
    events = []
    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug), on_event=events.append)
    s.start(); s.events = events; yield s; s.close()


def test_upload_index_retrieve(svc):
    assert svc.upload("geo.txt", b"Canberra is the capital of Australia.")["state"] == "queued"
    assert svc.wait_idle()
    r = svc.retrieve("What is the capital, Canberra?")
    assert r.context == "[geo.txt, part 1]\nCanberra is the capital of Australia." and r.chunks[0].score == pytest.approx(1.0)
    assert svc.status_event()["documents"][0]["state"] == "ready"


def test_threshold_top_k_and_char_cap(svc):
    svc.upload("geo.txt", b"Canberra one."); svc.upload("other.txt", b"Unrelated text."); svc.wait_idle()
    assert [h.name for h in svc.retrieve("canberra").chunks] == ["geo.txt"]          # other.txt scores 0 < 0.5
    svc.update({"min_score": 0.0, "top_k": 1}); assert len(svc.retrieve("canberra").chunks) == 1
    svc.update({"top_k": 2, "max_context_chars": 200}); r = svc.retrieve("canberra")
    assert len(r.context) <= 200


def test_off_or_empty_returns_empty(svc):
    assert svc.retrieve("x") == EMPTY
    svc.upload("geo.txt", b"Canberra."); svc.wait_idle(); svc.update({"enabled": False}); assert svc.retrieve("canberra") == EMPTY


def test_bad_upload_and_no_text(svc):
    assert "error" in svc.upload("x.doc", b"x")
    assert "error" in svc.upload("big.txt", b"x" * (25 * 1024 * 1024 + 1))
    svc.upload("blank.txt", b"   "); svc.wait_idle()
    assert svc.status_event()["documents"][0]["state"] == "no text"


def test_chunk_setting_change_reindexes(svc):
    svc.upload("geo.txt", b"Canberra. " * 200); svc.wait_idle(); before = svc.status_event()["documents"][0]["chunks"]
    svc.update({"chunk_size": 200, "chunk_overlap": 0}); svc.wait_idle()
    assert svc.status_event()["documents"][0]["chunks"] > before
    assert svc.config_event()["collection"]["state"] == "indexed"


def test_failed_switch_keeps_previous_embedder(tmp_path):     # Review Focus 4
    def make(slug):
        if slug == "ollama:missing": raise EmbedderError("model 'missing' not found")
        return FakeEmbedder(slug)
    s = RagService(tmp_path, RagSettings(), make_embedder=make); s.start()
    with pytest.raises(RagError, match="not found"): s.update({"embedder": "ollama:missing"})
    assert s.settings.embedder == "hf:BAAI/bge-small-en-v1.5"; s.close()


def test_dimension_change_rebuilds_collection(tmp_path):     # Review Focus 4
    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug, dim=2)); s.start()
    s.upload("a.txt", b"Canberra."); s.wait_idle(); s.close()
    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug, dim=3)); s.start()
    assert s.wait_idle()                                     # start() saw dim 3 vs collection dim 2: reset + reindex
    c = s.config_event()["collection"]; assert (c["dim"], c["state"]) == (3, "indexed")
    assert s.retrieve("canberra").chunks[0].name == "a.txt"; s.close()


def test_delete_while_queued_leaves_nothing(svc):            # Review Focus 5
    d = svc.upload("geo.txt", b"Canberra.")["doc_id"]; svc.delete(d); svc.wait_idle()
    assert svc.status_event()["documents"] == [] and svc.retrieve("canberra") == EMPTY


def test_restart_requeues_unfinished(tmp_path):
    lib = Library(tmp_path); d, _ = lib.add_file("geo.txt", b"Canberra."); lib.set_state(d.doc_id, "indexing")
    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug)); s.start()
    assert s.wait_idle() and s.status_event()["documents"][0]["state"] == "ready"; s.close()


def test_retrieve_failure_raises_embedder_error(tmp_path):
    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug)); s.start()
    s.upload("geo.txt", b"Canberra."); s.wait_idle(); s._embedder.fail = True
    with pytest.raises(EmbedderError): s.retrieve("canberra")
    s.close()


def test_duplicate_upload_reports_already_indexed(svc):
    svc.upload("geo.txt", b"Canberra."); svc.wait_idle()
    assert svc.upload("copy.txt", b"Canberra.")["state"] == "already indexed"
    assert len(svc.status_event()["documents"]) == 1


def test_corrupt_pdf_marks_failed_and_worker_survives(svc):
    svc.upload("bad.pdf", b"not a pdf"); svc.upload("geo.txt", b"Canberra."); assert svc.wait_idle()
    states = {d["name"]: d["state"] for d in svc.status_event()["documents"]}
    assert states["bad.pdf"].startswith("failed:") and states["geo.txt"] == "ready"


def test_switch_mid_index_finishes_in_new_collection(tmp_path, monkeypatch):
    import threading
    import voicebot.rag.service as svc_mod
    started, release = threading.Event(), threading.Event()
    real_load = svc_mod.load_text

    def slow_load(path):          # runs on the worker with no lock held
        if not release.is_set():
            started.set(); release.wait(5)
        return real_load(path)
    monkeypatch.setattr(svc_mod, "load_text", slow_load)

    s = RagService(tmp_path, RagSettings(enabled=True), make_embedder=lambda slug: FakeEmbedder(slug)); s.start()
    s.update({"embedder": "hf:other"}); s.update({"embedder": "hf:BAAI/bge-small-en-v1.5"})  # both collections exist, empty
    assert s.wait_idle()                      # let the queued empty reindex jobs drain first
    doc_id = s.upload("geo.txt", b"Canberra.")["doc_id"]
    assert started.wait(5)
    s.update({"embedder": "hf:other"})        # no-rebuild path: both doc-id sets are empty
    release.set()
    assert s.wait_idle()
    assert s.status_event()["documents"][0]["state"] == "ready"
    assert s._store.doc_ids("hf:other") == {doc_id}
    assert s.retrieve("canberra").chunks[0].name == "geo.txt"
    s.close()


def test_deleted_doc_not_resurrected_by_switching_back(svc):     # Review Focus 5
    svc.upload("a.txt", b"Canberra alpha."); svc.wait_idle()
    svc.update({"embedder": "hf:other"}); svc.wait_idle()
    for d in svc.status_event()["documents"]: svc.delete(d["doc_id"])
    svc.upload("b.txt", b"Canberra beta."); svc.wait_idle()
    svc.update({"embedder": "hf:BAAI/bge-small-en-v1.5"}); svc.wait_idle()
    ctx = svc.retrieve("canberra").context
    assert "beta" in ctx and "alpha" not in ctx
