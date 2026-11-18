import json
from types import SimpleNamespace as NS
from voicebot.config import REPO_ROOT
from voicebot.catalog import load_catalog
from voicebot.bench.selftest import check_imports, check_torch, update_status

POOL = REPO_ROOT / "bench" / "pool.toml"

def fake_torch(ver="2.7.1+cu128", cuda=True, arch=("sm_90", "sm_120")):
    return NS(__version__=ver, cuda=NS(is_available=lambda: cuda, get_arch_list=lambda: list(arch)))

def test_check_torch():
    assert check_torch(fake_torch()) is None
    assert "2.6.0" in check_torch(fake_torch(ver="2.6.0+cu124"))
    assert "sm_120" in check_torch(fake_torch(arch=("sm_90",)))
    assert check_torch(fake_torch(cuda=False, arch=())) is None

def test_check_imports_records_failures():
    entries = [e for e in load_catalog(POOL) if e.family == "kokoro"]
    def importer(name):
        if name == "RealtimeTTS": raise ImportError("No module named kokoro")
    assert check_imports(entries, importer=importer) == {"kokoro": "No module named kokoro"}

def test_update_status_merges(tmp_path):
    p = tmp_path / "s.json"; update_status(p, {"a": None}); update_status(p, {"b": "torch 2.6.0 != 2.7.1+cu128"})
    assert json.loads(p.read_text()) == {"a": {"state": "ok"}, "b": {"state": "dropped", "reason": "torch 2.6.0 != 2.7.1+cu128"}}
