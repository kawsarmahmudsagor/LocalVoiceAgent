import subprocess, sys
from voicebot.config import Settings
from voicebot.llm.factory import make_backend

def test_picks_backend_class():
    assert type(make_backend(Settings.load(env={"LLM_BACKEND": "llamacpp"}))).__name__ == "LlamaCppBackend"
    assert type(make_backend(Settings.load(env={"LLM_BACKEND": "transformers"}))).__name__ == "TransformersBackend"

def test_llamacpp_path_never_imports_torch():
    code = ("import sys; from voicebot.config import Settings; from voicebot.llm.factory import make_backend; "
            "make_backend(Settings.load(env={'LLM_BACKEND': 'llamacpp'})); "
            "bad = {'torch', 'transformers', 'bitsandbytes'} & set(sys.modules); assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
