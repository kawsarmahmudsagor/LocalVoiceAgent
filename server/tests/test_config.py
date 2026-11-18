import pytest
from voicebot.config import Settings, ConfigError, REPO_ROOT

def test_defaults_match_spec():
    s = Settings.load(env={})
    assert (s.llm_backend, s.audio_io) == ("llamacpp", "browser")
    assert (s.hf_model_id, s.hf_load_in_4bit) == ("Qwen/Qwen3.5-9B", True)
    assert s.llama_autostart is True
    assert s.llama_server_path == REPO_ROOT / "tools/llama/llama-server.exe"
    assert s.llama_hf_repo == "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    assert (s.llama_host, s.llama_port, s.llama_ctx, s.llama_start_timeout) == ("127.0.0.1", 8081, 8192, 900)
    assert s.llama_base_url == "http://127.0.0.1:8081"
    assert (s.stt_model, s.stt_device, s.stt_compute_type, s.stt_silence_sec) == ("small.en", "cuda", "int8", 0.6)
    assert s.tts_voice == "af_heart"
    assert s.system_prompt == ("You are a friendly voice assistant. Reply in one to three short spoken "
                               "sentences. Never use lists, markdown, code blocks or emoji.")
    assert (s.llm_temperature, s.llm_top_p, s.llm_max_tokens, s.history_turns) == (0.7, 0.8, 256, 10)
    assert (s.web_host, s.web_port) == ("127.0.0.1", 8000)

def test_overrides_and_derived_base_url():
    s = Settings.load(env={"LLM_BACKEND": "transformers", "LLAMA_PORT": "9000", "LLAMA_AUTOSTART": "No"})
    assert s.llm_backend == "transformers" and s.llama_port == 9000 and s.llama_autostart is False
    assert s.llama_base_url == "http://127.0.0.1:9000"
    assert Settings.load(env={"LLAMA_BASE_URL": "http://box:1234"}).llama_base_url == "http://box:1234"

@pytest.mark.parametrize("var,val", [("LLM_BACKEND", "ollama"), ("AUDIO_IO", "phone"),
    ("LLAMA_PORT", "abc"), ("LLM_TEMPERATURE", "hot"), ("HF_LOAD_IN_4BIT", "maybe")])
def test_invalid_value_names_variable(var, val):
    with pytest.raises(ConfigError, match=var):
        Settings.load(env={var: val})

def test_reads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_IO", raising=False)
    (tmp_path / ".env").write_text("AUDIO_IO=server\n")
    assert Settings.load(env_file=tmp_path / ".env", platform="win32").audio_io == "server"

def test_linux_defaults_and_server_audio_rejected():
    s = Settings.load(env={}, platform="linux")
    assert s.llama_runtime == "docker" and s.llama_image == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert Settings.load(env={}, platform="win32").llama_runtime == "binary"
    with pytest.raises(ConfigError, match="server audio is not supported on the VM"):
        Settings.load(env={"AUDIO_IO": "server"}, platform="linux")

def test_ssl_pair_required():
    with pytest.raises(ConfigError, match="WEB_SSL_KEYFILE"):
        Settings.load(env={"WEB_SSL_CERTFILE": "certs/cert.pem"}, platform="linux")
    s = Settings.load(env={"WEB_SSL_CERTFILE": "certs/cert.pem", "WEB_SSL_KEYFILE": "certs/key.pem"}, platform="linux")
    assert s.web_ssl_certfile.name == "cert.pem" and s.web_ssl_certfile.is_absolute()
