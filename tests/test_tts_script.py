import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "test_tts.py"


def load_script():
    spec = importlib.util.spec_from_file_location("test_tts_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None, status: int = 200):
        self._body = body
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


def test_fetch_health_requests_service_health(monkeypatch):
    script = load_script()
    requested = {}

    def fake_urlopen(request, timeout):
        requested["url"] = request.full_url
        requested["timeout"] = timeout
        return FakeResponse(json.dumps({"status": "ok"}).encode("utf-8"))

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)

    result = script.fetch_health("http://127.0.0.1:8004", timeout=3)

    assert requested == {"url": "http://127.0.0.1:8004/health", "timeout": 3}
    assert result == {"status": "ok"}


def test_synthesize_wav_posts_openai_payload_with_auth(monkeypatch):
    script = load_script()
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(b"RIFFxxxxWAVEdata", {"Content-Type": "audio/wav"})

    monkeypatch.setattr(script.urllib.request, "urlopen", fake_urlopen)

    audio, content_type = script.synthesize(
        base_url="http://127.0.0.1:8004/",
        text="Xin chào",
        voice="Thục Đoan",
        output_format="wav",
        api_key="secret",
        timeout=10,
    )

    assert audio.startswith(b"RIFF")
    assert content_type == "audio/wav"
    assert captured["url"] == "http://127.0.0.1:8004/v1/audio/speech"
    assert captured["method"] == "POST"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"] == {
        "model": "vieneu-v3-turbo",
        "input": "Xin chào",
        "voice": "Thục Đoan",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert captured["timeout"] == 10


def test_validate_audio_rejects_empty_or_invalid_wav():
    script = load_script()

    with pytest.raises(SystemExit, match="empty"):
        script.validate_audio(b"", "wav")

    with pytest.raises(SystemExit, match="valid WAV"):
        script.validate_audio(b"not wav", "wav")
