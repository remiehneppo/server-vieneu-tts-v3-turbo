import importlib
import math
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_app(monkeypatch):
    monkeypatch.setenv("VIENEU_PRELOAD", "false")
    monkeypatch.setenv("TTS_API_KEY", "")
    module = importlib.import_module("app")
    module.engine = module.FakeEngine(sample_rate=module.SAMPLE_RATE)
    return module


def test_health_reports_default_voice_and_voices(monkeypatch):
    module = load_app(monkeypatch)
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["sample_rate"] == 24000
    assert body["default_voice"] == "Thục Đoan (Nữ - Miền Nam)"
    assert body["loaded"] is True
    assert "Thục Đoan" in body["voices"]


def test_openai_speech_returns_wav_for_default_voice(monkeypatch):
    module = load_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "vieneu-v2-turbo",
            "input": "Xin chào, tôi là Xiaozhi.",
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert response.content[8:12] == b"WAVE"


def test_index_stream_tts_returns_raw_pcm_by_default(monkeypatch):
    module = load_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post("/tts", json={"text": "Xin chào", "character": "Thục Đoan"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/pcm"
    assert len(response.content) > 0
    assert not response.content.startswith(b"RIFF")
    assert len(response.content) % 2 == 0


def test_benchmark_returns_latency_and_resource_metrics(monkeypatch, tmp_path):
    module = load_app(monkeypatch)
    monkeypatch.setattr(module, "BENCHMARK_DIR", tmp_path)
    monkeypatch.setattr(module, "BENCHMARK_LATEST_PATH", tmp_path / "latest.json")
    client = TestClient(module.app)

    response = client.post("/benchmark", json={"text": "Xin chào benchmark"})

    assert response.status_code == 200
    body = response.json()
    assert body["cases"][0]["total_latency_ms"] >= 0
    assert body["cases"][0]["synthesis_ms"] >= 0
    assert body["cases"][0]["chars_per_sec"] > 0
    assert math.isfinite(body["cases"][0]["realtime_factor"])
    assert body["resources"]["ram_mb_peak"] >= body["resources"]["ram_mb_start"]
    assert (tmp_path / "latest.json").exists()


def test_benchmark_still_returns_when_latest_file_is_not_writable(monkeypatch, tmp_path):
    module = load_app(monkeypatch)

    class DenyingPath:
        def write_text(self, *_args, **_kwargs):
            raise PermissionError("permission denied")

    monkeypatch.setattr(module, "BENCHMARK_DIR", tmp_path)
    monkeypatch.setattr(module, "BENCHMARK_LATEST_PATH", DenyingPath())
    client = TestClient(module.app)

    response = client.post("/benchmark", json={"text": "Xin chào benchmark"})

    assert response.status_code == 200
    body = response.json()
    assert body["cases"][0]["total_latency_ms"] >= 0
    assert "permission denied" in body["benchmark_write_error"]


def test_invalid_voice_is_rejected(monkeypatch):
    module = load_app(monkeypatch)
    client = TestClient(module.app)

    response = client.post("/tts", json={"text": "Xin chào", "character": "khong-co"})

    assert response.status_code == 400
    assert "voice" in response.json()["detail"].lower()


def test_real_engine_uses_vieneu_turbo_constructor(monkeypatch, tmp_path):
    module = load_app(monkeypatch)
    captured = {}

    class FakeVieneuClient:
        pass

    def fake_vieneu(**kwargs):
        captured.update(kwargs)
        return FakeVieneuClient()

    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=fake_vieneu))
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path / "models"))

    engine = module.VieNeuEngine()

    assert isinstance(engine.tts, FakeVieneuClient)
    assert captured["mode"] == "v3turbo"
    assert captured["backbone_repo"] == "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    assert captured["device"] == "cpu"


def test_real_engine_resolves_requested_voice_to_sdk_preset(monkeypatch, tmp_path):
    module = load_app(monkeypatch)
    preset_voice = {"codes": [0.1] * 128, "text": ""}
    captured = {}

    class FakeVieneuClient:
        def list_preset_voices(self):
            return [
                ("Bích Ngọc (Nữ - Miền Bắc)", "Bích Ngọc (Nữ - Miền Bắc)"),
                ("Thục Đoan (Nữ - Miền Nam)", "Thục Đoan (Nữ - Miền Nam)"),
            ]

        def get_preset_voice(self, voice_name=None):
            captured["preset_name"] = voice_name
            return preset_voice

        def infer(self, **kwargs):
            captured["infer_voice"] = kwargs["voice"]
            return module.pcm_to_wav(b"\0\0" * 100)

    monkeypatch.setitem(sys.modules, "vieneu", types.SimpleNamespace(Vieneu=lambda **_kwargs: FakeVieneuClient()))
    monkeypatch.setattr(module, "MODEL_DIR", str(tmp_path / "models"))

    engine = module.VieNeuEngine()
    engine.synthesize_pcm(
        "Xin chào",
        {
            "id": "thuc_doan",
            "name": "Thục Đoan",
            "label": "Thục Đoan (Nữ - Miền Nam)",
            "aliases": ["Thuc Doan"],
        },
        1.0,
    )

    assert captured["preset_name"] == "Thục Đoan (Nữ - Miền Nam)"
    assert captured["infer_voice"] is preset_voice


def test_default_model_dir_is_project_local(monkeypatch):
    monkeypatch.delenv("VIENEU_MODEL_DIR", raising=False)
    module = importlib.reload(importlib.import_module("app"))

    assert Path(module.MODEL_DIR) == module.DEFAULT_MODEL_DIR
    assert Path(module.MODEL_DIR).is_absolute()
    assert Path(module.MODEL_DIR).name == "models"
    assert Path(module.MODEL_DIR) != Path("/models")


def test_synthesis_requests_are_serialized(monkeypatch):
    module = load_app(monkeypatch)

    class ConcurrentDetectingEngine(module.FakeEngine):
        def __init__(self):
            super().__init__(sample_rate=module.SAMPLE_RATE)
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def synthesize_pcm(self, text, voice, speed):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            try:
                return super().synthesize_pcm(text, voice, speed)
            finally:
                with self.lock:
                    self.active -= 1

    detecting_engine = ConcurrentDetectingEngine()
    module.engine = detecting_engine
    client = TestClient(module.app, raise_server_exceptions=False)

    def request_tts():
        return client.post("/tts", json={"text": "Xin chào", "character": "Thục Đoan"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: request_tts(), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert detecting_engine.max_active == 1

def test_openai_speech_supports_ref_audio_clone(monkeypatch, tmp_path):
    module = load_app(monkeypatch)
    client = TestClient(module.app)
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(module.pcm_to_wav(b"\x00" * 480))

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "vieneu-v3-turbo",
            "input": "Thử nghiệm clone với file mẫu",
            "ref_audio": str(ref_wav),
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
