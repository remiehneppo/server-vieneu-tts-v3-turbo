import audioop
import io
import json
import math
import os
import secrets
import struct
import subprocess
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_ID = os.getenv("VIENEU_MODEL_ID", "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF")
MODEL_DIR = os.getenv("VIENEU_MODEL_DIR", str(DEFAULT_MODEL_DIR))
MODEL_FILE = os.getenv("VIENEU_MODEL_FILE", "vieneu-tts-v2-turbo.gguf")
DEVICE = os.getenv("VIENEU_DEVICE", "cpu")
N_GPU_LAYERS = int(os.getenv("VIENEU_N_GPU_LAYERS", "0"))
THREADS = int(os.getenv("VIENEU_THREADS", str(max((os.cpu_count() or 2) - 1, 1))))
N_CTX = int(os.getenv("VIENEU_N_CTX", "2048"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("VIENEU_DEFAULT_VOICE", "Thục Đoan (Nữ - Miền Nam)")
API_KEY = os.getenv("TTS_API_KEY", "")
PRELOAD = os.getenv("VIENEU_PRELOAD", "true").lower() == "true"
USE_FAKE_ENGINE = os.getenv("VIENEU_FAKE", "false").lower() == "true"
BENCHMARK_DIR = Path(os.getenv("VIENEU_BENCHMARK_DIR", "benchmarks"))
BENCHMARK_LATEST_PATH = BENCHMARK_DIR / "latest.json"
VOICE_PATH = Path(__file__).with_name("voices.json")

engine: Optional["TTSEngine"] = None
last_benchmark: Optional[dict[str, Any]] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if PRELOAD:
        get_engine()
    yield


app = FastAPI(title="VieNeu TTS Server", version="1.0.0", lifespan=lifespan)
engine_lock = threading.Lock()


class SpeechRequest(BaseModel):
    model: str = Field(default="vieneu-v2-turbo")
    input: str
    voice: Optional[str] = None
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class IndexStreamRequest(BaseModel):
    text: str
    character: Optional[str] = None
    format: str = Field(default="pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class BenchmarkRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[list[str]] = None
    voice: Optional[str] = None
    format: str = Field(default="pcm")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    cold_start: bool = Field(default=False)


class TTSEngine:
    sample_rate: int

    def synthesize_pcm(self, text: str, voice: dict[str, Any], speed: float) -> bytes:
        raise NotImplementedError


class FakeEngine(TTSEngine):
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def synthesize_pcm(self, text: str, voice: dict[str, Any], speed: float) -> bytes:
        duration = max(0.25, min(2.0, len(text) / 24.0 / max(speed, 0.25)))
        frames = int(self.sample_rate * duration)
        freq = 420 + (sum(ord(ch) for ch in voice["name"]) % 180)
        samples = bytearray()
        for index in range(frames):
            value = int(9000 * math.sin(2 * math.pi * freq * index / self.sample_rate))
            samples.extend(struct.pack("<h", value))
        return bytes(samples)


class VieNeuEngine(TTSEngine):
    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE
        self.tts = self._load_sdk()
        self._sdk_voice_ids_by_key: Optional[dict[str, str]] = None

    def _load_sdk(self) -> Any:
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise RuntimeError(
                "Không import được SDK vieneu. Cài dependencies bằng `pip install -r requirements.txt`."
            ) from exc

        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        mode = os.getenv("VIENEU_MODE", "turbo")
        attempts = [
            {
                "mode": mode,
                "backbone_repo": MODEL_ID,
                "backbone_filename": MODEL_FILE,
                "device": DEVICE,
            },
            {"mode": mode, "backbone_repo": MODEL_ID, "backbone_filename": MODEL_FILE},
            {"mode": mode},
            {},
        ]
        last_error: Optional[Exception] = None
        for kwargs in attempts:
            try:
                return Vieneu(**kwargs)
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Không khởi tạo được Vieneu SDK: {last_error}")

    def synthesize_pcm(self, text: str, voice: dict[str, Any], speed: float) -> bytes:
        sdk_voice = self._resolve_sdk_voice(voice)
        audio = self._infer(text=text, voice=sdk_voice, speed=speed)
        wav_bytes = self._audio_to_wav_bytes(audio)
        return wav_to_pcm_24k_mono(wav_bytes)

    def _resolve_sdk_voice(self, voice: dict[str, Any]) -> Any:
        sdk_voice = voice.get("sdk_voice") or voice.get("label") or voice["name"]
        if not hasattr(self.tts, "get_preset_voice"):
            return sdk_voice

        voice_id = self._find_sdk_voice_id(voice, sdk_voice)
        try:
            return self.tts.get_preset_voice(voice_id)
        except Exception as exc:
            raise RuntimeError(f"Không lấy được preset voice từ VieNeu SDK: {sdk_voice}") from exc

    def _find_sdk_voice_id(self, voice: dict[str, Any], sdk_voice: str) -> Optional[str]:
        voice_ids = self._sdk_voice_ids()
        candidates = [
            sdk_voice,
            voice.get("label", ""),
            voice.get("name", ""),
            voice.get("id", ""),
            *voice.get("aliases", []),
        ]
        for candidate in candidates:
            key = normalize_voice_key(candidate)
            if key in voice_ids:
                return voice_ids[key]
        for candidate in candidates:
            key = normalize_voice_key(candidate)
            for sdk_key, voice_id in voice_ids.items():
                if key and (key in sdk_key or sdk_key in key):
                    return voice_id
        return sdk_voice

    def _sdk_voice_ids(self) -> dict[str, str]:
        if self._sdk_voice_ids_by_key is not None:
            return self._sdk_voice_ids_by_key
        voice_ids: dict[str, str] = {}
        if hasattr(self.tts, "list_preset_voices"):
            try:
                for description, voice_id in self.tts.list_preset_voices():
                    voice_ids[normalize_voice_key(str(description))] = str(voice_id)
                    voice_ids[normalize_voice_key(str(voice_id))] = str(voice_id)
            except Exception:
                pass
        self._sdk_voice_ids_by_key = voice_ids
        return voice_ids

    def _infer(self, text: str, voice: Any, speed: float) -> Any:
        attempts = [
            {"text": text, "voice": voice, "speed": speed},
            {"text": text, "voice": voice},
            {"text": text},
        ]
        for kwargs in attempts:
            try:
                return self.tts.infer(**kwargs)
            except TypeError:
                continue
        return self.tts.infer(text)

    def _audio_to_wav_bytes(self, audio: Any) -> bytes:
        if isinstance(audio, bytes):
            return audio
        if isinstance(audio, bytearray):
            return bytes(audio)
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, data = audio
            return pcm_to_wav(array_to_pcm16(data), int(sample_rate))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.tts.save(audio, tmp_path)
            return Path(tmp_path).read_bytes()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class ResourceSampler:
    def __init__(self) -> None:
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cpu_values: list[float] = []
        self.ram_values: list[float] = []
        self.gpu_utils: list[float] = []
        self.vram_values: list[float] = []
        self.gpu_name: Optional[str] = None
        self.process = None
        try:
            import psutil

            self.process = psutil.Process(os.getpid())
            self.process.cpu_percent(interval=None)
        except Exception:
            self.process = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        return {
            "cpu_percent_avg": average(self.cpu_values),
            "cpu_percent_max": max(self.cpu_values or [0.0]),
            "ram_mb_start": self.ram_values[0] if self.ram_values else 0.0,
            "ram_mb_peak": max(self.ram_values or [0.0]),
            "ram_mb_end": self.ram_values[-1] if self.ram_values else 0.0,
            "gpu_name": self.gpu_name,
            "gpu_util_avg": average(self.gpu_utils) if self.gpu_utils else None,
            "gpu_util_max": max(self.gpu_utils) if self.gpu_utils else None,
            "vram_mb_start": self.vram_values[0] if self.vram_values else None,
            "vram_mb_peak": max(self.vram_values) if self.vram_values else None,
            "vram_mb_end": self.vram_values[-1] if self.vram_values else None,
        }

    def _loop(self) -> None:
        while self.running:
            self._sample()
            time.sleep(0.1)
        self._sample()

    def _sample(self) -> None:
        if self.process:
            try:
                self.cpu_values.append(float(self.process.cpu_percent(interval=None)))
                self.ram_values.append(float(self.process.memory_info().rss / 1024 / 1024))
            except Exception:
                pass
        gpu = sample_nvidia_gpu()
        if gpu:
            self.gpu_name = gpu["name"]
            self.gpu_utils.append(gpu["util"])
            self.vram_values.append(gpu["vram"])


def load_voices() -> list[dict[str, Any]]:
    return json.loads(VOICE_PATH.read_text(encoding="utf-8"))


VOICES = load_voices()


def require_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token bearer không hợp lệ")
    token = authorization[len("Bearer "):]
    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Token bearer không hợp lệ")


def normalize_voice_key(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())


def resolve_voice(value: Optional[str]) -> dict[str, Any]:
    requested = value or DEFAULT_VOICE
    needle = normalize_voice_key(requested)
    for voice in VOICES:
        candidates = [voice["id"], voice["name"], voice["label"], *voice.get("aliases", [])]
        if any(normalize_voice_key(candidate) == needle for candidate in candidates):
            return voice
    for voice in VOICES:
        candidates = [voice["name"], voice["label"], *voice.get("aliases", [])]
        if any(needle in normalize_voice_key(candidate) for candidate in candidates):
            return voice
    raise HTTPException(status_code=400, detail=f"Voice không hợp lệ: {requested}")


def get_engine() -> TTSEngine:
    global engine
    if engine is None:
        engine = FakeEngine(sample_rate=SAMPLE_RATE) if USE_FAKE_ENGINE else VieNeuEngine()
    return engine


def validate_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text không được rỗng")
    if len(cleaned) > int(os.getenv("VIENEU_MAX_CHARS", "2000")):
        raise HTTPException(status_code=413, detail="Text quá dài")
    return cleaned


def validate_format(value: str) -> str:
    fmt = value.lower()
    if fmt not in {"pcm", "wav"}:
        raise HTTPException(status_code=400, detail="Format chỉ hỗ trợ pcm hoặc wav")
    return fmt


def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return wav_buf.getvalue()


def wav_to_pcm_24k_mono(wav_bytes: bytes) -> bytes:
    if not wav_bytes.startswith(b"RIFF"):
        return ensure_even_pcm(wav_bytes)
    wav_buf = io.BytesIO(wav_bytes)
    with wave.open(wav_buf, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
    if channels > 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if sample_rate != SAMPLE_RATE:
        pcm, _state = audioop.ratecv(pcm, 2, 1, sample_rate, SAMPLE_RATE, None)
    return ensure_even_pcm(pcm)


def ensure_even_pcm(pcm: bytes) -> bytes:
    return pcm if len(pcm) % 2 == 0 else pcm[:-1]


def array_to_pcm16(data: Any) -> bytes:
    try:
        import numpy as np

        array = np.asarray(data)
        if array.dtype.kind == "f":
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767).astype("<i2")
        else:
            array = array.astype("<i2")
        if array.ndim > 1:
            array = array.mean(axis=1).astype("<i2")
        return array.tobytes()
    except Exception:
        return bytes(data)


def synthesize(text: str, voice_name: Optional[str], fmt: str, speed: float) -> tuple[bytes, str, float, dict[str, Any]]:
    clean_text = validate_text(text)
    voice = resolve_voice(voice_name)
    output_format = validate_format(fmt)
    start = time.perf_counter()
    with engine_lock:
        pcm = get_engine().synthesize_pcm(clean_text, voice, speed)
    synthesis_ms = (time.perf_counter() - start) * 1000
    if output_format == "wav":
        return pcm_to_wav(pcm), "audio/wav", synthesis_ms, voice
    return pcm, "audio/pcm", synthesis_ms, voice


def audio_duration_sec(pcm: bytes) -> float:
    return len(pcm) / (SAMPLE_RATE * 2)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_nvidia_gpu() -> Optional[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
    except Exception:
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    if not first:
        return None
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "util": float(parts[1]), "vram": float(parts[2])}


def benchmark_cases(request: BenchmarkRequest) -> list[str]:
    if request.text:
        return [request.text]
    if request.texts:
        return request.texts
    return [
        "Xin chào, tôi là trợ lý giọng nói Xiaozhi.",
        "Hôm nay hệ thống sẽ kiểm tra tốc độ tổng hợp giọng nói tiếng Việt với một câu dài hơn để đo realtime factor chính xác hơn.",
        "Xiaozhi có thể nói tiếng Việt và English trong cùng một câu để kiểm tra code-switching.",
    ]


def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    global engine, last_benchmark
    cold_start_ms = None
    if request.cold_start:
        engine = None
        cold_start = time.perf_counter()
        get_engine()
        cold_start_ms = (time.perf_counter() - cold_start) * 1000

    sampler = ResourceSampler()
    sampler.start()
    cases = []
    first_latency: Optional[float] = None
    for index, text in enumerate(benchmark_cases(request)):
        total_start = time.perf_counter()
        pcm, _media_type, synthesis_ms, voice = synthesize(text, request.voice, "pcm", request.speed)
        total_latency_ms = (time.perf_counter() - total_start) * 1000
        if index == 0:
            first_latency = total_latency_ms
        duration = audio_duration_sec(pcm)
        cases.append(
            {
                "text": text,
                "voice": voice["label"],
                "chars": len(text),
                "audio_bytes": len(pcm),
                "audio_duration_sec": duration,
                "synthesis_ms": synthesis_ms,
                "total_latency_ms": total_latency_ms,
                "realtime_factor": (total_latency_ms / 1000) / duration if duration else None,
                "chars_per_sec": len(text) / (total_latency_ms / 1000) if total_latency_ms else 0.0,
            }
        )
    resources = sampler.stop()
    result = {
        "model": MODEL_ID,
        "device": DEVICE,
        "fake_engine": USE_FAKE_ENGINE,
        "sample_rate": SAMPLE_RATE,
        "default_voice": DEFAULT_VOICE,
        "cold_start_ms": cold_start_ms,
        "first_request_latency_ms": first_latency,
        "total_latency_ms": sum(case["total_latency_ms"] for case in cases),
        "cases": cases,
        "resources": resources,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        BENCHMARK_LATEST_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        result["benchmark_write_error"] = str(exc)
    last_benchmark = result
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "n_gpu_layers": N_GPU_LAYERS,
        "threads": THREADS,
        "n_ctx": N_CTX,
        "sample_rate": SAMPLE_RATE,
        "default_voice": DEFAULT_VOICE,
        "loaded": engine is not None,
        "voices": [voice["name"] for voice in VOICES],
    }


@app.get("/v1/voices")
def voices() -> dict[str, Any]:
    return {"object": "list", "data": VOICES}


@app.post("/v1/audio/speech")
def openai_speech(request: SpeechRequest, authorization: Optional[str] = Header(default=None)) -> Response:
    require_auth(authorization)
    content, media_type, _synthesis_ms, _voice = synthesize(
        request.input,
        request.voice,
        request.response_format,
        request.speed,
    )
    return Response(content=content, media_type=media_type)


@app.post("/tts")
def index_stream_tts(request: IndexStreamRequest, authorization: Optional[str] = Header(default=None)) -> Response:
    require_auth(authorization)
    content, media_type, _synthesis_ms, _voice = synthesize(
        request.text,
        request.character,
        request.format,
        request.speed,
    )
    return Response(content=content, media_type=media_type)


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    validate_format(request.format)
    return run_benchmark(request)


@app.get("/benchmark/last")
def benchmark_last() -> dict[str, Any]:
    if last_benchmark:
        return last_benchmark
    if BENCHMARK_LATEST_PATH.exists():
        return json.loads(BENCHMARK_LATEST_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Chưa có benchmark")
