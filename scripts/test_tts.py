#!/usr/bin/env python3
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8004"
DEFAULT_TEXT = "Xin chào, đây là bài kiểm tra dịch vụ VieNeu TTS."
DEFAULT_VOICE = "Thục Đoan (Nữ - Miền Nam)"


def service_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def fetch_health(base_url: str, timeout: int) -> dict:
    url = service_url(base_url, "/health")
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Health check failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Health check failed: {exc.reason}") from exc


def synthesize(
    base_url: str,
    text: str,
    voice: str,
    output_format: str,
    api_key: str,
    timeout: int,
    speed: float = 1.0,
) -> tuple[bytes, str]:
    payload = {
        "model": "vieneu-v2-turbo",
        "input": text,
        "voice": voice,
        "response_format": output_format,
        "speed": speed,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        service_url(base_url, "/v1/audio/speech"),
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            return response.read(), content_type
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"TTS request failed: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"TTS request failed: {exc.reason}") from exc


def validate_audio(audio: bytes, output_format: str) -> None:
    if not audio:
        raise SystemExit("TTS response was empty")
    if output_format == "wav" and not audio.startswith(b"RIFF"):
        raise SystemExit("TTS response is not a valid WAV file")
    if output_format == "pcm" and len(audio) % 2 != 0:
        raise SystemExit("TTS response is not valid 16-bit PCM: odd byte count")


def write_audio(path: Path, audio: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test VieNeu TTS service and write audio output.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL.")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name or alias.")
    parser.add_argument("--format", choices=["wav", "pcm"], default="wav", help="Audio response format.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--api-key", default="", help="Bearer token for services protected by TTS_API_KEY.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--output", default="", help="Output audio path.")
    args = parser.parse_args()

    output = Path(args.output or f"tmp/tts-test.{args.format}")
    health = fetch_health(args.base_url, args.timeout)
    audio, content_type = synthesize(
        base_url=args.base_url,
        text=args.text,
        voice=args.voice,
        output_format=args.format,
        api_key=args.api_key,
        timeout=args.timeout,
        speed=args.speed,
    )
    validate_audio(audio, args.format)
    write_audio(output, audio)

    print(f"Service: {args.base_url}")
    print(f"Health: {health.get('status', 'unknown')}")
    print(f"Voice: {args.voice}")
    print(f"Format: {args.format}")
    print(f"Content-Type: {content_type or 'unknown'}")
    print(f"Bytes: {len(audio)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
