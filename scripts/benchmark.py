#!/usr/bin/env python3
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_TEXTS = [
    "Xin chào, tôi là trợ lý giọng nói Xiaozhi.",
    "Hôm nay hệ thống sẽ kiểm tra tốc độ tổng hợp giọng nói tiếng Việt với một câu dài hơn để đo realtime factor chính xác hơn.",
    "Xiaozhi có thể nói tiếng Việt và English trong cùng một câu để kiểm tra code-switching.",
]


def post_json(url: str, payload: dict, api_key: str = "") -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Benchmark failed: HTTP {exc.code}: {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VieNeu TTS benchmark through the local HTTP API.")
    parser.add_argument("--url", default="http://127.0.0.1:8004/benchmark")
    parser.add_argument("--text", action="append", help="Benchmark text. Can be repeated.")
    parser.add_argument("--voice", default="Thục Đoan (Nữ - Miền Nam)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--output", default="benchmarks/latest.json")
    args = parser.parse_args()

    payload = {
        "texts": args.text or DEFAULT_TEXTS,
        "voice": args.voice,
        "speed": args.speed,
        "format": "pcm",
        "cold_start": args.cold_start,
    }
    result = post_json(args.url, payload, args.api_key)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
