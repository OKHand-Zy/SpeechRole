import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from common import (
    TurnResult,
    add_common_args,
    b64,
    load_dialogues,
    output_paths,
    parse_role_filter,
    pcm16_to_wav,
    require_env,
    wav_to_pcm16,
    write_role_outputs,
)


def require_websocket_client():
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install -r HF/eval/requirements.txt") from exc
    return websocket


class OpenAIRealtimeClient:
    def __init__(self, api_key: str, model: str, voice: str, output_rate: int = 24000, timeout: int = 180):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.output_rate = output_rate
        self.timeout = timeout
        self.websocket = require_websocket_client()

    def _connect(self):
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [f"Authorization: Bearer {self.api_key}"]
        ws = self.websocket.create_connection(url, header=headers, timeout=self.timeout)
        ws.settimeout(self.timeout)
        return ws

    def run_dialogue(self, dialogue, output_root: Path, model_name: str, max_turns: int | None, save_events: bool) -> list[TurnResult]:
        ws = self._connect()
        try:
            self._wait_for(ws, {"session.created"})
            self._send(
                ws,
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": dialogue.system_prompt,
                        "output_modalities": ["audio"],
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": 24000},
                                "turn_detection": None,
                            },
                            "output": {
                                "format": {"type": "audio/pcm", "rate": self.output_rate},
                                "voice": self.voice,
                            },
                        },
                    },
                },
            )
            self._wait_for(ws, {"session.updated"})

            results: list[TurnResult] = []
            for turn in dialogue.turns[:max_turns]:
                audio_path, event_path = output_paths(
                    output_root, model_name, dialogue.mode, dialogue.role, dialogue.dialogue_id, turn.turn
                )
                pcm = wav_to_pcm16(turn.user_audio_path, target_rate=24000)
                self._send(
                    ws,
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "audio": b64(pcm),
                                    "transcript": turn.user_text,
                                }
                            ],
                        },
                    },
                )
                self._send(ws, {"type": "response.create", "response": {"output_modalities": ["audio"]}})
                audio_pcm, transcript, events = self._receive_response(ws)
                pcm16_to_wav(audio_path, audio_pcm, self.output_rate)
                if save_events:
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                results.append(
                    TurnResult(
                        turn=turn.turn,
                        user_text=turn.user_text,
                        model_text=transcript,
                        audio_path=audio_path,
                        raw_events_path=event_path if save_events else None,
                    )
                )
            return results
        finally:
            ws.close()

    def _send(self, ws, event: dict) -> None:
        ws.send(json.dumps(event))

    def _wait_for(self, ws, event_types: set[str]) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            event = json.loads(ws.recv())
            if event.get("type") == "error":
                raise RuntimeError(json.dumps(event, ensure_ascii=False))
            if event.get("type") in event_types:
                return event
        raise TimeoutError(f"Timed out waiting for {sorted(event_types)}")

    def _receive_response(self, ws) -> tuple[bytes, str, list[dict]]:
        audio_chunks: list[bytes] = []
        transcript_chunks: list[str] = []
        events: list[dict] = []
        while True:
            event = json.loads(ws.recv())
            events.append(event)
            event_type = event.get("type")
            if event_type == "error":
                raise RuntimeError(json.dumps(event, ensure_ascii=False))
            if event_type == "response.output_audio.delta":
                audio_chunks.append(self._decode_b64(event.get("delta", "")))
            elif event_type == "response.output_audio_transcript.delta":
                transcript_chunks.append(event.get("delta", ""))
            elif event_type == "response.output_audio_transcript.done":
                transcript_chunks = [event.get("transcript", "".join(transcript_chunks))]
            elif event_type == "response.output_item.done":
                transcript = self._extract_transcript(event.get("item", {}))
                if transcript:
                    transcript_chunks = [transcript]
            elif event_type == "response.done":
                response = event.get("response", {})
                if response.get("status") not in (None, "completed"):
                    raise RuntimeError(json.dumps(event, ensure_ascii=False))
                return b"".join(audio_chunks), "".join(transcript_chunks), events

    @staticmethod
    def _decode_b64(value: str) -> bytes:
        import base64

        return base64.b64decode(value) if value else b""

    @staticmethod
    def _extract_transcript(item: dict) -> str:
        for content in item.get("content", []):
            if content.get("transcript"):
                return content["transcript"]
            if content.get("text"):
                return content["text"]
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpeechRole eval JSON through OpenAI Realtime.")
    add_common_args(parser)
    parser.add_argument("--model", default="gpt-realtime-2")
    parser.add_argument("--model-name", default="openai_realtime")
    parser.add_argument("--voice", default="marin")
    args = parser.parse_args()

    api_key = require_env("OPENAI_API_KEY")
    output_root = Path(args.output_root)
    client = OpenAIRealtimeClient(api_key=api_key, model=args.model, voice=args.voice)
    dialogues = load_dialogues(Path(args.eval_dir), args.mode, parse_role_filter(args.roles), args.max_dialogues)

    grouped: dict[str, dict[int, list[TurnResult]]] = defaultdict(dict)
    for index, dialogue in enumerate(dialogues, 1):
        print(f"[openai] {index}/{len(dialogues)} role={dialogue.role} dialogue={dialogue.dialogue_id}")
        grouped[dialogue.role][dialogue.dialogue_id] = client.run_dialogue(
            dialogue, output_root, args.model_name, args.max_turns, args.save_events
        )
        write_role_outputs(output_root, args.model_name, args.mode, dialogue.role, grouped[dialogue.role])


if __name__ == "__main__":
    main()
