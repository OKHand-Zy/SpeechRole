import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from common import (
    TurnResult,
    add_common_args,
    load_dialogues,
    output_paths,
    parse_role_filter,
    pcm16_to_wav,
    require_any_env,
    wav_to_pcm16,
    write_role_outputs,
)


def require_google_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Missing dependency: pip install -r HF/eval/requirements.txt") from exc
    return genai, types


class GeminiLiveClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        output_rate: int = 24000,
        input_rate: int = 16000,
        chunk_ms: int = 100,
        enable_output_transcription: bool = False,
    ):
        genai, types = require_google_genai()
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model
        self.output_rate = output_rate
        self.input_rate = input_rate
        self.chunk_bytes = max(1, int(input_rate * chunk_ms / 1000) * 2)
        self.enable_output_transcription = enable_output_transcription

    async def run_dialogue(self, dialogue, output_root: Path, model_name: str, max_turns: int | None, save_events: bool) -> list[TurnResult]:
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": dialogue.system_prompt,
        }
        if self.enable_output_transcription:
            config["output_audio_transcription"] = {}

        results: list[TurnResult] = []
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            for turn in dialogue.turns[:max_turns]:
                audio_path, event_path = output_paths(
                    output_root, model_name, dialogue.mode, dialogue.role, dialogue.dialogue_id, turn.turn
                )
                await self._send_audio_turn(session, turn.user_audio_path)
                audio_pcm, transcript, events = await self._receive_turn(session)
                pcm16_to_wav(audio_path, audio_pcm, self.output_rate)
                if save_events:
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    event_path.write_text(json.dumps(events, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
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

    async def _send_audio_turn(self, session, user_audio_path: Path) -> None:
        pcm = wav_to_pcm16(user_audio_path, target_rate=self.input_rate)
        mime_type = f"audio/pcm;rate={self.input_rate}"
        for offset in range(0, len(pcm), self.chunk_bytes):
            chunk = pcm[offset:offset + self.chunk_bytes]
            await session.send_realtime_input(
                audio=self.types.Blob(data=chunk, mime_type=mime_type)
            )
            await asyncio.sleep(0)
        await session.send_realtime_input(audio_stream_end=True)

    async def _receive_turn(self, session) -> tuple[bytes, str, list[dict]]:
        audio_chunks: list[bytes] = []
        transcript_chunks: list[str] = []
        events: list[dict] = []
        async for response in session.receive():
            events.append(response.model_dump() if hasattr(response, "model_dump") else {"response": str(response)})
            content = getattr(response, "server_content", None)
            if not content:
                continue
            output_transcription = getattr(content, "output_transcription", None)
            if output_transcription and getattr(output_transcription, "text", None):
                transcript_chunks.append(output_transcription.text)
            model_turn = getattr(content, "model_turn", None)
            if model_turn:
                for part in getattr(model_turn, "parts", []) or []:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        audio_chunks.append(inline_data.data)
            if getattr(content, "turn_complete", False):
                return b"".join(audio_chunks), "".join(transcript_chunks), events
        return b"".join(audio_chunks), "".join(transcript_chunks), events


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run SpeechRole eval JSON through Gemini Live.")
    add_common_args(parser)
    parser.add_argument("--model", default="gemini-3.1-flash-live-preview")
    parser.add_argument("--model-name", default="gemini_live")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--input-rate", type=int, default=16000)
    parser.add_argument("--enable-output-transcription", action="store_true")
    args = parser.parse_args()

    api_key = require_any_env(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    output_root = Path(args.output_root)
    client = GeminiLiveClient(
        api_key=api_key,
        model=args.model,
        input_rate=args.input_rate,
        chunk_ms=args.chunk_ms,
        enable_output_transcription=args.enable_output_transcription,
    )
    dialogues = load_dialogues(Path(args.eval_dir), args.mode, parse_role_filter(args.roles), args.max_dialogues)

    grouped: dict[str, dict[int, list[TurnResult]]] = defaultdict(dict)
    for index, dialogue in enumerate(dialogues, 1):
        print(f"[gemini] {index}/{len(dialogues)} role={dialogue.role} dialogue={dialogue.dialogue_id}")
        grouped[dialogue.role][dialogue.dialogue_id] = await client.run_dialogue(
            dialogue, output_root, args.model_name, args.max_turns, args.save_events
        )
        write_role_outputs(output_root, args.model_name, args.mode, dialogue.role, grouped[dialogue.role])


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
