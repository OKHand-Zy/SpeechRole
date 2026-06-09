import argparse
import base64
import json
import os
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
HF_DIR = EVAL_DIR.parent
REPO_ROOT = HF_DIR.parent


@dataclass
class EvalTurn:
    turn: int
    user_text: str
    user_audio_path: Path


@dataclass
class EvalDialogue:
    role: str
    mode: str
    dialogue_id: int
    system_prompt: str
    turns: list[EvalTurn]


@dataclass
class TurnResult:
    turn: int
    user_text: str
    model_text: str
    audio_path: Path
    raw_events_path: Path | None = None


def load_env() -> None:
    for env_path in (REPO_ROOT / ".env", HF_DIR / ".env", EVAL_DIR / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Put it in {REPO_ROOT / '.env'} or export it.")
    return value


def require_any_env(names: tuple[str, ...]) -> str:
    load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    joined = " or ".join(names)
    raise RuntimeError(f"Missing {joined}. Put it in {REPO_ROOT / '.env'} or export it.")


def parse_role_filter(raw_roles: str) -> set[str] | None:
    if not raw_roles:
        return None
    return {role.strip() for role in raw_roles.split(",") if role.strip()}


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    hf_relative = HF_DIR / path
    if hf_relative.exists():
        return hf_relative
    return REPO_ROOT / path


def load_dialogues(eval_dir: Path, mode: str, roles: set[str] | None, max_dialogues: int | None) -> list[EvalDialogue]:
    mode_dir = eval_dir / f"{mode}_turn"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing eval split: {mode_dir}")

    role_files = sorted(mode_dir.glob("*.json"))
    if roles is not None:
        role_files = [path for path in role_files if path.stem in roles]
        missing = roles - {path.stem for path in role_files}
        if missing:
            raise FileNotFoundError(f"Missing role files: {sorted(missing)}")

    dialogues: list[EvalDialogue] = []
    for role_file in role_files:
        payload = json.loads(role_file.read_text(encoding="utf-8"))
        for raw_dialogue in payload.get("dialogues", []):
            turns = [
                EvalTurn(
                    turn=int(turn["turn"]),
                    user_text=turn.get("user_input", {}).get("text", ""),
                    user_audio_path=resolve_repo_path(turn.get("user_input", {}).get("audio_path", "")),
                )
                for turn in raw_dialogue.get("turns", [])
            ]
            dialogues.append(
                EvalDialogue(
                    role=raw_dialogue.get("role", payload.get("role", role_file.stem)),
                    mode=raw_dialogue.get("mode", mode),
                    dialogue_id=int(raw_dialogue.get("dialogue_id", len(dialogues))),
                    system_prompt=raw_dialogue.get("system_prompt", ""),
                    turns=turns,
                )
            )
            if max_dialogues is not None and len(dialogues) >= max_dialogues:
                return dialogues
    return dialogues


def samples_from_wav(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported: {path}")

    samples = array("h")
    samples.frombytes(frames)
    if channels == 1:
        return samples.tolist(), sample_rate

    mono = []
    for idx in range(0, len(samples), channels):
        mono.append(int(sum(samples[idx:idx + channels]) / channels))
    return mono, sample_rate


def resample_linear(samples: list[int], source_rate: int, target_rate: int) -> list[int]:
    if source_rate == target_rate or not samples:
        return samples
    target_length = max(1, int(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    output = []
    last_index = len(samples) - 1
    for idx in range(target_length):
        src_pos = idx * ratio
        left = int(src_pos)
        right = min(left + 1, last_index)
        frac = src_pos - left
        value = int(samples[left] * (1.0 - frac) + samples[right] * frac)
        output.append(max(-32768, min(32767, value)))
    return output


def wav_to_pcm16(path: Path, target_rate: int) -> bytes:
    samples, source_rate = samples_from_wav(path)
    samples = resample_linear(samples, source_rate, target_rate)
    output = array("h", samples)
    return output.tobytes()


def pcm16_to_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def output_paths(output_root: Path, model_name: str, mode: str, role: str, dialogue_id: int, turn: int) -> tuple[Path, Path]:
    audio_path = output_root / model_name / f"{mode}_result" / "audio" / role / f"{role}_{dialogue_id}_{turn}.wav"
    event_path = output_root / model_name / f"{mode}_result" / "events" / role / f"{role}_{dialogue_id}_{turn}.json"
    return audio_path, event_path


def write_role_outputs(output_root: Path, model_name: str, mode: str, role: str, results_by_dialogue: dict[int, list[TurnResult]]) -> Path:
    payload = {"role": role}
    for dialogue_id in sorted(results_by_dialogue):
        payload[f"dialogue_{dialogue_id}"] = [
            {
                "turn": result.turn,
                "user_input": {"text": result.user_text},
                "model_output": {
                    "text": result.model_text,
                    "audio_path": str(result.audio_path.relative_to(HF_DIR)),
                },
            }
            for result in results_by_dialogue[dialogue_id]
        ]

    output_path = output_root / model_name / f"{mode}_result" / "json" / f"{role}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--roles", default="", help="Comma-separated role names. Default: all roles.")
    parser.add_argument("--eval-dir", default=str(EVAL_DIR), help="Directory containing single_turn/ and multi_turn/.")
    parser.add_argument("--output-root", default=str(HF_DIR / "model_output"))
    parser.add_argument("--max-dialogues", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--save-events", action="store_true")
