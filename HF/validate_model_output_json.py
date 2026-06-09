import argparse
import json
from pathlib import Path


def parse_roles(raw_roles, mode_dir):
    if raw_roles:
        return [role.strip() for role in raw_roles.split(",") if role.strip()]
    return sorted(path.stem for path in mode_dir.glob("*.json"))


def resolve_path(root, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return root / path


def get_dialogue(model_data, dialogue_id):
    if "dialogue" in model_data and dialogue_id == 0:
        return model_data["dialogue"]
    return model_data.get(f"dialogue_{dialogue_id}")


def validate_role(root, mode, test_model, role, baseline_speech_key):
    errors = []
    gt_path = root / "test_data" / f"{mode}_turn" / f"{role}.json"
    model_path = root / "model_output" / test_model / f"{mode}_result" / "json" / f"{role}.json"

    if not gt_path.exists():
        return [f"{role}: missing ground-truth JSON: {gt_path}"]
    if not model_path.exists():
        return [f"{role}: missing model-output JSON: {model_path}"]

    with gt_path.open("r", encoding="utf-8") as handle:
        gt_data = json.load(handle)
    with model_path.open("r", encoding="utf-8") as handle:
        model_data = json.load(handle)

    for dialogue_id, gt_item in enumerate(gt_data):
        gt_turns = gt_item.get("dialogue")
        if not isinstance(gt_turns, list) or not gt_turns:
            errors.append(f"{role} dialogue_{dialogue_id}: ground-truth dialogue must be a non-empty list")
            continue

        model_turns = get_dialogue(model_data, dialogue_id)
        if not isinstance(model_turns, list):
            errors.append(f"{role} dialogue_{dialogue_id}: missing model dialogue")
            continue
        if len(model_turns) != len(gt_turns):
            errors.append(
                f"{role} dialogue_{dialogue_id}: turn count mismatch, gt={len(gt_turns)} model={len(model_turns)}"
            )

        for turn_id, gt_turn in enumerate(gt_turns):
            if baseline_speech_key not in gt_turn:
                errors.append(f"{role} dialogue_{dialogue_id} turn_{turn_id}: missing {baseline_speech_key}")
            else:
                baseline_audio = resolve_path(root, gt_turn[baseline_speech_key])
                if not baseline_audio.exists():
                    errors.append(f"{role} dialogue_{dialogue_id} turn_{turn_id}: missing baseline audio {baseline_audio}")

            if turn_id >= len(model_turns):
                continue

            model_turn = model_turns[turn_id]
            model_output = model_turn.get("model_output", {})
            if not model_output.get("audio_path"):
                errors.append(f"{role} dialogue_{dialogue_id} turn_{turn_id}: missing model_output.audio_path")
            else:
                model_audio = resolve_path(root, model_output["audio_path"])
                if not model_audio.exists():
                    errors.append(f"{role} dialogue_{dialogue_id} turn_{turn_id}: missing model audio {model_audio}")

            if "text" not in model_output:
                errors.append(f"{role} dialogue_{dialogue_id} turn_{turn_id}: missing model_output.text")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate SpeechRole model_output JSON files before running gemini_judge.py."
    )
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--test-model", required=True)
    parser.add_argument("--roles", default="", help="Comma-separated role names. Default: all roles in test_data.")
    parser.add_argument("--root", default=".", help="HF directory containing test_data/ and model_output/.")
    parser.add_argument(
        "--baseline-speech-key",
        default="role_speech_path1",
        help="Ground-truth audio field used as Model B in gemini_judge.py.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    root = Path(args.root)
    if not root.is_absolute():
        root = base_dir / root
    mode_dir = root / "test_data" / f"{args.mode}_turn"
    roles = parse_roles(args.roles, mode_dir)

    all_errors = []
    for role in roles:
        all_errors.extend(validate_role(root, args.mode, args.test_model, role, args.baseline_speech_key))

    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s):")
        for error in all_errors[:100]:
            print(f"- {error}")
        if len(all_errors) > 100:
            print(f"... {len(all_errors) - 100} more")
        raise SystemExit(1)

    print(f"Validation passed: {len(roles)} role file(s), mode={args.mode}, test_model={args.test_model}")


if __name__ == "__main__":
    main()
