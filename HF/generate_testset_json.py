import argparse
import json
from pathlib import Path


def parse_roles(raw_roles):
    if not raw_roles:
        return None
    return {role.strip() for role in raw_roles.split(",") if role.strip()}


def load_role_files(test_data_dir, mode, roles):
    mode_dir = test_data_dir / f"{mode}_turn"
    if not mode_dir.exists():
        raise FileNotFoundError(f"Missing test data directory: {mode_dir}")

    role_files = sorted(mode_dir.glob("*.json"))
    if roles:
        role_files = [path for path in role_files if path.stem in roles]
        missing_roles = roles - {path.stem for path in role_files}
        if missing_roles:
            raise FileNotFoundError(f"Missing role json files: {sorted(missing_roles)}")
    return role_files


def build_model_inputs(role, mode, source_items):
    model_inputs = []
    for dialogue_id, item in enumerate(source_items):
        turns = []
        for turn_id, turn in enumerate(item.get("dialogue", [])):
            turns.append(
                {
                    "turn": turn_id,
                    "user_input": {
                        "text": turn.get("user", ""),
                        "audio_path": turn.get("user_speech_path", ""),
                    },
                }
            )

        model_inputs.append(
            {
                "role": role,
                "mode": mode,
                "dialogue_id": dialogue_id,
                "system_prompt": item.get("system_prompt", ""),
                "context": item.get("context", ""),
                "category": item.get("category", ""),
                "turns": turns,
            }
        )
    return model_inputs


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def generate_mode(test_data_dir, output_dir, mode, roles):
    written = []
    for role_file in load_role_files(test_data_dir, mode, roles):
        with role_file.open("r", encoding="utf-8") as handle:
            source_items = json.load(handle)

        role = role_file.stem
        model_inputs = build_model_inputs(role, mode, source_items)
        output_path = output_dir / f"{mode}_turn" / f"{role}.json"
        write_json(output_path, {"role": role, "mode": mode, "dialogues": model_inputs})
        written.append(output_path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Generate input-only SpeechRole test-set JSON files for model inference."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "all"],
        default="all",
        help="Which test split to export.",
    )
    parser.add_argument(
        "--roles",
        default="",
        help="Comma-separated role names. Default: export all available roles.",
    )
    parser.add_argument(
        "--test-data-dir",
        default="test_data",
        help="Directory containing single_turn/ and multi_turn/ ground-truth JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_test_sets",
        help="Directory where generated model input JSON files will be written.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    test_data_dir = Path(args.test_data_dir)
    output_dir = Path(args.output_dir)
    if not test_data_dir.is_absolute():
        test_data_dir = base_dir / test_data_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    roles = parse_roles(args.roles)
    modes = ["single", "multi"] if args.mode == "all" else [args.mode]

    written = []
    for mode in modes:
        written.extend(generate_mode(test_data_dir, output_dir, mode, roles))

    print(f"Wrote {len(written)} test-set JSON files to {output_dir}")
    for path in written[:10]:
        print(path)
    if len(written) > 10:
        print(f"... {len(written) - 10} more")


if __name__ == "__main__":
    main()
