import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json_or_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def iter_prediction_records(data):
    if isinstance(data, list):
        for record in data:
            yield record
        return

    if isinstance(data, dict) and "predictions" in data:
        yield from iter_prediction_records(data["predictions"])
        return

    if isinstance(data, dict) and "role" in data and any(key.startswith("dialogue_") for key in data):
        role = data["role"]
        for dialogue_key, turns in data.items():
            if not dialogue_key.startswith("dialogue_"):
                continue
            dialogue_id = int(dialogue_key.split("_", 1)[1])
            for turn_id, turn in enumerate(turns):
                record = dict(turn)
                record.setdefault("role", role)
                record.setdefault("dialogue_id", dialogue_id)
                record.setdefault("turn", turn_id)
                yield record
        return

    if isinstance(data, dict):
        for role, role_payload in data.items():
            if not isinstance(role_payload, dict):
                continue
            for dialogue_key, turns in role_payload.items():
                if not dialogue_key.startswith("dialogue_"):
                    continue
                dialogue_id = int(dialogue_key.split("_", 1)[1])
                for turn_id, turn in enumerate(turns):
                    record = dict(turn)
                    record.setdefault("role", role)
                    record.setdefault("dialogue_id", dialogue_id)
                    record.setdefault("turn", turn_id)
                    yield record
        return

    raise ValueError("Predictions must be a JSON list, a {'predictions': [...]} object, or a role-keyed object.")


def get_nested(record, *keys):
    value = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def normalize_prediction(record):
    role = record.get("role")
    dialogue_id = record.get("dialogue_id", record.get("idx", record.get("dialogue_index")))
    turn = record.get("turn", record.get("turn_id"))

    text = (
        record.get("text")
        or record.get("model_output_text")
        or get_nested(record, "model_output", "text")
        or record.get("response")
        or ""
    )
    audio_path = (
        record.get("audio_path")
        or record.get("model_output_audio_path")
        or get_nested(record, "model_output", "audio_path")
        or ""
    )

    if role is None or dialogue_id is None or turn is None:
        raise ValueError(f"Prediction record is missing role/dialogue_id/turn: {record}")
    if not audio_path:
        raise ValueError(f"Prediction record is missing audio_path: {record}")

    user_text = (
        record.get("user")
        or record.get("user_text")
        or get_nested(record, "user_input", "text")
        or ""
    )

    return {
        "role": str(role),
        "dialogue_id": int(dialogue_id),
        "turn": int(turn),
        "user_text": user_text,
        "model_text": text,
        "audio_path": audio_path,
    }


def load_test_data(test_data_dir, mode, roles):
    test_items = {}
    for role in roles:
        path = test_data_dir / f"{mode}_turn" / f"{role}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing ground-truth file for role {role}: {path}")
        with path.open("r", encoding="utf-8") as handle:
            test_items[role] = json.load(handle)
    return test_items


def build_output_payload(role, mode, role_records, test_items):
    payload = {"role": role}
    records_by_dialogue = defaultdict(dict)
    for record in role_records:
        records_by_dialogue[record["dialogue_id"]][record["turn"]] = record

    for dialogue_id, gt_item in enumerate(test_items[role]):
        output_turns = []
        gt_turns = gt_item.get("dialogue", [])
        for turn_id, gt_turn in enumerate(gt_turns):
            prediction = records_by_dialogue.get(dialogue_id, {}).get(turn_id)
            if prediction is None:
                raise ValueError(f"Missing prediction: role={role}, dialogue_id={dialogue_id}, turn={turn_id}")

            user_text = prediction["user_text"] or gt_turn.get("user", "")
            output_turns.append(
                {
                    "turn": turn_id,
                    "user_input": {"text": user_text},
                    "model_output": {
                        "text": prediction["model_text"],
                        "audio_path": prediction["audio_path"],
                    },
                }
            )
        payload[f"dialogue_{dialogue_id}"] = output_turns

    return payload


def write_model_outputs(predictions_path, test_data_dir, output_root, model_name, mode):
    data = read_json_or_jsonl(predictions_path)
    records_by_role = defaultdict(list)
    for raw_record in iter_prediction_records(data):
        record = normalize_prediction(raw_record)
        records_by_role[record["role"]].append(record)

    test_items = load_test_data(test_data_dir, mode, records_by_role.keys())
    output_dir = output_root / model_name / f"{mode}_result" / "json"
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for role in sorted(records_by_role):
        payload = build_output_payload(role, mode, records_by_role[role], test_items)
        output_path = output_dir / f"{role}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        written.append(output_path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Create SpeechRole judge-compatible model_output JSON files from inference predictions."
    )
    parser.add_argument("--predictions", required=True, help="Prediction JSON or JSONL file.")
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--test-data-dir", default="test_data")
    parser.add_argument("--output-root", default="model_output")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    test_data_dir = Path(args.test_data_dir)
    output_root = Path(args.output_root)
    if not test_data_dir.is_absolute():
        test_data_dir = base_dir / test_data_dir
    if not output_root.is_absolute():
        output_root = base_dir / output_root

    written = write_model_outputs(
        predictions_path=Path(args.predictions),
        test_data_dir=test_data_dir,
        output_root=output_root,
        model_name=args.model_name,
        mode=args.mode,
    )

    print(f"Wrote {len(written)} model-output JSON files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
