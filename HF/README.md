---
license: mit
---

## Local workflow

### 1. Generate model-facing test-set JSON

The benchmark ground-truth files in `test_data/{single,multi}_turn/` contain reference answers and reference audio.
Use this script to export input-only JSON files for model inference:

```bash
python3 generate_testset_json.py --mode all
```

For a small subset:

```bash
python3 generate_testset_json.py --mode multi --roles "Thor,Coriolanus"
```

The generated files are written to `generated_test_sets/{single,multi}_turn/*.json`.

### 2. Convert model predictions to judge-compatible JSON

After running your model on the generated test set, save predictions as JSONL records:

```json
{"role":"Thor","dialogue_id":0,"turn":0,"text":"model response text","audio_path":"model_output/my_model/multi_result/audio/Thor/Thor_0_0.wav"}
{"role":"Thor","dialogue_id":0,"turn":1,"text":"model response text","audio_path":"model_output/my_model/multi_result/audio/Thor/Thor_0_1.wav"}
```

Then build the JSON files expected by `gemini_judge.py`:

```bash
python3 create_model_output_json.py \
  --predictions predictions.jsonl \
  --mode multi \
  --model-name my_model
```

This writes `model_output/my_model/{single,multi}_result/json/{role}.json`.

### 3. Validate before judging

Run a local schema/path check before making API calls:

```bash
python3 validate_model_output_json.py \
  --mode multi \
  --test-model my_model \
  --roles "Thor"
```

`--baseline-speech-key` defaults to `role_speech_path1`. Use `role_speech_path2` or `role_speech_path3` to compare against another reference TTS set.

### 4. Run the judge

```bash
python3 gemini_judge.py \
  --mode multi \
  --test_model my_model \
  --baseline_speech_key role_speech_path1
```
