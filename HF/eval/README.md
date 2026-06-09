# SpeechRole Realtime Evaluation

This folder contains generated SpeechRole test JSON files plus runners for realtime speech models.

## Environment

Create `.env` at the repository root:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

`GOOGLE_API_KEY` is also accepted for Gemini.

Install dependencies:

```bash
python3 -m pip install -r HF/eval/requirements.txt
```

## Run a small smoke test

OpenAI Realtime:

```bash
python3 HF/eval/openai_realtime_runner.py \
  --mode multi \
  --roles Thor \
  --max-dialogues 1 \
  --model gpt-realtime-2 \
  --model-name openai_realtime
```

Gemini Live:

```bash
python3 HF/eval/gemini_live_runner.py \
  --mode multi \
  --roles Thor \
  --max-dialogues 1 \
  --model gemini-3.1-flash-live-preview \
  --model-name gemini_live
```

Gemini Live audio is streamed in 100ms chunks by default and the runner sends `audio_stream_end=True` after each user WAV. If the API closes the connection with policy code `1008`, try a smaller chunk size:

```bash
python3 HF/eval/gemini_live_runner.py \
  --mode multi \
  --roles Thor \
  --max-dialogues 1 \
  --chunk-ms 50
```

To store Gemini output transcriptions in `model_output.*.text`, enable:

```bash
--enable-output-transcription
```

Outputs are written in the judge-compatible layout:

```text
HF/model_output/{model_name}/{single,multi}_result/json/{role}.json
HF/model_output/{model_name}/{single,multi}_result/audio/{role}/*.wav
```

Then validate locally:

```bash
python3 HF/validate_model_output_json.py --mode multi --test-model openai_realtime --roles Thor
python3 HF/validate_model_output_json.py --mode multi --test-model gemini_live --roles Thor
```

Run the Gemini judge from the repository root:

```bash
python3 HF/gemini_judge.py \
  --mode multi \
  --test_model gemini_live \
  --baseline_speech_key role_speech_path1 \
  --roles Thor
```

## Notes

The generated eval JSON points at 24kHz WAV user audio. Gemini Live expects raw 16kHz PCM input, while OpenAI Realtime defaults to raw 24kHz PCM input. The runners convert WAV to the required PCM format automatically and save model output as 24kHz WAV.
