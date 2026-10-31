# CLAUDE.md

Realtime streaming TTS server (Orpheus 3B on vLLM, SNAC codec, FastAPI websocket) plus an audio
tokenization pipeline, measured on a single NVIDIA L4.

## Rules

- **Never write tokens or secrets into any file.** HF auth comes from `~/.cache/huggingface`;
  GitHub auth comes from the existing git remote.
- **Commit and push after every working step**, with a clear message.
- **Big files (datasets, shards) go to `/content/drive/MyDrive/speech-infra`, never into git.**
  Only small results (<5 MB) go in `results/`.
- **When measuring, change one thing at a time** and save the config with every result.
- **Every result file records GPU, model, vllm version, and date.**
- **Explain each change in plain words before making it.** Every piece of the project should be understandable.
