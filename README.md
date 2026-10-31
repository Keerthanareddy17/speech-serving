# speech-serving

A realtime streaming text-to-speech server built on Orpheus 3B served with vLLM, decoding audio tokens
with the SNAC codec and streaming audio to clients over a FastAPI websocket, together with an audio
tokenization pipeline (download → decode → VAD → SNAC tokenize → shard). The goal is to measure and
document how far a single NVIDIA L4 can go on latency (time-to-first-audio), real-time factor, and
concurrent streams, with every result reproducible from its saved config.

**Results coming.**
