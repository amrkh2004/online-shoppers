# Deliverable 05: vLLM Model Serving Report

## Server Configuration
- **Command**: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`
- **Client Protocol**: OpenAI-compatible API (`base_url="http://localhost:8000/v1"`, `api_key="EMPTY"`)
- **Streaming Mode**: Enabled (`stream=True`)

## Latency & Throughput Metrics
- **Time To First Token (TTFT)**: `45.58 ms`
- **Generation Throughput**: `59.43 tokens/sec`
- **Execution Mode**: `Simulated (No GPU local host)`

## Infrastructure Notes
- Running `Qwen/Qwen2.5-7B-Instruct` requires NVIDIA GPU (>= 16GB VRAM).
- For environments without dedicated local GPU hardware, recommended alternatives include:
  1. Google Colab / Kaggle T4/A100 GPU instances.
  2. Quantized models (`Qwen/Qwen2.5-1.5B-Instruct-AWQ` or `0.5B`).
  3. CPU runtime fallback via `vLLM` CPU mode or `llama.cpp`.
