"""
vLLM Inference Client & TTFT Benchmark Script (Deliverable 05)
Connects to vLLM server (Qwen/Qwen2.5-7B-Instruct), enables streaming,
and measures Time To First Token (TTFT) and throughput (tokens/sec).
"""

import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

BASE_DIR = Path(__file__).resolve().parent.parent


def test_vllm_streaming(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt: str = "Explain the difference between batch inference and event-driven streaming inference in 3 bullet points.",
):
    print("=== vLLM Client Streaming Benchmark (Deliverable 05) ===")
    print(f"Connecting to: {base_url} (Model: {model_name})")

    if OpenAI is None:
        print(
            "[vLLM Client WARN] openai package not installed. Operating in mock measurement mode..."
        )
        # Simulate TTFT and throughput measurement
        start_time = time.perf_counter()
        time.sleep(0.045)  # Simulated 45ms TTFT
        ttft_ms = (time.perf_counter() - start_time) * 1000.0

        mock_output = "1. Batch processes data in bulk periodically.\n2. Event-driven processes events real-time.\n3. Streaming minimizes latency."
        tokens_generated = len(mock_output.split()) * 1.3
        total_time = 0.35
        throughput = tokens_generated / total_time

        print(f"Response:\n{mock_output}\n")
        print("=== Performance Metrics (Simulated Benchmark) ===")
        print(f"Time To First Token (TTFT): {ttft_ms:.2f} ms")
        print(f"Generation Throughput    : {throughput:.2f} tokens/sec")
        write_vllm_report(ttft_ms, throughput, model_name, is_mock=True)
        return

    client = OpenAI(base_url=base_url, api_key=api_key)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    generated_text = ""

    try:
        response_stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7,
            max_tokens=256,
        )

        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_text += text_chunk
                token_count += 1
                sys.stdout.write(text_chunk)
                sys.stdout.flush()

        end_time = time.perf_counter()
        ttft_ms = ((first_token_time or end_time) - start_time) * 1000.0
        total_gen_time = end_time - (first_token_time or start_time)
        throughput = token_count / total_gen_time if total_gen_time > 0 else 0.0

        print("\n\n=== Performance Metrics ===")
        print(f"Time To First Token (TTFT): {ttft_ms:.2f} ms")
        print(f"Generation Throughput    : {throughput:.2f} tokens/sec")
        write_vllm_report(ttft_ms, throughput, model_name, is_mock=False)

    except Exception as e:
        print(f"\n[vLLM Server Connection Error]: {e}")
        print("Ensure vLLM server is running: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`")


def write_vllm_report(ttft_ms: float, throughput: float, model_name: str, is_mock: bool = False):
    report_path = BASE_DIR / "reports" / "vllm_performance_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        f"""# Deliverable 05: vLLM Model Serving Report

## Server Configuration
- **Command**: `vllm serve {model_name} --port 8000`
- **Client Protocol**: OpenAI-compatible API (`base_url="http://localhost:8000/v1"`, `api_key="EMPTY"`)
- **Streaming Mode**: Enabled (`stream=True`)

## Latency & Throughput Metrics
- **Time To First Token (TTFT)**: `{ttft_ms:.2f} ms`
- **Generation Throughput**: `{throughput:.2f} tokens/sec`
- **Execution Mode**: `{"Simulated (No GPU local host)" if is_mock else "Live vLLM GPU Server"}`

## Infrastructure Notes
- Running `Qwen/Qwen2.5-7B-Instruct` requires NVIDIA GPU (>= 16GB VRAM).
- For environments without dedicated local GPU hardware, recommended alternatives include:
  1. Google Colab / Kaggle T4/A100 GPU instances.
  2. Quantized models (`Qwen/Qwen2.5-1.5B-Instruct-AWQ` or `0.5B`).
  3. CPU runtime fallback via `vLLM` CPU mode or `llama.cpp`.
""",
        encoding="utf-8",
    )
    print(f"vLLM Report saved to: {report_path}")


if __name__ == "__main__":
    test_vllm_streaming()
