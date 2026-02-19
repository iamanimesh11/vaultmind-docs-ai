"""
benchmark_models.py
--------------------
Benchmarks multiple Ollama models to help you pick the best one
for your hardware. Reports inference time, tokens/sec, and RAM usage.

Usage:
    python scripts/benchmark_models.py
"""

import json
import platform
import subprocess
import time

import psutil

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_TO_TEST = [
    "llama3.2:1b",
    "llama3.2:3b",
    "mistral:7b-instruct-q4_K_M",
    "phi3:mini",
]

TEST_PROMPT = "Explain what a computer is in one short sentence."
OUTPUT_FILE = "scripts/benchmark_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def run_ollama_model(model: str, prompt: str) -> tuple[str | None, str]:
    """Invoke `ollama run <model>` with the given prompt and return output."""
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, err = process.communicate(input=prompt, timeout=120)
        return output.strip(), err.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return None, "Timeout after 120s"
    except FileNotFoundError:
        return None, "Ollama not found. Is it installed and on PATH?"


def benchmark_single_model(model_name: str) -> dict:
    """Run benchmark for one model and return a result dict."""
    print(f"\n🔍 Benchmarking: {model_name}")

    result = {
        "model":             model_name,
        "status":            "OK",
        "ram_before_gb":     round(get_available_ram_gb(), 2),
        "ram_after_gb":      None,
        "inference_time_sec": None,
        "approx_load_time_sec": None,
        "tokens_per_sec":    None,
        "response_preview":  None,
    }

    start = time.time()
    output, err = run_ollama_model(model_name, TEST_PROMPT)
    elapsed = round(time.time() - start, 2)

    result["ram_after_gb"] = round(get_available_ram_gb(), 2)

    if output is None:
        result["status"] = f"FAILED: {err}"
        print(f"  ❌ {result['status']}")
        return result

    token_count = len(output.split())
    result.update({
        "inference_time_sec":    elapsed,
        "approx_load_time_sec":  round(elapsed * 0.4, 2),
        "tokens_per_sec":        round(token_count / elapsed, 2) if elapsed > 0 else None,
        "response_preview":      output[:120] + ("..." if len(output) > 120 else ""),
    })

    print(f"  ✅ Done in {elapsed}s | ~{result['tokens_per_sec']} tok/s")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🚀 OLLAMA MODEL BENCHMARK")
    print(f"🖥️  OS  : {platform.system()} {platform.release()}")
    print(f"💾 RAM : {psutil.virtual_memory().total / (1024 ** 3):.1f} GB total")
    print(f"📝 Prompt: '{TEST_PROMPT}'\n")

    results = [benchmark_single_model(m) for m in MODELS_TO_TEST]

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊  BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n  Model  : {r['model']}")
        print(f"  Status : {r['status']}")
        print(f"  RAM Δ  : {r['ram_before_gb']} → {r['ram_after_gb']} GB")
        print(f"  Time   : {r['inference_time_sec']}s (load est. {r['approx_load_time_sec']}s)")
        print(f"  Speed  : {r['tokens_per_sec']} tok/s")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n💾 Results saved → {OUTPUT_FILE}")
    print("✅ Benchmark complete!\n")


if __name__ == "__main__":
    main()
