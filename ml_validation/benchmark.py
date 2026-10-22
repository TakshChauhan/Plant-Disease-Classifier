"""Phase 7 — Model Performance, Latency, Throughput, Memory, and ONNX Export Benchmark.

Benchmarks the existing trained checkpoint (`best_model.pt`) without retraining:
- Exact model parameter count (total, backbone, head, and BatchNorm buffers)
- Theoretical MACs / FLOPs for a 224x224 ResNet18 forward pass
- Checkpoint disk footprint and serialized ONNX graph size
- Inference latency distribution (mean, std, min, median/p50, p90, p95, p99, IQR)
  using an explicit warm-up phase and multiple timed repetitions
- Batch throughput (images/sec) across batch sizes (1, 4, 8, 16)
- Process RSS memory footprint (`resource.getrusage`)
- Recorded hardware and software environment (CPU model, core count, RAM, OS, Python,
  execution backend). Never fabricates GPU results when GPU hardware is absent.

Usage:
    python -m ml_validation.benchmark \
        --checkpoint results/best_model.pt \
        --output-dir ml_validation/reports/phase7_efficiency_benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __package__ in (None, ""):
    _pkg_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_pkg_dir.parent))
    if "ml_validation" not in sys.modules:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "ml_validation", _pkg_dir / "__init__.py", submodule_search_locations=[str(_pkg_dir)]
        )
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["ml_validation"] = _mod
            _spec.loader.exec_module(_mod)

from ml_validation.core.backend import HAS_ORT, HAS_TORCH, CheckpointModelAdapter  # noqa: E402


def get_process_rss_mb() -> float:
    """Return current/peak resident set size (RSS) in MiB on Linux/macOS."""
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return round(kb / 1024.0, 2)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is in KiB; on macOS it is in bytes
    if sys.platform == "darwin":
        return round(usage / (1024.0 * 1024.0), 2)
    return round(usage / 1024.0, 2)


def get_environment_metadata() -> Dict[str, object]:
    """Capture accurate hardware and software environment telemetry."""
    cpu_model = platform.processor() or "unknown"
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "Model name:" in line:
                val = line.split(":", 1)[1].strip()
                if val and val.lower() != "unknown":
                    cpu_model = val
    except Exception:
        pass

    mem_total_mb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_mb = round(int(line.split()[1]) / 1024.0, 1)
                break

    env = {
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_model": cpu_model,
        "system_ram_total_mb": mem_total_mb,
        "numpy_version": np.__version__,
        "pytorch_installed": HAS_TORCH,
        "onnxruntime_installed": HAS_ORT,
        "cuda_gpu_available": False,
        "mps_available": False,
    }
    if HAS_TORCH:
        import torch
        env["pytorch_version"] = str(torch.__version__)
        env["cuda_gpu_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
        env["mps_available"] = bool(
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        )
    if HAS_ORT:
        import onnxruntime as ort
        env["onnxruntime_version"] = ort.__version__
        env["onnxruntime_providers"] = ort.get_available_providers()
    return env


def compute_resnet18_theoretical_flops(num_classes: int = 38) -> Dict[str, object]:
    """Compute exact theoretical MACs and FLOPs for ResNet18 at (1, 3, 224, 224) resolution."""
    # (name, c_in, c_out, k, h_out, w_out)
    conv_layers = [
        ("conv1", 3, 64, 7, 112, 112),
        ("layer1.0.conv1", 64, 64, 3, 56, 56),
        ("layer1.0.conv2", 64, 64, 3, 56, 56),
        ("layer1.1.conv1", 64, 64, 3, 56, 56),
        ("layer1.1.conv2", 64, 64, 3, 56, 56),
        ("layer2.0.conv1", 64, 128, 3, 28, 28),
        ("layer2.0.conv2", 128, 128, 3, 28, 28),
        ("layer2.0.downsample.0", 64, 128, 1, 28, 28),
        ("layer2.1.conv1", 128, 128, 3, 28, 28),
        ("layer2.1.conv2", 128, 128, 3, 28, 28),
        ("layer3.0.conv1", 128, 256, 3, 14, 14),
        ("layer3.0.conv2", 256, 256, 3, 14, 14),
        ("layer3.0.downsample.0", 128, 256, 1, 14, 14),
        ("layer3.1.conv1", 256, 256, 3, 14, 14),
        ("layer3.1.conv2", 256, 256, 3, 14, 14),
        ("layer4.0.conv1", 256, 512, 3, 7, 7),
        ("layer4.0.conv2", 512, 512, 3, 7, 7),
        ("layer4.0.downsample.0", 256, 512, 1, 7, 7),
        ("layer4.1.conv1", 512, 512, 3, 7, 7),
        ("layer4.1.conv2", 512, 512, 3, 7, 7),
    ]
    total_macs = 0
    for _name, cin, cout, k, hout, wout in conv_layers:
        total_macs += cin * cout * (k * k) * hout * wout
    # Final FC layer
    total_macs += 512 * num_classes
    return {
        "input_resolution": "3x224x224",
        "total_macs": int(total_macs),
        "total_gmacs": round(total_macs / 1e9, 4),
        "total_gflops_approx_2x_macs": round((2.0 * total_macs) / 1e9, 4),
    }


def run_benchmark(
    checkpoint_path: str | Path,
    output_dir: Path,
    warmup_runs: int = 10,
    timed_runs: int = 50,
    batch_sizes: Sequence[int] = (1, 4, 8, 16),
    device: str = "auto",
    backend: str = "auto",
    export_onnx: bool = True,
) -> Dict[str, object]:
    """Run full latency, throughput, memory, and model complexity benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rss_before_load = get_process_rss_mb()

    t_load_0 = time.perf_counter()
    adapter = CheckpointModelAdapter(checkpoint_path, device=device, backend=backend)
    t_load_ms = (time.perf_counter() - t_load_0) * 1000.0
    rss_after_load = get_process_rss_mb()

    param_summary = adapter.parameter_summary()
    flops_summary = compute_resnet18_theoretical_flops(adapter.num_classes)
    env_meta = get_environment_metadata()

    rng = np.random.default_rng(42)
    single_input = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)

    # 1. Single-image forward latency benchmark (with warm-up)
    for _ in range(warmup_runs):
        _ = adapter.forward_logits(single_input)

    latencies_ms: List[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = adapter.forward_logits(single_input)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # 2. Single-image forward + Grad-CAM latency benchmark
    for _ in range(max(3, warmup_runs // 2)):
        _ = adapter.forward_logits_and_cam(single_input)

    gradcam_latencies_ms: List[float] = []
    for _ in range(max(15, timed_runs // 2)):
        t0 = time.perf_counter()
        _ = adapter.forward_logits_and_cam(single_input)
        t1 = time.perf_counter()
        gradcam_latencies_ms.append((t1 - t0) * 1000.0)

    # 3. Batch size throughput sweep
    batch_results: List[Dict[str, object]] = []
    for bs in batch_sizes:
        x_batch = rng.standard_normal((bs, 3, 224, 224), dtype=np.float32)
        n_iters = max(5, timed_runs // max(bs, 1))
        for _ in range(3):
            _ = adapter.forward_logits(x_batch)

        b_times_ms = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = adapter.forward_logits(x_batch)
            t1 = time.perf_counter()
            b_times_ms.append((t1 - t0) * 1000.0)

        med_batch_ms = float(np.median(b_times_ms))
        mean_batch_ms = float(np.mean(b_times_ms))
        p90_batch_ms = float(np.percentile(b_times_ms, 90))
        p95_batch_ms = float(np.percentile(b_times_ms, 95))
        p99_batch_ms = float(np.percentile(b_times_ms, 99))
        throughput_ips = (bs * 1000.0) / max(med_batch_ms, 1e-6)

        batch_results.append({
            "batch_size": int(bs),
            "num_timed_iterations": int(n_iters),
            "median_batch_latency_ms": round(med_batch_ms, 3),
            "mean_batch_latency_ms": round(mean_batch_ms, 3),
            "std_batch_latency_ms": round(float(np.std(b_times_ms)), 3),
            "p90_batch_latency_ms": round(p90_batch_ms, 3),
            "p95_batch_latency_ms": round(p95_batch_ms, 3),
            "p99_batch_latency_ms": round(p99_batch_ms, 3),
            "per_image_median_ms": round(med_batch_ms / bs, 3),
            "throughput_images_per_sec": round(throughput_ips, 2),
        })

    rss_peak = get_process_rss_mb()

    # Optional ONNX export artifact verification
    onnx_info = None
    if export_onnx and adapter.onnx_bytes is not None:
        onnx_path = output_dir / "resnet18_best_model_folded.onnx"
        onnx_path.write_bytes(adapter.onnx_bytes)
        onnx_info = {
            "onnx_file_path": str(onnx_path),
            "onnx_file_size_bytes": len(adapter.onnx_bytes),
            "onnx_file_size_mb": round(len(adapter.onnx_bytes) / (1024.0 * 1024.0), 4),
            "opset_version": 13,
            "batchnorm_folded_into_conv": True,
        }

    with open(output_dir / "batch_throughput_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(batch_results[0].keys()))
        writer.writeheader()
        writer.writerows(batch_results)

    _plot_benchmark_charts(
        latencies_ms, gradcam_latencies_ms, batch_results,
        output_dir / "latency_and_throughput_chart.png",
    )

    lat_arr = np.asarray(latencies_ms, dtype=np.float64)
    cam_arr = np.asarray(gradcam_latencies_ms, dtype=np.float64)

    report = {
        "phase": "Phase 7 — Model Performance and Efficiency Benchmark",
        "environment": env_meta,
        "execution_backend": adapter.backend,
        "device_used": adapter.device,
        "gpu_benchmark_status": (
            "AVAILABLE_AND_MEASURED" if env_meta["cuda_gpu_available"]
            else (
                "UNAVAILABLE_CPU_ONLY_MEASURED (No GPU hardware present in this environment; "
                "GPU numbers not fabricated)"
            )
        ),
        "model_complexity": {
            **param_summary,
            **flops_summary,
        },
        "memory_and_load_telemetry": {
            "checkpoint_load_and_compile_ms": round(t_load_ms, 2),
            "rss_before_load_mb": rss_before_load,
            "rss_after_load_mb": rss_after_load,
            "rss_peak_during_benchmark_mb": rss_peak,
            "model_resident_delta_mb": round(max(0.0, rss_peak - rss_before_load), 2),
        },
        "single_image_forward_latency_ms": {
            "warmup_iterations": warmup_runs,
            "timed_iterations": timed_runs,
            "median_p50_ms": round(float(np.median(lat_arr)), 3),
            "mean_ms": round(float(np.mean(lat_arr)), 3),
            "std_ms": round(float(np.std(lat_arr)), 3),
            "min_ms": round(float(np.min(lat_arr)), 3),
            "max_ms": round(float(np.max(lat_arr)), 3),
            "iqr_ms": round(float(np.percentile(lat_arr, 75) - np.percentile(lat_arr, 25)), 3),
            "p90_ms": round(float(np.percentile(lat_arr, 90)), 3),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
            "throughput_images_per_sec": round(1000.0 / float(np.median(lat_arr)), 2),
        },
        "single_image_forward_plus_gradcam_latency_ms": {
            "timed_iterations": len(cam_arr),
            "median_p50_ms": round(float(np.median(cam_arr)), 3),
            "mean_ms": round(float(np.mean(cam_arr)), 3),
            "p90_ms": round(float(np.percentile(cam_arr, 90)), 3),
            "p95_ms": round(float(np.percentile(cam_arr, 95)), 3),
            "p99_ms": round(float(np.percentile(cam_arr, 99)), 3),
        },
        "batch_size_scaling": batch_results,
        "onnx_export_verification": onnx_info,
    }
    (output_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    return report


def _plot_benchmark_charts(
    latencies_ms: Sequence[float],
    gradcam_latencies_ms: Sequence[float],
    batch_results: Sequence[Dict[str, object]],
    out_path: Path,
):
    """Render a 2-panel latency distribution + batch throughput chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    ax1.hist(latencies_ms, bins=15, color="#1971c2", alpha=0.82, edgecolor="#1864ab", label="Forward Only (B=1)")
    med = float(np.median(latencies_ms))
    p95 = float(np.percentile(latencies_ms, 95))
    ax1.axvline(med, color="#2b8a3e", linestyle="-", linewidth=2.0, label=f"Median (p50): {med:.1f} ms")
    ax1.axvline(p95, color="#c92a2a", linestyle="--", linewidth=2.0, label=f"p95: {p95:.1f} ms")
    ax1.set_xlabel("Single-Image Inference Latency (ms)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Iteration Count", fontsize=10, fontweight="bold")
    ax1.set_title("Single-Image CPU Inference Latency Distribution", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.8)
    ax1.grid(True, linestyle="--", alpha=0.35)

    bs = [r["batch_size"] for r in batch_results]
    tput = [r["throughput_images_per_sec"] for r in batch_results]
    per_img = [r["per_image_median_ms"] for r in batch_results]

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(bs, tput, "-o", color="#2b8a3e", linewidth=2.2, markersize=6, label="Throughput (img/s)")
    l2 = ax2_twin.plot(bs, per_img, "--s", color="#e67700", linewidth=2.2, markersize=6, label="Latency / Image (ms)")
    ax2.set_xticks(bs)
    ax2.set_xlabel("Batch Size", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Throughput (images / sec)", fontsize=10, fontweight="bold", color="#2b8a3e")
    ax2_twin.set_ylabel("Median Latency per Image (ms)", fontsize=10, fontweight="bold", color="#e67700")
    ax2.set_title("Batch Throughput & Amortized Per-Image Latency", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="center right", fontsize=8.8)

    fig.suptitle(
        "Phase 7 — ResNet18 Checkpoint (best_model.pt) CPU Latency & Throughput Benchmark",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="results/best_model.pt")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--timed-runs", type=int, default=50)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto", choices=["auto", "torch", "onnx"])
    parser.add_argument("--no-onnx-export", action="store_true")
    parser.add_argument("--output-dir", default="ml_validation/reports/phase7_efficiency_benchmark")
    args = parser.parse_args(argv)

    report = run_benchmark(
        args.checkpoint,
        Path(args.output_dir),
        warmup_runs=args.warmup_runs,
        timed_runs=args.timed_runs,
        batch_sizes=args.batch_sizes,
        device=args.device,
        backend=args.backend,
        export_onnx=not args.no_onnx_export,
    )
    lat = report["single_image_forward_latency_ms"]
    print(
        f"[Phase 7] Benchmarked {Path(args.checkpoint).name} on {report['environment']['cpu_model']} "
        f"({report['execution_backend']} backend): "
        f"params={report['model_complexity']['total_learnable_params']:,}, "
        f"median={lat['median_p50_ms']:.2f} ms, p95={lat['p95_ms']:.2f} ms, "
        f"throughput={lat['throughput_images_per_sec']:.2f} img/s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
