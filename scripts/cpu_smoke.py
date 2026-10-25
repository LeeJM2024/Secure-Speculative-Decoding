"""Run the small cached Qwen pair through five decoding configurations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="cpu_smoke")
    parser.add_argument("--output-len", type=int, default=32)
    args = parser.parse_args()
    if args.output_len < 1:
        parser.error("--output-len must be positive")
    if not args.experiment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.experiment):
        parser.error("--experiment must contain letters, numbers, underscores or hyphens")
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / ".results/cpu_models.json"
    if not manifest_path.is_file():
        parser.error("Download both models first: .venv-cpu/bin/python scripts/download_cpu_models.py")
    models = json.loads(manifest_path.read_text())
    dest = root / ".results" / args.experiment
    if dest.exists():
        parser.error("Result directory already exists; select a new --experiment name")
    logs = dest / "logs"
    logs.mkdir(parents=True)
    jobs = [
        ("target", ["--method", "ar_target", "--eta_len", "0"]),
        ("draft", ["--method", "ar_draft", "--eta_len", "0"]),
        ("sd", ["--method", "sd", "--eta_len", "0"]),
        ("lossy", ["--method", "lossy_sd", "--lossy_epsilon", "0.2", "--eta_len", "0"]),
        ("secure", ["--method", "lossy_sd", "--lossy_epsilon", "0.2", "--eta_len", "1"]),
    ]
    common = [
        sys.executable, "-u", "bench/bench.py", "--backend", "cpu",
        "--target_model", "Qwen/Qwen3-1.7B", "--draft_model", "Qwen/Qwen3-0.6B",
        "--target_revision", models["Qwen/Qwen3-1.7B"]["revision"],
        "--draft_revision", models["Qwen/Qwen3-0.6B"]["revision"], "--local_files_only",
        "--dataset", "jailbreaking", "--num_seqs", "1", "--max_num_seqs", "1",
        "--output_len", str(args.output_len), "--decode_tps_n", str(args.output_len),
        "--max_model_len", "4096", "--speculate_k", "5", "--cpu_threads", "4",
        "--cpu_dtype", "bfloat16", "--seed", "0", "--temp", "0", "--dtemp", "0",
        "--think", "0", "--eta_schedule", "step", "--eta_start", "0", "--eta_end", "1",
        "--no_utility_eval", "--experiment_name", args.experiment,
    ]
    for label, flags in jobs:
        log = logs / f"{label}.log"
        print(f"[{label}] Running; log: {log}", flush=True)
        with log.open("w") as f:
            result = subprocess.run(common + flags, cwd=root, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-6000:], file=sys.stderr)
            raise SystemExit(result.returncode)
        print(f"[{label}] Finished", flush=True)
    reports = []
    for path in sorted(dest.rglob("*.json")):
        if path.name.startswith("raw_"):
            continue
        report = json.loads(path.read_text())
        meta, summary = report["meta"], report["summary"]
        eff = summary["efficiency"]
        reports.append({
            "method": meta["method"], "eta_len": meta["eta_len"],
            "tokens": eff["decode_tokens"], "generation_time_s": eff["time_s"],
            "output_tps": eff["output_tps"], "acceptance_rate": eff.get("acceptance_rate"),
            "peak_rss_gb": summary["resources"]["process_peak_rss_gb"],
            "report": str(path),
        })
    (dest / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
