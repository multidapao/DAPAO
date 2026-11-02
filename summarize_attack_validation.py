#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


BIT_LENGTH = {
    "hidden": 30,
    "cin": 30,
    "pimog": 30,
    "stegastamp": 100,
    "rivagan": 32,
    "dwtdct": 32,
    "dwtdctsvd": 32,
    "signature": 48,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join target-watermark verification CSV with attack record metrics, "
            "then report ASR and image quality."
        )
    )
    parser.add_argument("--task", choices=["evade", "spoof"], required=True)
    parser.add_argument("--algo", choices=sorted(BIT_LENGTH), required=True)
    parser.add_argument("--decoder-csv", type=Path, required=True,
                        help="Decoder results with image and bit_accuracy columns; legacy column names are also accepted.")
    parser.add_argument("--record-csv", type=Path, required=True,
                        help="Attack records: sample ID, perturbation, PSNR, SSIM; extra columns are ignored.")
    parser.add_argument("--suffix", default="20", help="Record epoch suffix, or 'latest'.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--pvalue", type=float, default=0.05)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--per-sample-csv", type=Path, default=None)
    return parser.parse_args()


def find_threshold(nbit, pvalue):
    comb = getattr(math, "comb", None)
    if comb is None:
        def comb(n, k):
            return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    for k in range(nbit // 2, nbit):
        tail = sum(comb(nbit, i) for i in range(k + 1, nbit + 1)) * (0.5 ** nbit)
        if pvalue > tail:
            return (k - 1) / nbit
    return 1.0


def sample_key_from_image(value):
    stem = Path(str(value).strip()).stem
    if stem.endswith("-8"):
        stem = stem[:-2]
    return stem.split("-")[0]


def parse_record_id(sample_id):
    parts = str(sample_id).strip().split("-")
    if len(parts) == 1:
        return parts[0], None
    epoch = parts[-1]
    return parts[0], epoch


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_acc(decoder_csv):
    grouped = {}
    with decoder_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if {"image", "bit_accuracy"} <= fields:
            image_column, accuracy_column = "image", "bit_accuracy"
        elif {"cnm img", "cnm acc_c"} <= fields:
            # Read existing decoder exports without rewriting experimental data.
            image_column, accuracy_column = "cnm img", "cnm acc_c"
        else:
            raise SystemExit(f"{decoder_csv} must contain 'image' and 'bit_accuracy' columns.")
        for row in reader:
            acc = safe_float(row.get(accuracy_column))
            if acc is None:
                continue
            key = sample_key_from_image(row.get(image_column))
            grouped.setdefault(key, []).append(acc)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def choose_record_rows(rows, suffix):
    if suffix != "latest":
        return [row for row in rows if row["epoch"] == suffix or row["epoch"] is None]

    latest = {}
    for row in rows:
        epoch = row["epoch"]
        rank = int(epoch) if epoch is not None and epoch.isdigit() else 10**9
        current = latest.get(row["name"])
        if current is None or rank > current[0]:
            latest[row["name"]] = (rank, row)
    return [item[1] for item in latest.values()]


def load_record(record_csv, suffix):
    rows = []
    with record_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            perturb = safe_float(row[1])
            psnr = safe_float(row[2])
            ssim = safe_float(row[3])
            if None in (perturb, psnr, ssim):
                continue
            name, epoch = parse_record_id(row[0])
            rows.append(
                {
                    "name": name,
                    "epoch": epoch,
                    "perturb": perturb,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )
    selected = choose_record_rows(rows, suffix)
    grouped = {}
    for row in selected:
        grouped.setdefault(row["name"], []).append(row)

    averaged = {}
    for name, values in grouped.items():
        averaged[name] = {
            metric: sum(v[metric] for v in values) / len(values)
            for metric in ("perturb", "psnr", "ssim")
        }
    return averaged


def mean(rows, key):
    if not rows:
        return float("nan")
    return sum(row[key] for row in rows) / len(rows)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    threshold = args.threshold
    if threshold is None:
        threshold = find_threshold(BIT_LENGTH[args.algo], args.pvalue)

    acc_by_name = load_acc(args.decoder_csv)
    metrics_by_name = load_record(args.record_csv, args.suffix)

    joined = []
    for name, metrics in metrics_by_name.items():
        if name not in acc_by_name:
            continue
        acc = acc_by_name[name]
        success = acc < threshold if args.task == "evade" else acc >= threshold
        joined.append({"name": name, "acc_c": acc, "success": success, **metrics})

    if not joined:
        raise SystemExit("No overlapping samples between decoder CSV and record CSV.")

    success_rows = [row for row in joined if row["success"]]
    summary = {
        "task": args.task,
        "algo": args.algo,
        "threshold": threshold,
        "samples": len(joined),
        "success_samples": len(success_rows),
        "success_rate": len(success_rows) / len(joined),
        "mean_acc_c": mean(joined, "acc_c"),
        "mean_perturb": mean(joined, "perturb"),
        "mean_psnr": mean(joined, "psnr"),
        "mean_ssim": mean(joined, "ssim"),
        "success_mean_perturb": mean(success_rows, "perturb"),
        "success_mean_psnr": mean(success_rows, "psnr"),
        "success_mean_ssim": mean(success_rows, "ssim"),
    }

    print(f"task: {summary['task']}")
    print(f"algo: {summary['algo']}")
    print(f"threshold: {summary['threshold']:.10f}")
    print(f"samples: {summary['samples']}")
    print(f"success_samples: {summary['success_samples']}")
    print(f"success_rate: {summary['success_rate']:.10f}")
    print(f"mean_acc_c: {summary['mean_acc_c']:.10f}")
    print(f"mean_perturb: {summary['mean_perturb']:.10f}")
    print(f"mean_psnr: {summary['mean_psnr']:.10f}")
    print(f"mean_ssim: {summary['mean_ssim']:.10f}")
    print(f"success_mean_perturb: {summary['success_mean_perturb']:.10f}")
    print(f"success_mean_psnr: {summary['success_mean_psnr']:.10f}")
    print(f"success_mean_ssim: {summary['success_mean_ssim']:.10f}")

    if args.summary_csv:
        write_csv(args.summary_csv, [summary], list(summary))
    if args.per_sample_csv:
        write_csv(
            args.per_sample_csv,
            joined,
            ["name", "acc_c", "success", "perturb", "psnr", "ssim"],
        )


if __name__ == "__main__":
    main()
