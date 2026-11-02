import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def image_files(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Image directory does not exist: {}".format(directory))
    paths = sorted((p.resolve() for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: p.name)
    if not paths:
        raise ValueError("No supported images in {}".format(directory))
    names = [p.name for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Image names must be unique after resolving symlinks")
    stems = [p.name.split(".")[0] for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Image names collide under the legacy output naming convention")
    return paths


def read_carriers(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Carrier list does not exist: {}".format(path))
    carriers = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = Path(line.strip()).expanduser()
        item = (path.parent / item).resolve() if not item.is_absolute() else item.resolve()
        if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("Invalid carrier image: {}".format(item))
        carriers.append(item)
    return carriers


def parser(method):
    if method not in {"remove", "forgery"}:
        raise ValueError("Unknown method: {}".format(method))
    p = argparse.ArgumentParser(description=(
        "Baseline no-box removal with selected-feature SSIM."
        if method == "remove" else "Baseline no-box watermark forgery."))
    p.set_defaults(method=method)
    p.add_argument("--input", type=Path, required=True, help="Flat directory of watermarked donors.")
    p.add_argument("--output", type=Path, required=True, help="New or empty output directory.")
    p.add_argument("--experiment-name", default="attack", help="Name for diagnostic files inside the run directory.")
    p.add_argument("--gpu", default="0", help="Physical GPU index or CUDA device UUID.")
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--epochs", type=int, default=21, help="At least 11, and 1 modulo 5 to save the final iteration.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="Use the first N donors in sorted filename order.")
    p.add_argument("--dry-run", action="store_true", help="Validate and print configuration without loading models or writing files.")
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--selected-clusters", type=int, default=2)
    if method == "forgery":
        p.add_argument("--mode", choices=["dual", "transfer"], default="dual")
        p.add_argument("--carrier-list", type=Path, required=True,
                       help="Ordered clean carrier paths, one per donor.")
    return p


def prepare(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.experiment_name):
        raise ValueError("experiment-name must be 1-80 letters, digits, underscores, dots or hyphens, starting with a letter or digit")
    if args.epochs < 11 or (args.epochs - 1) % 5:
        raise ValueError("epochs must be >=11 and 1 modulo 5 (11, 16, 21, ...)")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not args.gpu or "," in args.gpu or args.gpu == "-1":
        raise ValueError("Select one CUDA GPU per run")
    if not 1 <= args.selected_clusters < args.clusters <= 64:
        raise ValueError("Require 1 <= selected-clusters < clusters <= 64")
    donors = image_files(args.input)
    if args.limit:
        donors = donors[:args.limit]
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty; nothing will be deleted: {}".format(output))
    input_root = args.input.expanduser().resolve()
    if output == input_root or input_root in output.parents or output in input_root.parents:
        raise ValueError("Input and output directory trees must not overlap")
    carriers = []
    config = {
        "method": args.method, "entrypoint": args.method + ".py",
        "experiment_name": args.experiment_name,
        "gpu": args.gpu, "seed": args.seed, "epochs": args.epochs,
        "batch_size": 1, "image_size": 256,
        "clusters": args.clusters, "selected_clusters": args.selected_clusters,
        "input": str(input_root), "output": str(output),
        "samples": len(donors), "status": "prepared",
    }
    if args.method == "forgery":
        if len(donors) > 100:
            raise ValueError("Forgery supports at most 100 pairs per run; split into separate runs")
        carriers = read_carriers(args.carrier_list)
        if len(carriers) < len(donors):
            raise ValueError("Not enough carriers for the selected donors")
        carriers = carriers[:len(donors)]
        if any(a == b for a, b in zip(donors, carriers)):
            raise ValueError("A forgery carrier must differ from its watermarked donor")
        config.update(mode=args.mode, function="attack_" + args.mode)
        default_eps = 0.078 if args.mode == "dual" else 0.108
    else:
        config["function"] = "attack"
        default_eps = 0.108
    eps = args.eps if args.eps is not None else default_eps
    if not math.isfinite(eps) or not 0 < eps <= 1:
        raise ValueError("eps must be finite and in (0, 1]")
    config["eps"] = eps
    return config, donors, carriers


def execute(config, donors, carriers):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu"]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to silently run a CPU attack")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    work = output / "work"
    work.mkdir()
    donor_dir = work / "donors"
    donor_dir.mkdir()
    for donor in donors:
        (donor_dir / donor.name).symlink_to(donor)
    carrier_file = work / "carriers.txt"
    carrier_file.write_text("".join(str(p) + "\n" for p in carriers))
    with (output / "pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["output_name", "donor", "carrier", "quality_reference"])
        for i, donor in enumerate(donors):
            carrier = carriers[i] if carriers else ""
            writer.writerow([donor.name.split(".")[0] + "-8.png", donor, carrier, carrier or donor])
    environment = os.environ.copy()
    environment.update({
        "TRAIN": str(donor_dir),
        "OUT_PATH": str(output / "images"), "CARRIER_LIST": str(carrier_file),
        "EXPERIMENT_NAME": config["experiment_name"], "IMAGE_SIZE": "256",
        "EPS": str(config["eps"]), "EPOCH": str(config["epochs"]),
        "PYTHONHASHSEED": str(config["seed"]), "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    environment["K_CLUSTER"] = str(config["clusters"])
    environment["N_CLUSTER"] = str(config["selected_clusters"])
    config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0), status="running")
    config_file = output / "run.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    worker = (
        "import sys,random,runpy,numpy as np,torch; "
        "sys.path.insert(0,sys.argv[1]); seed=int(sys.argv[3]); "
        "random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); "
        "runpy.run_path(sys.argv[2],run_name='dapao_attack')[sys.argv[4]]()"
    )
    print("Running {} on GPU {}; log: {}".format(
        config["method"], config["gpu"], output / "attack.log"), flush=True)
    try:
        with (output / "attack.log").open("w") as log:
            subprocess.run([sys.executable, "-B", "-c", worker, str(ROOT),
                            str(ROOT / config["entrypoint"]), str(config["seed"]), config["function"]],
                           cwd=str(work), env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        expected = {p.name.split(".")[0] + "-8.png" for p in donors}
        actual = {p.name for p in (output / "images").glob("*.png")}
        if actual != expected:
            raise RuntimeError("Output image set does not match the requested donors")
        config["status"] = "completed"
    except BaseException:
        config["status"] = "failed"
        raise
    finally:
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    print("Completed. Images: {}".format(output / "images"))


def main(method):
    p = parser(method)
    args = p.parse_args()
    try:
        config, donors, carriers = prepare(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    execute(config, donors, carriers)
