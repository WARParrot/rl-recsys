import argparse, json
from pathlib import Path
from .data import download, prepare
from .experiment import train, final_evaluation


def main(argv=None):
    p = argparse.ArgumentParser(prog="rl-recsys")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("download")
    q.add_argument("--config", default="configs/pilot.json")
    q.add_argument("--output", default="data/listens.parquet")
    q = sub.add_parser("prepare")
    q.add_argument("--config", default="configs/pilot.json")
    q.add_argument("--input", default="data/listens.parquet")
    q.add_argument("--output", default="data/prepared")
    q = sub.add_parser("train")
    q.add_argument("--config", default="configs/pilot.json")
    q.add_argument("--data", default="data/prepared")
    q.add_argument("--run", required=True)
    q = sub.add_parser("evaluate")
    q.add_argument("--data", default="data/prepared")
    q.add_argument("--run", required=True)
    args = p.parse_args(argv)
    if args.command in ["download", "prepare"]:
        cfg = json.loads(Path(args.config).read_text())
        result = download(cfg, args.output) if args.command == "download" else prepare(args.input, cfg, args.output)
        if result is not None: print(json.dumps(result, indent=2))
    elif args.command == "train":
        train(json.loads(Path(args.config).read_text()), args.data, args.run)
    else:
        print(json.dumps(final_evaluation(args.data, args.run), indent=2))

if __name__ == "__main__": main()
