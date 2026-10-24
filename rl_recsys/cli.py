import argparse, json
from pathlib import Path
from .config import validate_config
from .data import download, prepare
from .experiment import train, final_evaluation
from .preflight import run_preflight


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
    q = sub.add_parser("preflight")
    q.add_argument("--config", default="configs/pilot.json")
    q.add_argument("--device", choices=["cpu", "cuda"], default=None)
    q = sub.add_parser("import-run", help="Verify and import uploaded historical inputs")
    q.add_argument("--bundle", required=True)
    q.add_argument("--assets", required=True, help="Directory with prepared.npz and all checkpoints")
    q.add_argument("--output", required=True)
    q = sub.add_parser("verify-run", help="Reconstruct historical metrics from targets, without GPU")
    q.add_argument("--data", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("reanalyze", help="New exploratory inference; never overwrite historical evaluation")
    q.add_argument("--data", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--batch-size", type=int, default=128)
    q.add_argument("--max-minutes", type=float, default=240)
    q.add_argument("--rerank-weights", type=float, nargs="+", default=[0., .1, 1., 10.])
    q.add_argument("--skip-diagnostics", action="store_true")
    q = sub.add_parser("bundle-reanalysis", help="ZIP64 bundle including all data and checkpoints")
    q.add_argument("--imported", required=True)
    q.add_argument("--analysis", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("followup", help="Opt-in new training controls, not inference or a fresh test")
    q.add_argument("--data", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    q.add_argument("--steps", type=int, default=1000)
    q.add_argument("--repeat-penalty", type=float, default=.1)
    q = sub.add_parser("evaluate-followup", help="Explicit exploratory inference of frozen follow-up controls")
    q.add_argument("--data", required=True)
    q.add_argument("--run", required=True)
    q.add_argument("--followup", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--batch-size", type=int, default=128)
    q.add_argument("--max-minutes", type=float, default=240)
    args = p.parse_args(argv)
    if args.command in {"import-run", "verify-run", "reanalyze", "bundle-reanalysis", "followup", "evaluate-followup"}:
        from . import reanalysis
        if args.command == "import-run":
            result = reanalysis.import_run(args.bundle, args.assets, args.output)
        elif args.command == "verify-run":
            result = reanalysis.reconstruct_metrics(args.run, args.data, args.output)
        elif args.command == "reanalyze":
            result = reanalysis.run_reanalysis(args.data, args.run, args.output,
                        batch_size=args.batch_size, max_minutes=args.max_minutes,
                        rerank_weights=args.rerank_weights, diagnostics=not args.skip_diagnostics)
        elif args.command == "bundle-reanalysis":
            result = reanalysis.bundle_reanalysis(args.imported, args.analysis, args.output)
        elif args.command == "evaluate-followup":
            from .followup_evaluate import evaluate_followup
            result = evaluate_followup(args.data, args.run, args.followup, args.output,
                                       batch_size=args.batch_size, max_minutes=args.max_minutes)
        else:
            from .followup import run_followup
            result = run_followup(args.data, args.run, args.output, seeds=args.seeds,
                                  steps=args.steps, repeat_penalty=args.repeat_penalty)
        print(json.dumps(result, indent=2))
        return
    if args.command == "evaluate":
        cfg = json.loads((Path(args.run) / "provenance.json").read_text())["config"]
    else:
        cfg = json.loads(Path(args.config).read_text())
    if args.command != "download":
        validate_config(cfg)
    if args.command in ["download", "prepare"]:
        result = download(cfg, args.output) if args.command == "download" else prepare(args.input, cfg, args.output)
        if result is not None: print(json.dumps(result, indent=2))
    elif args.command == "train":
        train(cfg, args.data, args.run)
    elif args.command == "evaluate":
        print(json.dumps(final_evaluation(args.data, args.run), indent=2))
    else:
        print(json.dumps(run_preflight(cfg, device=args.device), indent=2))

if __name__ == "__main__": main()
