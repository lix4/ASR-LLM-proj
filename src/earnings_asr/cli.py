"""Command-line entrypoints; --help works without importing the model."""

import argparse
import json

from .common import read_config, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Noise- and reverberation-robust English ASR experiments")
    parser.add_argument("--config", default="configs/default.json")
    sub = parser.add_subparsers(dest="command", required=True)
    environment = sub.add_parser("doctor", help="Record hardware, software and dataset access")
    environment.add_argument("--output", default="reports/environment.json")
    environment.add_argument("--offline", action="store_true")
    sub.add_parser("data-plan", help="Inspect selected dataset shards without downloading audio")
    sub.add_parser("download-data", help="Download only S/train, dev/validation and test/test shards")
    sub.add_parser("robust-data-inventory", help="Hash and report first-stage OpenSLR archives")
    sub.add_parser("extract-robust-data", help="Safely extract LibriSpeech, MUSAN and SLR28 archives")
    sub.add_parser("prepare-robust-data", help="Build fixed speech, noise, RIR and corruption manifests")
    prepare = sub.add_parser("prepare-data", help="Validate and build fixed manifests")
    prepare.add_argument("--output")
    prepare.add_argument("--recording-id-pattern", help="Verified regex with recording ID in capture group 1")
    export = sub.add_parser("export-sft", help="Export a train/dev manifest to official SFT format")
    export.add_argument("manifest")
    export.add_argument("output")
    transcribe = sub.add_parser("transcribe", help="Transcribe local English audio")
    transcribe.add_argument("audio")
    transcribe.add_argument("--checkpoint")
    transcribe.add_argument("--output", default="outputs/transcription.json")
    evaluate = sub.add_parser("evaluate", help="Score a fixed manifest and save evidence")
    evaluate.add_argument("manifest")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--allow-test", action="store_true")
    training = sub.add_parser("train", help="Run the pinned official SFT adapter")
    training.add_argument("--train-file", required=True)
    training.add_argument("--dev-file", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--smoke", action="store_true", help="One optimizer step; not domain fine-tuning")
    training.add_argument("--max-steps", type=int, default=-1)
    training.add_argument("--checkpoint-steps", type=int,
                          help="Save and run diagnostic eval every N optimizer steps")
    training.add_argument("--augmentation", choices=("clean", "robust"), default="clean")
    training.add_argument("--projector-mode", choices=("train", "frozen", "lora"),
                          help="Train projectors directly, freeze them, or add LoRA")
    training.add_argument("--checkpoint")
    service = sub.add_parser("serve", help="Start the local API and upload page")
    service.add_argument("--host", default="127.0.0.1")
    service.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    try:
        if args.command == "doctor":
            from .environment import inspect_environment
            result = inspect_environment(config, network=not args.offline)
            write_json(args.output, result)
        elif args.command == "data-plan":
            from .data import shard_plan
            result = shard_plan(config)
            write_json("reports/data-download-plan.json", result)
        elif args.command == "download-data":
            from .data import download_spgi
            result = {"path": str(download_spgi(config))}
        elif args.command == "robust-data-inventory":
            from .robust_data import archive_inventory
            result = archive_inventory(config)
        elif args.command == "extract-robust-data":
            from .robust_data import extract_archives
            result = extract_archives(config)
        elif args.command == "prepare-robust-data":
            from .robust_data import prepare_all_manifests
            result = prepare_all_manifests(config)
        elif args.command == "prepare-data":
            from .data import build_spgi
            result = build_spgi(config, args.output, args.recording_id_pattern)
        elif args.command == "export-sft":
            from .data import export_sft
            result = export_sft(args.manifest, args.output)
        elif args.command == "transcribe":
            from .inference import QwenBackend, transcribe_file
            backend = QwenBackend(config, args.checkpoint)
            result = transcribe_file(backend, args.audio, config)
            result["model"] = backend.metadata
            result["model_load_seconds_excluding_download"] = backend.load_seconds
            write_json(args.output, result)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(config, args.manifest, args.output, args.checkpoint, args.allow_test)
        elif args.command == "train":
            from .training import train
            result = train(config, args.train_file, args.dev_file, args.output, args.smoke,
                           args.max_steps, augmentation_mode=args.augmentation,
                           checkpoint=args.checkpoint, projector_mode=args.projector_mode,
                           checkpoint_steps=args.checkpoint_steps)
        elif args.command == "serve":
            import uvicorn
            from .service import create_app
            uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)
            return
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
