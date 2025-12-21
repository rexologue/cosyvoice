import argparse
import os
import argparse
import os
import shutil
from pathlib import Path
from .fill_template import fill_template


def copy_template_dir(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare Triton model repository for bi-stream runtime")
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine-dir", type=str, required=True)
    parser.add_argument("--llm-tokenizer-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--max-queue-delay", type=int, default=1000)
    parser.add_argument("--decoupled", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--max-sessions", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=512)
    parser.add_argument("--mix-ratio", default="5,15")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # copy template dirs
    for name in os.listdir(args.template_root):
        src = args.template_root / name
        if src.is_dir():
            copy_template_dir(src, args.output_root / name)

    substitutions = {
        "engine_dir": args.engine_dir,
        "llm_tokenizer_dir": args.llm_tokenizer_dir,
        "model_dir": args.model_dir,
        "triton_max_batch_size": str(args.max_batch_size),
        "max_queue_delay_microseconds": str(args.max_queue_delay),
        "bls_instance_num": str(args.instances),
        "decoupled_mode": "True" if args.decoupled else "False",
        "max_sessions": str(args.max_sessions),
        "max_batch_tokens": str(args.max_batch_tokens),
        "mix_ratio": args.mix_ratio,
        "log_level": args.log_level,
    }

    # fill templates
    for pbtxt in args.output_root.rglob("config.pbtxt"):
        fill_template(pbtxt, substitutions)


if __name__ == "__main__":
    main()
