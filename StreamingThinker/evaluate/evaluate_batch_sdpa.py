from common import REPO_ROOT, build_parser, run_evaluation


def parse_args():
    parser = build_parser(
        default_mode="batch",
        default_config=REPO_ROOT / "configs" / "eval" / "params_qwen_D21.json",
        default_attn_implementation="sdpa",
        description="Evaluate StreamingThinker in batch generation mode with SDPA attention.",
    )
    parser.set_defaults(max_new_tokens=2048)
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
