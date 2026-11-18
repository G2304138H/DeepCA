#!/usr/bin/env python3
"""Evaluate a shared RCA/LCA checkpoint using the existing evaluation pipeline."""
from deepca.config import load_config
from evaluate import main as evaluate_main, parse_args


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if config.get('data', {}).get('vessel_type') != 'both':
        raise ValueError('evaluate_unified.py requires data.vessel_type: both.')
    return evaluate_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
