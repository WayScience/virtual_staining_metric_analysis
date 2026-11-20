"""
Command-line interface for the degradation pipeline.

See 1.image_degradation_simulation/README.md for detailed usage instructions.
"""

import argparse
from pathlib import Path

import numpy as np

from utils.degradation import pipeline
from utils.fov_ingestion.collection import MicroscopyCollection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'generate', 'report'),
                        nargs='?', default='run',
                        help='Default: generate all 36 variants and save montages')
    parser.add_argument('--config', help='Project YAML; paths resolve relative to that file')
    parser.add_argument('--reference', help='Override the configured PedCP reference')
    parser.add_argument('--output-root', help='Override the configured degradation output root')
    parser.add_argument('--channels', nargs='+', help='Default: all intensity channels, including BF')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--fov-ids-file', help='One FOV ID per line; optional frozen cohort')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--results-dir', help='Montage output; default: stage results/degradation')
    parser.add_argument('--montage-fov', help='Selected FOV for all channel montages')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='Centered montage crop in pixels; 0 displays the full FOV')
    args = parser.parse_args(argv)
    if args.dry_run and args.command == 'report':
        parser.error('--dry-run is not supported for report')
    if args.crop_size < 0:
        parser.error('--crop-size must be nonnegative')
    if args.config is not None or args.reference is None or args.output_root is None:
        from utils.pedcp_workflow import WorkflowPaths
        from utils.project_config import load_project_config
        paths = WorkflowPaths(load_project_config(args.config))
        args.reference = args.reference or paths.reference
        args.output_root = args.output_root or paths.degradations
    ref = MicroscopyCollection(args.reference)
    ids = (Path(args.fov_ids_file).read_text().splitlines() if args.fov_ids_file else list(ref.fovs.fov_id))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        ids = list(np.random.default_rng(args.sample_seed).choice(ids, size=min(args.limit, len(ids)), replace=False))
    if args.command != 'report':
        pipeline.run(ref, args.output_root, channels=args.channels, fov_ids=ids,
                     workers=args.workers, dry_run=args.dry_run)
    if args.command != 'generate' and not args.dry_run:
        results_dir = (Path(args.results_dir) if args.results_dir else
                       Path(__file__).resolve().parents[2] /
                       '1.image_degradation_simulation' / 'results' / 'degradation')
        for path in pipeline.report(ref, args.output_root, results_dir,
                                    fov_id=args.montage_fov, channels=args.channels,
                                    crop_size=args.crop_size or None, seed=args.sample_seed):
            print(path, flush=True)


if __name__ == '__main__':
    main()
