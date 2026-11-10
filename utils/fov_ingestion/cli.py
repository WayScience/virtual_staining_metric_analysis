import argparse

from utils.fov_ingestion.ingest import ingest


def main():
    p = argparse.ArgumentParser(description='Ingest LoadData and binary outlines into immutable OME-Arrow FOV shards.')
    p.add_argument('inputs', nargs='+', help='CSV or Parquet paths (shell glob expansion supported)')
    p.add_argument('--outline-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dataset-id', required=True)
    p.add_argument('--image-root')
    p.add_argument('--channels', nargs='+')
    p.add_argument('--pixel-size-um', type=float)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--dry-run', action='store_true')
    args = vars(p.parse_args())
    result = ingest(**args)
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()
