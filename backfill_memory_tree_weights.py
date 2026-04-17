from __future__ import annotations

import argparse

from memory_tree_core import MemoryTreeCore
from runtime_config import configure_logging, get_runtime_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Mellon's weighted memory tree typed columns.")
    parser.add_argument("--user-id", help="Optional user_id scope for the backfill.")
    parser.add_argument("--batch-size", type=int, default=500, help="Maximum number of rows to inspect.")
    args = parser.parse_args()

    configure_logging()
    memory_tree = MemoryTreeCore()
    updated = memory_tree.backfill_normalized_fields(user_id=args.user_id, batch_size=max(1, args.batch_size))
    print(
        {
            "updated_rows": updated,
            "user_id": args.user_id,
            "batch_size": max(1, args.batch_size),
            "runtime": get_runtime_snapshot(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
