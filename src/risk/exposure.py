"""
Phase 9 / exposure — prevent the backtest (and eventually the live
scanner) from filling every open slot with correlated same-bucket
stocks. If 8 of 10 open positions are all smallcaps, a single
smallcap-sector selloff can hurt far more than the position count
alone suggests.
"""

import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs import config


def can_open_new_position(open_positions: Dict[str, dict], cap_bucket: str, max_per_bucket: int = None) -> bool:
    """
    open_positions: symbol -> position dict, each expected to have a
    'cap_bucket' key.
    """
    max_per_bucket = max_per_bucket if max_per_bucket is not None else config.MAX_POSITIONS_PER_CAP_BUCKET
    current_count = sum(1 for p in open_positions.values() if p.get("cap_bucket") == cap_bucket)
    return current_count < max_per_bucket
