from dataclasses import dataclass


@dataclass
class Metrics:
    valid_pops: int = 0
    expanded: int = 0
    enqueued: int = 0
    stale_pops: int = 0
    queue_peak: int = 0
    visited: int = 0
    evaluated: int = 0
    prunes: int = 0
