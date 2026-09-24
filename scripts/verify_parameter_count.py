#!/usr/bin/env python3
"""Reproduce the official unique trainable parameter count."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from anka import RMW3, RMW3_CONFIG, count_parameters


model = RMW3(RMW3_CONFIG)
parameter_count = count_parameters(model)
print(f"ANKA-50M RMW3 parameters: {parameter_count:,}")
assert parameter_count == 48_944_657
assert parameter_count <= 50_000_000
assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()
print("Parameter limit and weight tying: PASS")
