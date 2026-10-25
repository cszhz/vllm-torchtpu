# SPDX-License-Identifier: Apache-2.0
"""Compare two record_actuals files byte for byte: python compare_actuals.py A.npz B.npz"""

import sys

import numpy as np

A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
common = sorted(set(A.files) & set(B.files))
diff = [k for k in common if A[k].shape != B[k].shape or not np.array_equal(A[k], B[k])]
only = sorted(set(A.files) ^ set(B.files))
for k in diff[:40]:
    print("DIFF", k)
print(
    f"compared {len(common)} actuals; {len(diff)} differ; {len(only)} recorded on one side only"
)
print("BITEXACT" if not diff else "NOT BITEXACT")
sys.exit(1 if diff else 0)
