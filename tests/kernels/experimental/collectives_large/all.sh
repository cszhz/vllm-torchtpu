#!/bin/bash
# One device job: Pallas large collectives (chunking), decode-side flag check, overlap check.
D=tests/kernels/experimental/collectives_large
bash $D/run2.sh
bash $D/decode_flags.sh
bash $D/overlap.sh
