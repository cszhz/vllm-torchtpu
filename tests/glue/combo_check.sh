#!/bin/bash
# Bit-exactness of the glue flags with the other combo flags on (TPU_FP8_SKINNY_TABLE=v2 in every run).
set -x
export TPU_FP8_SKINNY_TABLE=v2
PIN_SITES=${PIN_SITES:-skinny:proj:combine:select:plan} bash tests/glue/pin_parity.sh || echo "PIN PARITY FAILED"
CAP=${CAP:-24} bash tests/glue/vmem_tests.sh
