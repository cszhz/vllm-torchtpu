#!/bin/bash
# Job #1 shards (probe 01). usage: run_shard.sh <i>
cd /root/kp
export PYTHONPATH=/root/kp/src:/root/kp:$PYTHONPATH
D=tests/kernels/fused_moe/mixed_plan
O=$CDK_OUTPUT_DIR/kplan
mkdir -p $O
python3 -c "import vllm_torchtpu.kernels.fused_moe.v2.host as h; print('host from', h.__file__)"
md5sum src/vllm_torchtpu/kernels/fused_moe/v2/host.py src/vllm_torchtpu/kernels/fused_moe/v2/layer.py $D/*.py
case $1 in
0) timeout -s KILL 2400 python3 $D/bench.py --cells mix128,mix256 --kinds served,served_pad --arms plan,combine --trace --out $O/b0 ;;
1) pip install -q pytest >/dev/null 2>&1; timeout -s KILL 2400 python3 -m pytest -q -s -p no:cacheprovider $D/test_accuracy.py ;;
2) timeout -s KILL 1800 python3 $D/probe_primitives.py ;;
3) timeout -s KILL 2400 python3 $D/bench.py --cells mix32,mix64,mix128,mix256 --kinds uniform,one_shard --arms plan --me 0 --out $O/b3 ;;
4) KPLAN_RS=16,96,256 KPLAN_PROBE_OUT=kplan_probe00b.json timeout -s KILL 1800 python3 $D/probe_primitives.py ;;
5) timeout -s KILL 1800 python3 $D/probe_lowering.py ;;
6) timeout -s KILL 1500 python3 $D/probe_tripcount.py ;;
esac
