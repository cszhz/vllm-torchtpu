# Point 8 (c72) with DSpark on the attention DP2 x DCP16 decode: 3 drafts at
# synthetic acceptance 2.87 for decode batches up to 64 (verify as two
# 32-request launches until the kernel takes 64 x 4 rows in one launch).
RECIPE_SET=oct16
RECIPE_CONCURRENCY=72
RECIPE_DECODE=dspark_dp2
RECIPE_DECODE_MAX_NUM_SEQS=64
RECIPE_DRAFTS=3
RECIPE_ACCEPTANCE=2.87
RECIPE_DRAFTS_PER_BATCH_SIZE='[[1,64,3]]'
RECIPE_PREFILL=long
RECIPE_STORE_SERVERS=1
source "$(dirname "${BASH_SOURCE[0]}")/../kimi-k3-megakernel/recipe-base.sh"
