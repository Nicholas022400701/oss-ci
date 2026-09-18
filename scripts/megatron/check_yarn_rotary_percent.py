"""Check that YarnRotaryEmbedding honors rotary_percent the same way RotaryEmbedding does.

Runs on CPU. torch.cuda.current_device is patched so the lazy device moves inside
get_emb resolve to the CPU. With scaling_factor=1.0 YaRN reduces to plain RoPE, so
the two embeddings must agree exactly for every rotary_percent.
"""

import sys

import torch

torch.cuda.current_device = lambda: torch.device("cpu")

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding

KV_CHANNELS = 64
SEQ_LEN = 16

failures = []
for rotary_percent in (1.0, 0.5, 0.25):
    rope = RotaryEmbedding(KV_CHANNELS, rotary_percent, use_cpu_initialization=True)
    yarn = YarnRotaryEmbedding(
        KV_CHANNELS,
        rotary_percent=rotary_percent,
        use_cpu_initialization=True,
        scaling_factor=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
    )
    expected = rope(SEQ_LEN)
    emb, mscale = yarn(SEQ_LEN)
    rot_dim = int(KV_CHANNELS * rotary_percent)
    print(
        f"rotary_percent={rotary_percent}: rope last dim {expected.shape[-1]}, "
        f"yarn last dim {emb.shape[-1]}, yarn.dim {yarn.dim}, inv_freq_extra {yarn.inv_freq_extra.numel()}, "
        f"mscale {mscale}"
    )
    if expected.shape[-1] != rot_dim:
        failures.append(f"rotary_percent={rotary_percent}: rope last dim {expected.shape[-1]} != {rot_dim}")
        continue
    if emb.shape[-1] != rot_dim:
        failures.append(f"rotary_percent={rotary_percent}: yarn last dim {emb.shape[-1]} != {rot_dim}")
        continue
    if mscale != 1.0:
        failures.append(f"rotary_percent={rotary_percent}: mscale {mscale} != 1.0")
    if not torch.allclose(emb, expected):
        failures.append(f"rotary_percent={rotary_percent}: yarn emb differs from rope emb")
    cos, sin = yarn.get_cached_cos_sin(SEQ_LEN)
    if cos.shape[-1] != rot_dim or sin.shape[-1] != rot_dim:
        failures.append(
            f"rotary_percent={rotary_percent}: cached cos/sin last dims {cos.shape[-1]}/{sin.shape[-1]} != {rot_dim}"
        )
    if not torch.allclose(cos, expected.cos()) or not torch.allclose(sin, expected.sin()):
        failures.append(f"rotary_percent={rotary_percent}: cached cos/sin differ from rope cos/sin")

if failures:
    print("FAILURES:")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("ALL_CHECKS_PASSED")
