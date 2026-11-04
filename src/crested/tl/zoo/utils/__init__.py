"""Model blocks, utility functions and custom layers for model building."""

# All functions and classes exported in `crested.tl.zoo.utils` are used as fallback custom objects in `crested.utils.load_model`

from ._alphagenome import (
    Bias,
    MultiQueryAttention,
    RowAttention,
    Scale,
    SequenceToPair,
    StandardizedConv1D,
    apply_rope,
    attention_bias_block_ag,
    conv_block_ag,
    down_res_block_ag,
    mha_block_ag,
    mlp_block_ag,
    pair_update_block_ag,
    rms_norm_ag,
    up_res_block_ag,
)
from ._attention import (
    AttentionPool1D,
    MultiheadAttention,
)
from ._hydra import (
    Hydra,
    hydra_block,
    ssd_scan,
)
from ._layers import (
    DropPath,
    RMSBatchNormalization,
    activate,
    conv_block,
    conv_block_bs,
    dense_block,
    dilated_residual,
    ffn_block_enf,
    gelu_approx,
    gelu_enf,
    get_output,
    legnet_eff_block,
    legnet_se_block,
    mha_block_enf,
    normalize,
    pool,
)
