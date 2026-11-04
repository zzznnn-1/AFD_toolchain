from pnm.models.qwen38.loader import correct_skip_prefix_collisions
from transformers.quantizers.quantizers_utils import should_convert_module


def test_official_gate_prefix_does_not_drop_fp8_scale():
    config={'quantization_config':{'modules_to_not_convert':['model.language_model.layers.3.mlp.gate','lm_head']}}
    weights={'model.language_model.layers.3.mlp.gate_proj.weight_scale_inv':'weights.safetensors'}
    original=list(config['quantization_config']['modules_to_not_convert'])
    assert not should_convert_module('model.language_model.layers.3.mlp.gate_proj',original)
    corrected,collisions=correct_skip_prefix_collisions(config,weights)
    assert should_convert_module('model.language_model.layers.3.mlp.gate_proj',corrected)
    assert not should_convert_module('model.language_model.layers.3.mlp.gate',corrected)
    assert not should_convert_module('lm_head',corrected)
    assert config['quantization_config']['modules_to_not_convert']==original
    assert len(collisions)==1
