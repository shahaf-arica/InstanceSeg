from modeling.dino.vision_transformer import VisionTransformer
import torch
import torch.nn as nn
from functools import partial


def mae_enc_vit_base(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_enc_vitb16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = mae_enc_vit_base(num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth",
            map_location="cpu",
        )
        state_dict = state_dict['model']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('decoder') or k.startswith('mask_token'):
                del state_dict[k]
        model.load_state_dict(state_dict, strict=True)
    return model


def mae_dec_vitl16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vits.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=True)
    return model
