import torch


class ViTFeatureExtractor:
    def __init__(self, model, blocks_to_extract=None):
        self.b = 0
        self.out = {}
        # self.which_model = which_model
        # if which_model == 'dino':
        #     blocks = model.blocks
        # elif which_model == 'mae_decoder':
        #     blocks = model.decoder_blocks
        # elif which_model == 'mae_encoder':
        #     blocks = model.encoder_blocks
        # else:
        #     raise ValueError('Unknown model')
        blocks = model.blocks
        if blocks_to_extract is not None:
            blocks = [blocks[b] for b in blocks_to_extract]
            self.block_names = ["block " + str(b) for b in blocks_to_extract]
        else:
            self.block_names = ["block " + str(b) for b in range(len(blocks))]
        self.blocks = blocks
        self.heads_num = blocks[-1].attn.num_heads
        self.att_scales = { i: blocks[i].attn.scale for i in range(len(blocks))}

        for block in blocks:
            block.attn.qkv.register_forward_hook(self.hook)
        self.model = model

    def reset(self):
        self.out = {}
        self.b = 0

    def hook(self, module, input, output):
        self.out[self.b] = output.detach().cpu()
        self.b += 1

    def _get_attention_score(self, q, k, scale=1.0, mask_col=None, remove_cls=False):
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask_col is not None:
            attn[:, :, :, ~mask_col] = -1e10
        attn = attn.softmax(dim=-1)
        # sum all heads
        attn = torch.sum(attn, dim=1)
        if remove_cls:
            # remove the cls token
            attn = attn[0, 1:, 1:]
        return attn[0]

    def _extract_qkv(self, qkv_out, batch_size, token_num):
        qkv = (
            qkv_out
            .reshape(batch_size, token_num + 1, 3, self.heads_num, -1 // self.heads_num)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def _model_forward(self, img, patch_mask):
        with torch.no_grad():
            # if self.which_model == 'dino' or self.which_model == 'mae_encoder':
            #     last_self_att = self.model.get_last_self_attention(img[None, :, :, :], patch_mask)
            #     batch_size = last_self_att.shape[0]
            #     token_num = last_self_att.shape[2] - 1
            # elif self.which_model == 'mae_decoder':
            #     y = self.model.inference(img[None, :, :, :].float(), patch_mask_encoder=patch_mask, patch_mask_decoder=patch_mask)
            #     batch_size = y.shape[0]
            #     token_num = y.shape[1]
            # else:
            #     raise ValueError('Unknown model')
            last_self_att = self.model.get_last_self_attention(img[None, :, :, :], patch_mask)
            batch_size = last_self_att.shape[0]
            token_num = last_self_att.shape[2] - 1
        return token_num, batch_size

    def get_features(self, img, patch_mask):
        token_num, batch_size = self._model_forward(img, patch_mask)

        features = {}

        for b, qkv_out in self.out.items():
            block_name = self.block_names[b]
            q, k, v = self._extract_qkv(qkv_out, batch_size, token_num)
            attn = self._get_attention_score(q, k, scale=self.att_scales[b])
            q = q.transpose(1, 2).reshape(batch_size, token_num + 1, -1)
            k = k.transpose(1, 2).reshape(batch_size, token_num + 1, -1)
            v = v.transpose(1, 2).reshape(batch_size, token_num + 1, -1)
            features[block_name] = {'q': q, 'k': k, 'v': v, 'attn': attn}

        return features



