from collections import OrderedDict
from typing import Tuple, Union
from detic.modeling.utils import multi_apply
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from detic.modeling.clip.simple_tokenizer import SimpleTokenizer
from .custom import MultiheadSelfAttention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        image_tokens = self.c_proj(
            self.v_proj(x[1:])).permute(1, 2, 0).contiguous().view(N, C, H, W)
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0], image_tokens


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x, image_tokens = self.attnpool(x)

        return x, image_tokens


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, return_tokens: bool, attn_masks=None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        length = x.shape[0]
        if attn_masks is None:
            attn_mask = None \
                if self.attn_mask is None \
                else self.attn_mask[:length, :length]
        else:
            # import pdb; pdb.set_trace()
            attn_mask = attn_masks \
                if self.attn_mask is None \
                else attn_masks + self.attn_mask[None, :length, :length]
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask,
                         return_tokens=return_tokens)[:2]

    def forward(self, x, return_tokens=False, cls_indices=None, attn_masks=None):
        att, tokens = self.attention(self.ln_1(x), return_tokens, attn_masks=attn_masks)
        if return_tokens:
            assert cls_indices is not None
            if not isinstance(cls_indices, int):
                assert len(cls_indices) == x.shape[1]   # x: LNC
            cls_tokens = x[cls_indices, torch.arange(x.shape[1])]
            tokens = cls_tokens[None] + tokens
            tokens = tokens + self.mlp(self.ln_2(tokens))

            x = x + att
            x = x + self.mlp(self.ln_2(x))

            return x, tokens
        else:
            assert tokens is None
            x = x + att
            # x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))

            return x, None


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, return_tokens=False, cls_indices=None, attn_masks=None):
        for i in range(self.layers - 1):
            x, _ = self.resblocks[i](x, attn_masks=attn_masks)
        return self.resblocks[-1](x, return_tokens=return_tokens, cls_indices=cls_indices,
                                  attn_masks=attn_masks)
        # return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.pe_grid_size = input_resolution // patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.num_heads = heads

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def rescale_positional_embedding(self, out_size, dtype):
        rescaled_positional_embedding = \
            self.positional_embedding.new_zeros(1 + out_size ** 2, self.positional_embedding.shape[1])
        rescaled_positional_embedding[0] = self.positional_embedding[0]
        pe_2d = self.positional_embedding[1:].T.contiguous().view(
            1, -1, self.pe_grid_size, self.pe_grid_size)
        pe_2d = F.interpolate(pe_2d, (out_size, out_size), mode='bilinear').view(-1, out_size**2)
        rescaled_positional_embedding[1:] = pe_2d.T.contiguous()

        return rescaled_positional_embedding.to(dtype=dtype)

    def forward(self, x: torch.Tensor, return_tokens, attn_masks=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_size = x.shape[-1]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype)
                       + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                       x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if grid_size == self.pe_grid_size:
            pe = self.positional_embedding.to(x.dtype)
        else:
            pe = self.rescale_positional_embedding(out_size=grid_size, dtype=x.dtype)
        x = x + pe
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, image_tokens = self.transformer(x, return_tokens=return_tokens, cls_indices=0,
                                           attn_masks=attn_masks)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
        if return_tokens:
            image_tokens = image_tokens.permute(1, 0, 2)
            image_tokens = self.ln_post(image_tokens)
            if self.proj is not None:
                image_tokens = image_tokens @ self.proj

            # return the processed image token embeddings
            image_tokens = image_tokens[:, 1:].permute(0, 2, 1).contiguous()
            image_tokens = image_tokens.view(x.shape[0], -1, grid_size, grid_size)
        else:
            assert image_tokens is None

        return x, image_tokens


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 state_file: str,
                 # vision
                 use_image_encoder: bool,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.state_file = state_file
        self.context_length = context_length
        self.use_image_encoder = use_image_encoder
        self.input_resolution = image_resolution
        if use_image_encoder:
            if isinstance(vision_layers, (tuple, list)):
                vision_heads = vision_width * 32 // 64
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_heads,
                    input_resolution=image_resolution,
                    width=vision_width
                )
            else:
                vision_heads = vision_width // 64
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )
        else:
            self.visual = None

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.tokenizer = SimpleTokenizer()
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def init_weights(self):
        print(f'Initiate clip parameters by loading {self.state_file}', flush=True)
        state_dict = torch.jit.load(self.state_file).state_dict()
        self.load_state_dict(state_dict, strict=False)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.visual:
            if isinstance(self.visual, ModifiedResNet):
                if self.visual.attnpool is not None:
                    std = self.visual.attnpool.c_proj.in_features ** -0.5
                    nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                    nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

                for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                    for name, param in resnet_block.named_parameters():
                        if name.endswith("bn3.weight"):
                            nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        # return self.visual.conv1.weight.dtype
        return torch.float16

    def encode_image(self, image, normalize=True, return_image_tokens=False, attn_masks=None):
        assert self.use_image_encoder
        x, image_tokens = self.visual(image.type(self.dtype), return_tokens=return_image_tokens,
                                      attn_masks=attn_masks)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        if return_image_tokens:
            return x, image_tokens
        else:
            assert image_tokens is None
            return x

    def encode_text(self, text, normalize=True, return_word_tokens=False, attn_masks=None, pe_indices=None):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if pe_indices is None:
            x = x + self.positional_embedding.type(self.dtype)[:text.shape[1]]
        else:
            assert x.shape[:2] == pe_indices.shape
            x = x + self.positional_embedding.type(self.dtype)[pe_indices]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, word_tokens = self.transformer(x, return_tokens=return_word_tokens,
                                          cls_indices=text.argmax(dim=-1), attn_masks=attn_masks)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        out = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            out = F.normalize(out, p=2, dim=-1)

        if return_word_tokens:
            word_tokens = word_tokens.permute(1, 0, 2)  # LND -> NLD
            word_tokens = self.ln_final(word_tokens).type(self.dtype)
            word_tokens = word_tokens @ self.text_projection
            word_tokens = [seq[1:end_token_id]
                           for seq, end_token_id in zip(word_tokens, text.argmax(dim=-1))]
            return out, word_tokens
        else:
            assert word_tokens is None
            return out

    def encode_intermediate_k(self, x, end_token_ids, stepk=2, normalize=True,
                              return_word_tokens=False, attn_masks=None, **kwargs):

        num_steps = len(self.transformer.resblocks)
        for i in range(stepk, num_steps - 1):
            x, _ = self.transformer.resblocks[i](x, attn_masks=attn_masks)
        x, word_tokens = self.transformer.resblocks[num_steps-1](x, return_tokens=return_word_tokens,
                                                                 cls_indices=end_token_ids,
                                                                 attn_masks=attn_masks)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        out = x[torch.arange(x.shape[0]), end_token_ids] @ self.text_projection

        if normalize:
            out = F.normalize(out, dim=-1, p=2)
        if return_word_tokens:
            word_tokens = word_tokens.permute(1, 0, 2)  # LND -> NLD
            word_tokens = self.ln_final(word_tokens).type(self.dtype)
            word_tokens = word_tokens @ self.text_projection
            word_tokens = [seq[1:end_token_id]
                           for seq, end_token_id in zip(word_tokens, end_token_ids)]
            return out, word_tokens
        else:
            assert word_tokens is None
            return out

    def encode_text_endk(self, text, stepk=2, normalize=True, attn_masks=None, pe_indices=None, **kwargs):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if pe_indices is None:
            x = x + self.positional_embedding.type(self.dtype)[:text.shape[1]]
        else:
            assert x.shape[:2] == pe_indices.shape
            x = x + self.positional_embedding.type(self.dtype)[pe_indices]
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(stepk):
            x, _ = self.transformer.resblocks[i](x, attn_masks=attn_masks)

        out = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        out = out[torch.arange(out.shape[0]), text.argmax(dim=-1)]  # @ self.text_projection

        if normalize:
            out = F.normalize(out, dim=-1, p=2)

        return out, x, text.argmax(dim=-1)

    def prepare_pseudo_text(self, pseudo_tokens, context_length,
                            prefixes=[], suffixes=[]):
        device = pseudo_tokens[0].device
        sot_token = self.token_embedding(torch.tensor([self.sot_token],
                                                      device=device)).type(self.dtype)  # [batch_size, n_ctx, d_model]
        eot_token = self.token_embedding(torch.tensor([self.eot_token],
                                                      device=device)).type(self.dtype)
        empty_token = self.token_embedding(torch.tensor([0],
                                                        device=device)).type(self.dtype)

        pseudo_tokens = [torch.cat([sot_token, tokens, eot_token], dim=0) for tokens in pseudo_tokens]

        def _pad_sequence(tokens):
            if tokens.shape[0] > context_length:
                x = tokens[list(range(context_length - 1)) + [tokens.shape[0] - 1]]
                end_token_id = context_length - 1
            else:
                x = torch.cat([tokens, empty_token.repeat(
                    context_length - tokens.shape[0], 1)], dim=0)
                end_token_id = tokens.shape[0] - 1
            return x, end_token_id

        x, end_token_ids = multi_apply(_pad_sequence, pseudo_tokens)
        x = torch.stack(x, dim=0)

        return x, torch.tensor(end_token_ids, dtype=torch.long, device=x.device)

    def prepare_pseudo_text_with_mask(self, pseudo_tokens,
                                      word_masks, context_length):
        device = pseudo_tokens[0].device
        sot_token = self.token_embedding(torch.tensor([self.sot_token],
                                                      device=device)).type(self.dtype)  # [batch_size, n_ctx, d_model]
        eot_token = self.token_embedding(torch.tensor([self.eot_token],
                                                      device=device)).type(self.dtype)
        empty_token = self.token_embedding(torch.tensor([0],
                                                        device=device)).type(self.dtype)   # 1 x d_model

        # pseudo_tokens = [torch.cat([sot_token, tokens, eot_token], dim=0) for tokens in pseudo_tokens]

        def _pad_sequence(tokens, masks):
            tokens = torch.cat([sot_token, tokens, eot_token], dim=0)
            ones_padding = torch.ones_like(masks[:1])
            masks = torch.cat([ones_padding, masks, ones_padding])
            assert masks.shape[0] == tokens.shape[0]
            assert context_length >= tokens.shape[0]    # TODO more options
            if tokens.shape[0] > context_length:
                x = tokens[list(range(context_length - 1)) + [tokens.shape[0] - 1]]
                m = masks[list(range(context_length - 1)) + [tokens.shape[0] - 1]]
                end_token_id = context_length - 1
            else:
                x = torch.cat([tokens, empty_token.repeat(
                    context_length - tokens.shape[0], 1)], dim=0)
                m = torch.cat([masks, ones_padding.repeat(
                    context_length - tokens.shape[0])], dim=0)
                end_token_id = tokens.shape[0] - 1
            return x, m, end_token_id

        x, m, end_token_ids = multi_apply(_pad_sequence, pseudo_tokens, word_masks)
        x = torch.stack(x, dim=0)
        m = torch.stack(m, dim=0)
        assert m.shape[0] == x.shape[0]
        assert m.shape[1] == context_length
        m = m[:, None] * m[..., None]
        attn_masks = torch.where(m > 0.0, 0.0, float('-inf'))
        attn_masks[:, range(context_length), range(context_length)] = 0.0
        attn_masks = attn_masks[:, None].repeat(1, self.transformer.heads, 1, 1)

        return x, attn_masks.flatten(0, 1), \
               torch.tensor(end_token_ids, dtype=torch.long, device=x.device)

    def prepare_pseudo_text_tensor(self, pseudo_tokens):
        device = pseudo_tokens.device
        num_preds, num_words, word_dim = pseudo_tokens.shape
        sot_token = self.token_embedding(torch.tensor([self.sot_token],
                                                      device=device)).type(self.dtype)
        eot_token = self.token_embedding(torch.tensor([self.eot_token],
                                                      device=device)).type(self.dtype)
        sot_token = sot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)
        eot_token = eot_token.view(1, 1, word_dim).repeat(num_preds, 1, 1)

        pseudo_tokens = torch.cat([sot_token, pseudo_tokens, eot_token], dim=1)
        end_token_ids = torch.tensor([num_words + 1] * num_preds,
                                     dtype=torch.long, device=device)

        return pseudo_tokens, end_token_ids

    def encode_pseudo_text_endk(self, x, end_token_ids, text_pe=True,
                                stepk=2, normalize=True, attn_masks=None, pe_indices=None):
        if pe_indices is None:
            x = x + self.positional_embedding.type(self.dtype)[:x.shape[1]]
        else:
            assert x.shape[:2] == pe_indices.shape
            x = x + self.positional_embedding.type(self.dtype)[pe_indices]
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(stepk):
            x, _ = self.transformer.resblocks[i](x, attn_masks=attn_masks)

        out = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        out = out[torch.arange(out.shape[0]), end_token_ids]  # @ self.text_projection

        if normalize:
            out = F.normalize(out, dim=-1, p=2)

        return out, x, end_token_ids

    def encode_pseudo_text(self, x, end_token_ids, text_pe=True, normalize=True,
                           return_word_tokens=False, attn_masks=None, pe_indices=None):
        if pe_indices is None:
            x = x + self.positional_embedding.type(self.dtype)[:x.shape[1]]
        else:
            assert x.shape[:2] == pe_indices.shape
            x = x + self.positional_embedding.type(self.dtype)[pe_indices]
        x = x.permute(1, 0, 2)  # NLD -> LND
        return self.encode_intermediate_k(x, end_token_ids,
                                          stepk=0, normalize=normalize,
                                          return_word_tokens=return_word_tokens, attn_masks=attn_masks)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict, state_file, use_image_encoder, **kwargs):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim,
        state_file=state_file,
        # vision
        use_image_encoder=use_image_encoder,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        # text
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    # model.load_state_dict(state_dict)
    return model.eval()
