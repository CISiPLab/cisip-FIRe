import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPConfig
from transformers.models.vit.modeling_vit import ViTLayer, ViTConfig, ViTOutput


class Adapter(nn.Module):
    def __init__(
            self,
            in_dim,
            bottleneck_dim,
            dropout=0.0,
            adapter_scalar="learnable_scalar",
            adapter_layernorm_option="in",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm = nn.LayerNorm(self.in_dim)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.in_dim, self.bottleneck_dim)
        self.non_linear_func = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.in_dim)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm(up)

        output = up
        return output


class CLIPEncoderLayerWithAdapter(CLIPEncoderLayer):
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.adapt_mlp_1 = None
        self.adapt_mlp_2 = None

    def get_adapt_params(self):
        return list(nn.ModuleDict({
            'adapt_mlp_1': self.adapt_mlp_1,
            'adapt_mlp_2': self.adapt_mlp_2
        }).named_parameters())

    def setup_adapt_mlp(self, bottleneck_dim, dropout):
        self.adapt_mlp_1 = Adapter(
            self.embed_dim,
            bottleneck_dim,
            dropout,
        )
        self.adapt_mlp_2 = Adapter(
            self.embed_dim,
            bottleneck_dim,
            dropout,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            causal_attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        adapted_states = self.adapt_mlp_1(hidden_states)

        hidden_states = residual + hidden_states + adapted_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        adapted_states = self.adapt_mlp_2(hidden_states)

        hidden_states = residual + hidden_states + adapted_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VitOutputWithAdapter(ViTOutput):
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, adapter_layer=None) -> torch.Tensor:
        assert adapter_layer is not None

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        adapted_states = adapter_layer(hidden_states)

        hidden_states = hidden_states + input_tensor + adapted_states

        return hidden_states


class ViTLayerWithAdapter(ViTLayer):
    def __init__(self, config: ViTConfig):
        super().__init__(config)

        self.config = config
        self.output = VitOutputWithAdapter(config)
        self.adapt_mlp_1 = None
        self.adapt_mlp_2 = None

    def get_adapt_params(self):
        return list(nn.ModuleDict({
            'adapt_mlp_1': self.adapt_mlp_1,
            'adapt_mlp_2': self.adapt_mlp_2
        }).named_parameters())

    def setup_adapt_mlp(self, bottleneck_dim, dropout):
        self.adapt_mlp_1 = Adapter(
            self.config.hidden_size,
            bottleneck_dim,
            dropout,
        )
        self.adapt_mlp_2 = Adapter(
            self.config.hidden_size,
            bottleneck_dim,
            dropout,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        adapted_states = self.adapt_mlp_1(attention_output)

        # first residual connection
        hidden_states = attention_output + hidden_states + adapted_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states, self.adapt_mlp_2)

        outputs = (layer_output,) + outputs

        return outputs
#
#
# def clip_add_adapter(vision_model, adapter_bottleneck_dim):
#     vision_model = self.backbone
#     current_layers = vision_model.encoder.layers
#
#     clip_encoder_cfg = vision_model.encoder.config
#     new_layers = nn.ModuleList([CLIPEncoderLayerWithAdapter(clip_encoder_cfg) for _ in current_layers])
#     for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: CLIPEncoderLayerWithAdapter
#         nlayer.load_state_dict(clayer.state_dict())
#         nlayer.setup_adapt_mlp(adapter_bottleneck_dim, 0.0)
#         nlayer_named_params = nlayer.get_adapt_params()
#         for pname, param in nlayer_named_params:
#             self.trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
#     vision_model.encoder.layers = new_layers
#     hidden_size = vision_model.config.hidden_size
#
