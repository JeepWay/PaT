from typing import Dict, List, Optional, Tuple, Type, Union, Any

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F
import math

class CustomMaxPool(nn.Module):
    def __init__(self, dim=1):
        super(CustomMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.max(dim=self.dim)[0]
    
class CustomMeanPool(nn.Module):
    def __init__(self, dim=1):
        super(CustomMeanPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return th.mean(x, dim=self.dim)   

class SingleHeadAttention(nn.Module):
    def __init__(self, 
        in_embed_dim:int, 
        out_embed_dim:int, 
        seq_len:int, 
        bias:bool = True, 
        normalize:bool = False
    ):
        super(SingleHeadAttention, self).__init__()
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.seq_len = seq_len
        self.normalize = normalize
        self.q_proj = nn.Linear(in_embed_dim, out_embed_dim, bias=bias)    # (batch_size, seq_len, out_embed_dim)
        self.k_proj = nn.Linear(in_embed_dim, out_embed_dim, bias=bias)    # (batch_size, seq_len, out_embed_dim)
        self.v_proj = nn.Linear(in_embed_dim, out_embed_dim, bias=bias)    # (batch_size, seq_len, out_embed_dim)
        if self.normalize:
            self.q_norm = nn.LayerNorm((seq_len, out_embed_dim), elementwise_affine=True)
            self.k_norm = nn.LayerNorm((seq_len, out_embed_dim), elementwise_affine=True)
            self.v_norm = nn.LayerNorm((seq_len, out_embed_dim), elementwise_affine=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = th.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.out_embed_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = th.softmax(attn_scores, dim=-1)     # (batch_size, seq_length, seq_length)
        output = th.matmul(attn_probs, V)                # (batch_size, seq_length, out_embed_dim)
        return output

    def forward(self, Q, K, V, mask=None):
        Q = self.q_proj(Q)  # (batch_size, seq_len, out_embed_dim)
        K = self.k_proj(K)  # (batch_size, seq_len, out_embed_dim)
        V = self.v_proj(V)  # (batch_size, seq_len, out_embed_dim)
        if self.normalize:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
            V = self.v_norm(V)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, seq_len, out_embed_dim)
        return attn_output
    

class MultiheadAttention(nn.Module):
    def __init__(self, 
            in_embed_dim:int, 
            out_embed_dim:int, 
            seq_len:int,
            num_heads:int = 4, 
            dropout:float = 0.0, 
            bias:bool = True, 
            normalize:bool = False
        ):
        super(MultiheadAttention, self).__init__()
        assert num_heads >= 1, "num_heads must be greater than or equal to 1"
        assert in_embed_dim % num_heads == 0, "in_embed_dim must be divisible by num_heads"
        self.in_embed_dim = in_embed_dim
        self.out_embed_dim = out_embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.one_head_dim = in_embed_dim // num_heads
        self.normalize = normalize
        self.q_proj = nn.Linear(in_embed_dim, in_embed_dim, bias=bias)
        self.k_proj = nn.Linear(in_embed_dim, in_embed_dim, bias=bias)
        self.v_proj = nn.Linear(in_embed_dim, in_embed_dim, bias=bias)
        if self.normalize:
            self.q_norm = nn.LayerNorm((num_heads, seq_len, self.one_head_dim), elementwise_affine=True)
            self.k_norm = nn.LayerNorm((num_heads, seq_len, self.one_head_dim), elementwise_affine=True)
            self.v_norm = nn.LayerNorm((num_heads, seq_len, self.one_head_dim), elementwise_affine=True)

        self.out_proj = nn.Linear(in_embed_dim, out_embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = th.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.in_embed_dim)    # (batch_size, num_heads, seq_length, seq_length)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = th.softmax(attn_scores, dim=-1)
        output = th.matmul(attn_probs, V)    # (batch_size, num_heads, seq_length, one_head_dim)
        return output
    
    def split_heads(self, x):
        '''
        Reshape the input to have num_heads for multi-head attention.This method reshapes the 
        input x into the shape (batch_size, num_heads, seq_length, one_head_dim). It enables the model 
        to process multiple attention heads concurrently, allowing for parallel computation.
        '''
        batch_size, seq_length, embed_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.one_head_dim).transpose(1, 2)     # (batch_size, num_heads, seq_length, one_head_dim)
        
    def combine_heads(self, x):
        '''
        Combine the multiple heads back to original shape. After applying attention to each 
        head separately, this method combines the results back into a single tensor of shape 
        (batch_size, seq_length, d_model). This prepares the result for further processing.
        '''
        batch_size, num_heads, seq_length, embed_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.in_embed_dim)    # (batch_size, seq_length, in_embed_dim)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.q_proj(Q))   # (batch_size, num_heads, seq_length, one_head_dim)
        K = self.split_heads(self.k_proj(K))   # (batch_size, num_heads, seq_length, one_head_dim)
        V = self.split_heads(self.k_proj(V))   # (batch_size, num_heads, seq_length, one_head_dim)
        if self.normalize:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
            V = self.v_norm(V)

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)  # (batch_size, num_heads, seq_length, one_head_dim)

        # Combine heads and apply output transformation
        output = self.out_proj(self.combine_heads(attn_output))  # (batch_size, seq_length, out_embed_dim)
        return output
    

class ImplicitPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1600):
        super(ImplicitPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.encoding = th.zeros(max_len, embed_dim)                    # torch.Size([max_len, embed_dim])
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)   # torch.Size([max_len, 1])
        div_term = th.exp(th.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))    # torch.Size([embed_dim//2])
        self.encoding[:, 0::2] = th.sin(position * div_term)            # torch.Size([max_len, embed_dim//2])
        self.encoding[:, 1::2] = th.cos(position * div_term)            # torch.Size([max_len, embed_dim//2])
        self.encoding = self.encoding.unsqueeze(0)                      # torch.Size([1, max_len, embed_dim])

    def forward(self, x):
        seq_len, embed_dim = x.shape[1], x.shape[2]
        assert embed_dim == self.embed_dim, "input feature's embed_dim must be equal to the module's embed_dim"
        assert seq_len <= self.max_len, "seq_len must be less than max_len"
        return x + self.encoding[:seq_len, :].to(x.device)              # torch.Size([N, seq_len, embed_dim])


class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 100,
        normalize_images: bool = False,
        position_encode: bool = False,
        cnn_shortcut: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "BaseNetwork must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__()
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.normalize_images = normalize_images
        self.position_encode = position_encode
        self.cnn_shortcut = cnn_shortcut
        self.share_input_channels = observation_space.shape[0]

        # to be defined in the subclasses 
        self.share_extractor: nn.Sequential = None
        self.attention: SingleHeadAttention = None
        self.positional_encoding: ImplicitPositionalEncoding = None
        self.mask_net: nn.Sequential = None
        self.actor_net: nn.Sequential = None
        self.critic_net: nn.Sequential = None

        self.mask_n_flatten: int = None
        self.mask_extractor: nn.Sequential = None
        self.actor_n_flatten: int = None
        self.actor_extractor: nn.Sequential = None
        self.critic_n_flatten: int = None
        self.critic_extractor: nn.Sequential = None
        
        self.l1: nn.Linear = None
        self.l2: nn.Linear = None
        self.transformer_encoder: nn.TransformerEncoder = None
        self.transformer: nn.Transformer = None

    def forward(self, observations: th.Tensor, 
                interest_mask: th.Tensor = None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        cnn_f = self.share_extractor(observations)  # [N, share_out_channels, cW, cH]
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, share_out_channels])

        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # torch.Size([N, cW*cH, share_out_channels])

        attn_output = self.attention(cnn_f, cnn_f, cnn_f)
        if self.cnn_shortcut is True:
            mask_probs = self.mask_net(attn_output + cnn_f)               # torch.Size([N, action_dim])
            action_logits = self.actor_net(attn_output + cnn_f)           # torch.Size([N, action_dim])
            values = self.critic_net(attn_output + cnn_f)                 # torch.Size([N, 1])
        else:
            mask_probs = self.mask_net(attn_output)               # torch.Size([N, action_dim])
            action_logits = self.actor_net(attn_output)           # torch.Size([N, action_dim])
            values = self.critic_net(attn_output)                 # torch.Size([N, 1])
        return mask_probs, action_logits, values

    def forward_mask_probs(self, 
        observations: th.Tensor, 
        interest_mask: th.Tensor = None
    ) -> th.Tensor:
        mask_probs, _, _ = self.forward(observations, interest_mask)
        return mask_probs

    def forward_action_logits(self, 
        observations: th.Tensor,
        interest_mask: th.Tensor = None
    ) -> th.Tensor:
        _, action_logits, _ = self.forward(observations, interest_mask)
        return action_logits
    
    def forward_critic(self, 
        observations: th.Tensor,
        interest_mask: th.Tensor = None
    ) -> th.Tensor:
        _, _, values = self.forward(observations, interest_mask)
        return values
    
    def _get_n_flatten(self, share_extractor: nn.Sequential, in_channels: int, out_channels: int) -> int:
        # Compute shape after flattening by doing one forward pass
        with th.no_grad():
            obs_tenosr = th.as_tensor(self.observation_space.sample()[None]).float()
            tmp_layer = nn.Sequential(
                (nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            n_flatten = tmp_layer(share_extractor(obs_tenosr)).shape[1]
            del tmp_layer, obs_tenosr
        return n_flatten


class TransfromerNetwork1(BaseNetwork):
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        normalize_images: bool = False, 
        hidden_dim: int = 100,
        position_encode: bool = True, 
        num_layers: int = 2,
        use_pad_mask: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
        )
        self.use_pad_mask = use_pad_mask
        self.l1 = nn.Linear(observation_space.shape[0], transformer_kwargs["d_model"])
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=transformer_kwargs["d_model"], max_len=self.action_dim)
        layer = nn.TransformerEncoderLayer(**transformer_kwargs)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.mask_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None):
        # src_key_padding_mask = (interest_mask == 0)
        x = observations.flatten(2).transpose(1, 2)    # torch.Size([N, 3, W, H]) -> torch.Size([N, W*H, 3])
        x = self.l1(x)
        x = F.relu(x)

        if self.position_encode is True:
            x = self.positional_encoding(x)         # torch.Size([N, W*H, d_model])

        if interest_mask is not None:
            src_key_padding_mask = (interest_mask == 0)
            # handle the case where all the elements in the mask are zeros, which will make src_key_padding_mask 
            # all True, leading to NaN output
            all_zeros = (interest_mask.sum(dim=-1) == 0)
            src_key_padding_mask = src_key_padding_mask & ~all_zeros.unsqueeze(-1)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask if self.use_pad_mask else None)
        else:
            x = self.transformer_encoder(x)         # torch.Size([N, W*H, d_model])

        x = x.mean(dim=1)                           # torch.Size([N, d_model])
        mask_probs = self.mask_net(x)               # torch.Size([N, action_dim])
        action_logits = self.actor_net(x)           # torch.Size([N, action_dim])
        values = self.critic_net(x)                 # torch.Size([N, 1])
        return mask_probs, action_logits, values


class TransfromerNetwork2(BaseNetwork):
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        normalize_images: bool = False, 
        hidden_dim: int = 100,
        position_encode: bool = True, 
        num_layers: int = 2,
        use_pad_mask: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
        )
        self.use_pad_mask = use_pad_mask
        self.l1 = nn.Linear(observation_space.shape[0], transformer_kwargs["d_model"])
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=transformer_kwargs["d_model"], max_len=self.action_dim)
        layer = nn.TransformerEncoderLayer(**transformer_kwargs)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.mask_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.action_dim),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None):
        # src_key_padding_mask = (interest_mask == 0)
        x = observations.flatten(2).transpose(1, 2)    # torch.Size([N, 3, W, H]) -> torch.Size([N, W*H, 3])
        x = self.l1(x)
        x = F.relu(x)

        if self.position_encode is True:
            x = self.positional_encoding(x)         # torch.Size([N, W*H, d_model])

        if interest_mask is not None:
            src_key_padding_mask = (interest_mask == 0)
            # handle the case where all the elements in the mask are zeros, which will make src_key_padding_mask 
            # all True, leading to NaN output
            all_zeros = (interest_mask.sum(dim=-1) == 0)
            src_key_padding_mask = src_key_padding_mask & ~all_zeros.unsqueeze(-1)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask if self.use_pad_mask else None)
        else:
            x = self.transformer_encoder(x)         # torch.Size([N, W*H, d_model])

        x = x.mean(dim=1)                           # torch.Size([N, d_model])
        # x = F.relu(x)
        mask_probs = self.mask_net(x)               # torch.Size([N, action_dim])
        action_logits = self.actor_net(x)           # torch.Size([N, action_dim])
        values = self.critic_net(x)                 # torch.Size([N, 1])
        return mask_probs, action_logits, values
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        normalize_images: bool = False, 
        hidden_dim: int = 100,
        position_encode: bool = True, 
        num_layers: int = 2,
        use_pad_mask: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
        )
        self.use_pad_mask = use_pad_mask
        self.l1 = nn.Linear(observation_space.shape[0], transformer_kwargs["d_model"])
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=transformer_kwargs["d_model"], max_len=self.action_dim)
        layer = nn.TransformerEncoderLayer(**transformer_kwargs)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.mask_net = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None):
        # src_key_padding_mask = (interest_mask == 0)
        x = observations.flatten(2).transpose(1, 2)    # torch.Size([N, 3, W, H]) -> torch.Size([N, W*H, 3])
        x = self.l1(x)
        x = F.relu(x)

        if self.position_encode is True:
            x = self.positional_encoding(x)         # torch.Size([N, W*H, d_model])

        if interest_mask is not None:
            src_key_padding_mask = (interest_mask == 0)
            # handle the case where all the elements in the mask are zeros, which will make src_key_padding_mask 
            # all True, leading to NaN output
            all_zeros = (interest_mask.sum(dim=-1) == 0)
            src_key_padding_mask = src_key_padding_mask & ~all_zeros.unsqueeze(-1)
            x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask if self.use_pad_mask else None)
        else:
            x = self.transformer_encoder(x)         # torch.Size([N, W*H, d_model])

        x = x.mean(dim=-1)                          # torch.Size([N, W*H])
        mask_probs = self.mask_net(x)               # torch.Size([N, action_dim])
        action_logits = self.actor_net(x)           # torch.Size([N, action_dim])
        values = self.critic_net(x)                 # torch.Size([N, 1])
        return mask_probs, action_logits, values
    
    
class TransfromerNetwork3(BaseNetwork):
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        normalize_images: bool = False, 
        hidden_dim: int = 100,
        position_encode: bool = True,
        use_pad_mask: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
        )
        self.use_pad_mask = use_pad_mask
        self.l1 = nn.Linear(observation_space.shape[0], transformer_kwargs["d_model"])
        self.l2 = nn.Linear(1, transformer_kwargs["d_model"])
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=transformer_kwargs["d_model"], max_len=self.action_dim)
        self.transformer = nn.Transformer(**transformer_kwargs)

        self.mask_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(transformer_kwargs["d_model"], self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor):
        assert(interest_mask.shape[1] == self.action_dim, "interest_mask's shape[1] must be equal to action_dim")
        x = observations.flatten(2).transpose(1, 2) # torch.Size([N, 3, W, H]) -> torch.Size([N, W*H, 3])
        x = self.l1(x)
        x = F.relu(x)
        y = interest_mask.unsqueeze(-1)             # torch.Size([N, W*H, 1])
        y = self.l2(y)
        y = F.relu(y)

        if self.position_encode is True:
            x = self.positional_encoding(x)         # torch.Size([N, W*H, d_model])
            y = self.positional_encoding(y)         # torch.Size([N, W*H, d_model])

        src_key_padding_mask = (interest_mask == 0)
        # handle the case where all the elements in the mask are zeros, which will make src_key_padding_mask 
        # all True, leading to NaN output
        all_zeros = (interest_mask.sum(dim=-1) == 0)
        src_key_padding_mask = src_key_padding_mask & ~all_zeros.unsqueeze(-1)
        output = self.transformer(
            x, 
            y, 
            src_key_padding_mask=src_key_padding_mask if self.use_pad_mask else None
        )   # torch.Size([N, W*H, d_model])
         
        output = output.mean(dim=1)                 # torch.Size([N, d_model])
        mask_probs = self.mask_net(output)          # torch.Size([N, action_dim])
        action_logits = self.actor_net(output)      # torch.Size([N, action_dim])
        values = self.critic_net(output)            # torch.Size([N, 1])
        return mask_probs, action_logits, values
    

class TransfromerNetwork4(BaseNetwork):
    def __init__(
        self, 
        observation_space, 
        action_dim, 
        normalize_images: bool = False, 
        hidden_dim: int = 100,
        position_encode: bool = True,
        use_pad_mask: bool = False,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
        )
        self.use_pad_mask = use_pad_mask
        self.l1 = nn.Linear(observation_space.shape[0], transformer_kwargs["d_model"])
        self.l2 = nn.Linear(1, transformer_kwargs["d_model"])
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=transformer_kwargs["d_model"], max_len=self.action_dim)
        self.transformer = nn.Transformer(**transformer_kwargs)

        self.mask_net = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(self.action_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(self.action_dim, 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor):
        assert(interest_mask.shape[1] == self.action_dim, "interest_mask's shape[1] must be equal to action_dim")
        x = observations.flatten(2).transpose(1, 2) # torch.Size([N, 3, W, H]) -> torch.Size([N, W*H, 3])
        x = self.l1(x)
        x = F.relu(x)
        y = interest_mask.unsqueeze(-1)             # torch.Size([N, W*H, 1])
        y = self.l2(y)
        y = F.relu(y)

        if self.position_encode is True:
            x = self.positional_encoding(x)         # torch.Size([N, W*H, d_model])
            y = self.positional_encoding(y)         # torch.Size([N, W*H, d_model])

        src_key_padding_mask = (interest_mask == 0)
        # handle the case where all the elements in the mask are zeros, which will make src_key_padding_mask 
        # all True, leading to NaN output
        all_zeros = (interest_mask.sum(dim=-1) == 0)
        src_key_padding_mask = src_key_padding_mask & ~all_zeros.unsqueeze(-1)
        output = self.transformer(
            x, 
            y, 
            src_key_padding_mask=src_key_padding_mask if self.use_pad_mask else None
        )   # torch.Size([N, W*H, d_model])
         
        output = output.mean(dim=-1)                # torch.Size([N, action_dim])
        output = F.tanh(output)
        mask_probs = self.mask_net(output)          # torch.Size([N, action_dim])
        action_logits = self.actor_net(output)      # torch.Size([N, action_dim])
        values = self.critic_net(output)            # torch.Size([N, 1])
        return mask_probs, action_logits, values   
   
    
class CnnAttenMlpNetwork1_v1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        position_encode: bool = True,
        cnn_shortcut: bool = False,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = SingleHeadAttention(in_embed_dim=share_out_channels, seq_len=action_dim, **attention_kwargs)
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=share_out_channels, max_len=self.action_dim)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMeanPool(dim=1),
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )


class CnnAttenMlpNetwork1_v2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        position_encode: bool = True,
        cnn_shortcut: bool = False,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = MultiheadAttention(in_embed_dim=share_out_channels, seq_len=action_dim, **attention_kwargs)
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=share_out_channels, max_len=self.action_dim)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMeanPool(dim=1),
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )


class CnnAttenMlpNetwork1_v3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        position_encode: bool = True,
        cnn_shortcut: bool = False,
        share_out_channels: int = 64,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
            position_encode=position_encode,
            cnn_shortcut=cnn_shortcut,
        )

        self.share_out_channels = share_out_channels
        self.attention = SingleHeadAttention(in_embed_dim=share_out_channels, seq_len=action_dim, **attention_kwargs)
        self.positional_encoding = ImplicitPositionalEncoding(embed_dim=share_out_channels, max_len=self.action_dim)
        self.layer_norm = nn.LayerNorm((action_dim, attention_kwargs["out_embed_dim"]), elementwise_affine=True)

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid(),
        )

        self.actor_net = nn.Sequential(
            CustomMeanPool(dim=1), 
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        self.critic_net = nn.Sequential(
            CustomMeanPool(dim=1),
            nn.Linear(self.attention.out_embed_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim//2, 1),
        )
        
    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        cnn_f = self.share_extractor(observations)  # [N, share_out_channels, cW, cH]
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, share_out_channels])

        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # torch.Size([N, cW*cH, share_out_channels])

        attn_output = self.attention(cnn_f, cnn_f, cnn_f)
        if self.cnn_shortcut is True:
            mask_probs = self.mask_net(self.layer_norm(attn_output + cnn_f))               # torch.Size([N, action_dim])
            action_logits = self.actor_net(self.layer_norm(attn_output + cnn_f))           # torch.Size([N, action_dim])
            values = self.critic_net(self.layer_norm(attn_output + cnn_f))                 # torch.Size([N, 1])
        else:
            mask_probs = self.mask_net(self.layer_norm(attn_output))               # torch.Size([N, action_dim])
            action_logits = self.actor_net(self.layer_norm(attn_output))           # torch.Size([N, action_dim])
            values = self.critic_net(self.layer_norm(attn_output))                 # torch.Size([N, 1])
        return mask_probs, action_logits, values

    def forward_mask_probs(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        
        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # torch.Size([N, cW*cH, share_out_channels])
            
        attn_output = self.attention(cnn_f, cnn_f, cnn_f)     
        if self.cnn_shortcut is True:
            mask_probs = self.mask_net(attn_output + cnn_f)
        else:
            mask_probs = self.mask_net(attn_output)    
        return mask_probs

    def forward_action_logits(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        
        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # torch.Size([N, cW*cH, share_out_channels])
        
        attn_output = self.attention(cnn_f, cnn_f, cnn_f)
        if self.cnn_shortcut is True:
            action_logits = self.actor_net(attn_output + cnn_f)
        else:
            action_logits = self.actor_net(attn_output) 
        return action_logits
    
    def forward_critic(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        cnn_f = self.share_extractor(observations)
        cnn_f = cnn_f.flatten(2).transpose(1, 2)    # torch.Size([N, cW*cH, output_channels])
        
        if self.position_encode is True:
            cnn_f = self.positional_encoding(cnn_f)    # torch.Size([N, cW*cH, share_out_channels])
        
        attn_output = self.attention(cnn_f, cnn_f, cnn_f) 
        if self.cnn_shortcut is True:
            values = self.critic_net(attn_output + cnn_f)
        else:    
            values = self.critic_net(attn_output)      
        return values
    
    
class CnnMlpNetwork1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )

    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
    def forward_mask_probs(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_mk = self.mask_extractor(share_features)
        return self.mask_net(latent_mk)
    
    def forward_critic(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_vf = self.critic_extractor(share_features)
        return self.critic_net(latent_vf)
    
class CnnMlpNetwork2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )

    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
    def forward_mask_probs(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_mk = self.mask_extractor(share_features)
        return self.mask_net(latent_mk)
    
    def forward_critic(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_vf = self.critic_extractor(share_features)
        return self.critic_net(latent_vf)
    
class CnnMlpNetwork3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )

    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
    def forward_mask_probs(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_mk = self.mask_extractor(share_features)
        return self.mask_net(latent_mk)
    
    def forward_critic(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_vf = self.critic_extractor(share_features)
        return self.critic_net(latent_vf)
    
class CnnMlpNetwork4(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int,
        normalize_images: bool = False,
        share_out_channels: int = 64,
        mask_out_channels: int = 8,
        actor_out_channels: int = 8,
        critic_out_channels: int = 4,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        
        self.share_out_channels = share_out_channels
        self.mask_out_channels = mask_out_channels
        self.actor_out_channels = actor_out_channels
        self.critic_out_channels = critic_out_channels

        self.share_extractor = nn.Sequential(
            (nn.Conv2d(self.share_input_channels, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
            (nn.Conv2d(64, share_out_channels, kernel_size=(3,3), stride=1, padding=1)),
            nn.ReLU(),
        )

        self.mask_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, mask_out_channels)
        self.actor_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, actor_out_channels)
        self.critic_n_flatten = self._get_n_flatten(self.share_extractor, share_out_channels, critic_out_channels)

        self.mask_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, mask_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.mask_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.mask_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
            nn.Sigmoid(),
        )

        self.actor_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, actor_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.actor_n_flatten, self.hidden_dim)),
            nn.ReLU(),
        )

        self.actor_net = nn.Sequential(
            (nn.Linear(self.hidden_dim, self.action_dim)),
        )

        self.critic_extractor = nn.Sequential(
            nn.Conv2d(share_out_channels, critic_out_channels, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(self.critic_n_flatten, self.hidden_dim//2)),
            nn.ReLU(),
        )

        self.critic_net = nn.Sequential(
            (nn.Linear(self.hidden_dim//2, 1))
        )

    def forward(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        share_features = self.share_extractor(observations)

        latent_mk = self.mask_extractor(share_features)
        mask_probs = self.mask_net(latent_mk)

        latent_pi = self.actor_extractor(share_features)
        action_logits = self.actor_net(latent_pi)

        latent_vf = self.critic_extractor(share_features)
        values = self.critic_net(latent_vf)
        return mask_probs, action_logits, values

    def forward_action_logits(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_pi = self.actor_extractor(share_features)
        return self.actor_net(latent_pi)
    
    def forward_mask_probs(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_mk = self.mask_extractor(share_features)
        return self.mask_net(latent_mk)
    
    def forward_critic(self, observations: th.Tensor, interest_mask: th.Tensor = None) -> th.Tensor:
        share_features = self.share_extractor(observations)
        latent_vf = self.critic_extractor(share_features)
        return self.critic_net(latent_vf)
