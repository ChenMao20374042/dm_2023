import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch
import numpy as np

def generate_mask_for_batch(sequence_lengths, num_attn_heads, max_length):

    key_masks = []

    for length in sequence_lengths:
        key_mask = torch.ones([1, max_length]) == 1
        key_mask[:, 0:length] = False
        key_masks.append(key_mask)

    key_masks = torch.cat(key_masks, dim=0)
    
    return key_masks

class Trajformer(nn.Module):

    def __init__(self, in_dim, hid_dim, num_attn_heads, num_layers, max_len=300):
        super().__init__()

        self.num_attn_heads = num_attn_heads

        self.input_embedding = nn.Linear(in_dim, hid_dim)
        
        self.cls = nn.Parameter(torch.rand([1, hid_dim]), requires_grad=True)

        encoder_layer = TransformerEncoderLayer(hid_dim, num_attn_heads, hid_dim, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)


        self.output_layer = nn.Sequential(nn.Linear(hid_dim,hid_dim), nn.ReLU(), nn.Linear(hid_dim,1))

    def forward(self, x, pad_val):
        """
            x in shape (B, L, D), where B is batch size, L is sequence length, D is input dimension.
        """
        B, L, D = x.shape
        original_mask = x[:, :, 0] == pad_val
        length = torch.sum(x[:, :, 0] != pad_val, dim=1)
        key_masks = generate_mask_for_batch(length+1, self.num_attn_heads, max_length=L+1)
        key_masks = key_masks.to(x.device)
        x = self.input_embedding(x)

        x = torch.cat([self.cls.repeat(B,1,1), x], dim=1)
        x = self.encoder(x, src_key_padding_mask=key_masks)
        x = x[:, 0, :]
        x = self.output_layer(x)
        
        x = x.squeeze(-1)
        return x

        
