
import torch
import torch.nn as nn
from Model.MambaS import MambaS

class MSCM_Block(nn.Module):
    def __init__(self,
                 in_channels_t2: int = 32, # Intermediate skip connection channels
                 in_channels_t3: int = 64, # Intermediate skip connection channels
                 in_channels_t4: int = 96  # Shallowest skip connection channels
                ):
        super().__init__()

        self.mamba1 = MambaS(hidden_dim=32, d_state=16)
        self.mamba2 = MambaS(hidden_dim=32, d_state=16)

        self.skip_scale = nn.Parameter(torch.ones(1))

        self.norm1 = nn.LayerNorm(32)
        self.norm2 = nn.LayerNorm(32)

    def _FeatureTokenizer(self, t2, t3, t4):
        """
        Aggregates multi-resolution feature maps into 1D token sequences 
        for long-range dependency modeling via Mamba.
        """
        B = t2.shape[0]
        n_tokens_list = [t.shape[2:].numel() for t in [t2, t3, t4]]
        n_sizes_list = [t.shape[2:] for t in [t2, t3, t4]]

        # Flatten spatial dimensions to (B, N, C)
        t_flat_list = [t.flatten(2).transpose(-1, -2) for t in [t2, t3, t4]] 

        # Construct parallel Mamba inputs through channel-wise slicing and concatenation
        mamba_input1 = torch.cat((t_flat_list[0][:, :, :32],
                                  t_flat_list[1][:, :, :32],
                                  t_flat_list[2][:, :, :32]), dim=1)

        mamba_input2 = torch.cat((t_flat_list[1][:, :, 32:64],
                                  t_flat_list[2][:, :, 32:64],
                                  t_flat_list[2][:, :, 64:96]), dim=1)

        return mamba_input1, mamba_input2, n_tokens_list, n_sizes_list

    def _FeatureDetokenizer(self,
                             mamba_out1: torch.Tensor,
                             mamba_out2: torch.Tensor,
                             orig_tokens: list,
                             orig_sizes: list):
        """
        Partitions and reshapes processed tokens back into 2D feature maps 
        to maintain spatial consistency for the decoder.
        """
        B = mamba_out1.shape[0]

        # Reconstruct x2: Refined features from t2
        x2_out = mamba_out1[:, :orig_tokens[0], :].transpose(-1, -2).reshape(B, 32, orig_sizes[0][0], orig_sizes[0][1])

        # Reconstruct x3: Cross-stream fusion for t3
        x3_p1 = mamba_out1[:, orig_tokens[0] : (orig_tokens[0] + orig_tokens[1]), :]
        x3_p2 = mamba_out2[:, :orig_tokens[1], :]
        x3_out = torch.cat((x3_p1, x3_p2), dim=2).transpose(-1, -2).reshape(B, 64, orig_sizes[1][0], orig_sizes[1][1])

        # Reconstruct x4: Multi-segment fusion for t4
        x4_p1 = mamba_out1[:, orig_tokens[0] + orig_tokens[1]:, :] 
        x4_p2 = mamba_out2[:, orig_tokens[1]:orig_tokens[1] + orig_tokens[2], :] 
        x4_p3 = mamba_out2[:, orig_tokens[1] + orig_tokens[2]:, :] 
        x4_out = torch.cat((x4_p1, x4_p2, x4_p3), dim=2).transpose(-1, -2).reshape(B, 96, orig_sizes[2][0], orig_sizes[2][1])

        return x2_out, x3_out, x4_out

    def forward(self, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor):
        # Feature Tokenization
        mamba_input1, mamba_input2, n_tokens_list, n_sizes_list = self._FeatureTokenizer(t2, t3, t4)

        # Sequence Modeling with Residual Connections
        mamba_output1 = self.norm1(self.mamba1(mamba_input1))
        mamba_output2 = self.norm2(self.mamba2(mamba_input2))
        mamba_out1 = self.skip_scale * mamba_input1 + mamba_output1
        mamba_out2 = self.skip_scale * mamba_input2 + mamba_output2

        # Feature Detokenization
        x2_out, x3_out, x4_out = self._FeatureDetokenizer(
            mamba_out1,
            mamba_out2,
            n_tokens_list,
            n_sizes_list
        )
        return x2_out, x3_out, x4_out