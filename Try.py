import torch

from torch import nn

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_sequence_length):
#         super().__init__()
#         self.max_sequence_length = max_sequence_length
#         self.d_model = d_model
#
#     def forward(self):
#         even_i = torch.arange(0, self.d_model, 2).float()
#         print("even_i: ", even_i)
#         denominator = torch.pow(10000, even_i/self.d_model)
#         print("denominator: ", denominator)
#         position = (torch.arange(self.max_sequence_length)
#                           .reshape(self.max_sequence_length, 1))
#         print("position: ", position)
#         even_PE = torch.sin(position / denominator)
#         print("even_PE: ", even_PE)
#         odd_PE = torch.cos(position / denominator)
#         print("odd_PE: ", odd_PE)
#         stacked = torch.stack([even_PE, odd_PE], dim=2)
#         print("stacked: ", stacked)
#         PE = torch.flatten(stacked, start_dim=1, end_dim=2)
#         print("PE: ", PE)
#         return PE
#
#
# PE = PositionalEncoding(4, 3)
# PE.forward()

# t = torch.arange(0, 10)
# # torch.reshape(, (2, 2)
# print(t.reshape(2, 5))


def forward(inputs):
    dims = [-1, -2]
    mean = inputs.mean(dim=dims, keepdim=True)
    v1 = (inputs - mean)
    v2 = v1**2
    var = v2.mean(dim=dims, keepdim=True)
    std = (var + 0.0001).sqrt()
    y = (inputs - mean) / std
    print(y)


forward(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],[6.0, 7.0, 8.0, 9.0, 10.0]]))