import torch

tensor_example = torch.tensor(
    [
        [1, -1, 1],
        [2, 2, 2],
        [3, 4, 3]
    ]
)
print(tensor_example.max(dim=1))
tensor_example2 = torch.tensor(
    [
        [[1, -1, 1],
        [2, 2, 2],
        [3, 4, 3]],
        [[1, -1, 1],
        [2, 2, 2],
        [3, 4, 3]]
    ]
)
tensor_example3 = torch.tensor(
    [
        [1, -1, 1],
        [2, 2, 2],
        [3, 4, 3],
        [1, -1, 1],
        [2, 2, 2],
        [3, 4, 3]
    ]
)
print(tensor_example3.max(dim=0))
a = torch.ones_like(tensor_example) * -1
b = torch.tensor([1, 2, 4])

print(tensor_example * b)
print(tensor_example2.shape)
c = torch.max(tensor_example2, dim=2, keepdim=True)[0]
print(c)
print(c.shape)
d = (c>2)[:, :, 0]
print(d.shape)
print(tensor_example.unsqueeze(dim=1)[..., 1])
