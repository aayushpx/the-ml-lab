import torch
import torch.nn.functional as F

image = torch.tensor([
    [1., 2., 3., 4., 5.],
    [5., 4., 3., 2., 1.],
    [1., 2., 3., 4., 5.],
    [5., 4., 3., 2., 1.],
    [1., 2., 3., 4., 5.]
])

kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
])

# Manual convolution
kernel_size = 3

output_height = image.shape[0] - kernel_size + 1
output_width = image.shape[1] - kernel_size + 1

output = torch.zeros(output_height, output_width)

for i in range(output_height):
    for j in range(output_width):

        patch = image[
        i:i + kernel_size,
        j:j + kernel_size
        ]

        output[i, j] = torch.sum(patch * kernel)

print("Image")
print(image)
print("\nKernel")
print(kernel)
print("\nFeature map:")
print(output)

# PytTorch expects:
# [batch, channels, height, width]

image_4d = image.unsqueeze(0).unsqueeze(0) # inserts a new dimension
kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

pytorch_output = F.conv2d(
    image_4d,
    kernel_4d
)

pytorch_output = pytorch_output.squeeze()

print("\nPyTorch feature map:")
print(pytorch_output)

print("\nSame result:", torch.equal(output, pytorch_output))

