# import torch
# import deep_hough
#
# """
# 이 py 실행이 정상 작동시 deep_hough library ok
# """
# # Create a dummy input tensor (e.g., a single 4D tensor)
# feat = torch.rand(1, 3, 64, 64, device='cuda')
# output = torch.zeros(1, 3, 180, 128, device='cuda')
#
# # Call the forward function
# res = deep_hough.forward(feat, output, 180, 128)
# print("Forward output sum:", res[0].sum().item())


# import os
# import torch
# print(torch.cuda.is_available())
# print(os.environ.get("CUDA_PATH"))

import numpy as np
import torch

# Create a NumPy array
np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

# Convert the NumPy array to a PyTorch tensor
torch_tensor = torch.from_numpy(np_array.copy())

print(torch_tensor)