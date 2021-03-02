# import torch
# import torchvision.models as models
# from nets.network import draftNet, ResNet50
#
# backbone = models.resnet101(pretrained=True)
# # model = ResNet50(backbone, num_classes=15)
#
# print(backbone)
import numpy as np

# a = [12.,  5., 8.5,  13., 8.5,  3.5,  8.5, 14.5,  1.,  14.5,  8.5,  3.5, 11.,   2.,  6. ]
# b = [ 5.,  4., 14.,  1.,  6.,   15.,  11., 2.,    10., 3.,    9.,   7.,  12.,   13., 8.]
#
# # a = [12.,  5., 7,  13., 8,  3,  9, 14,  1.,  15,  10,  4, 11.,   2.,  6. ]
# # b = [ 5.,  4., 14.,  1.,  6.,   15.,  11., 2.,    10., 3.,    9.,   7.,  12.,   13., 8.]
#
# s = 0
# N = 15
# for i in range(15):
#     c = float(a[i]-b[i])
#     s += c * c
# s *= 6
# s /= N*(N*N-1)
# s = 1 - s
# print(s)
# # -0.5830666876544621
# from functools import reduce
# import operator
# import numpy as np
# l = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
# print(l)
# sum1 = reduce(operator.add, l)
#
# print(sum1)
import numpy as np
# from random import shuffle
# shuffle_list = [i for i in range(15)]
# shuffle(shuffle_list)
# a = [12.,  5., 8.5,  13., 8.5,  3.5,  8.5, 14.5,  1.,  14.5,  8.5,  3.5, 11.,   2.,  6. ]
# a = [a[i] for i in shuffle_list]
# print(shuffle_list)
# print(a)
import torchvision.models as models
backbone = models.resnet34(pretrained=True)

print(backbone)

