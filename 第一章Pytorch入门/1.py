# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import torch

x = torch.rand(5, 3)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device = device)
    x = x.to(device)
    z = torch.add(x, y)
    print(z)
    print(z.to("cpu", torch.double))
    