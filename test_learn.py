import numpy as np
import shapely
from shapely.geometry import Polygon
import torch

target = torch.ones(size=(100, 5)) * -1
print(target[torch.lt(torch.tensor([0.5008]), 0.4), :])