import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from models.projector import MultimodalProjector

# fake fusion feature
fusion_feature = torch.randn(1, 1024)

model = MultimodalProjector()

out = model(fusion_feature)

print("Input shape:", fusion_feature.shape)
print("Output shape:", out.shape)