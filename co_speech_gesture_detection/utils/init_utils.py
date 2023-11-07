import math
import random

import torch
from torch import nn
import numpy as np


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reinitialize_classification_layer(model, num_classes=2):
    for name, module in model.named_children():
        if name == 'fc':
            module = nn.Linear(in_features=module.in_features, out_features=num_classes, bias=True)
            nn.init.normal_(module.weight, 0, math.sqrt(2. / num_classes))
            model.fc = module
    return model
