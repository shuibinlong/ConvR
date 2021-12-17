import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    @classmethod
    def init_model(cls, config):
        model = cls(config)
        return model
