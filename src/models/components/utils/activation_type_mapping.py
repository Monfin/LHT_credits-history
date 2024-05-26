from torch import nn

ACTIVATION_TYPE_MAPPING = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "none": nn.Identity
}