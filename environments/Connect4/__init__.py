from .config import training_config, network_config, env_config
from .env_cython import Env
from .Network import CNN, ViT, CNN_old
from .utils import instant_augment, inspect