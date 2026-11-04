"""OLVQ: three learnable lattice routes, six neural image backbones."""
from .model import ModelConfig, OLVQModel, build_model, MODELS, ROUTES
__version__ = '0.2.0'
__all__ = ['ModelConfig', 'OLVQModel', 'build_model', 'MODELS', 'ROUTES']
