"""Package initialization."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

__version__ = "0.0.1"


from mblm.model.config import MBLMModelConfig, MBLMReturnType
from mblm.model.mamba import MambaBlock
from mblm.model.mblm import MBLM
from mblm.model.transformer import TransformerBlock

__all__ = [
    "MBLM",
    "MBLMModelConfig",
    "MBLMReturnType",
    "TransformerBlock",
    "MambaBlock",
]
