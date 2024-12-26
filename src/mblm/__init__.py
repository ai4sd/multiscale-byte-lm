"""Package initialization."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

__version__ = "0.0.1"


from mblm.model.config import MBLMModelConfig, MBLMReturnType
from mblm.model.mblm import MBLM

__all__ = ["MBLM", "MBLMModelConfig", "MBLMReturnType"]
