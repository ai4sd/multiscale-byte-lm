from mblm import MambaBlock, TransformerBlock
from mblm.model.config import block_registry
from mblm.train.mblm import dataset_registry


def test_registry():
    assert "clevr" in dataset_registry
    assert "pg19" in dataset_registry
    assert TransformerBlock in block_registry
    assert MambaBlock in block_registry
