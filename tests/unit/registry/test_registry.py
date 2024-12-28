from mblm import MambaBlockConfig, TransformerBlockConfig
from mblm.model.config import block_registry
from mblm.train.mblm import dataset_registry


def test_registry():
    assert "clevr" in dataset_registry
    assert "pg19" in dataset_registry
    assert TransformerBlockConfig in block_registry
    assert MambaBlockConfig in block_registry
