# pyright: reportIndexIssue=false, reportArgumentType=false

import csv
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, cast

import pytest
import torch
from pydantic import BaseModel

from mblm import MBLM, MBLMModelConfig, TransformerBlock
from mblm.model.embeddings import MBLM_TOKEN_EMB_MIGRATION
from mblm.utils.io import (
    CSVWriter,
    NDJSONWriter,
    dump_yml,
    load_model_state,
    load_yml,
    read_jsonl,
    save_model_state,
)

# TODO: Python 3.12, assert_type


class TestYMLUtils:
    def test_cfrom_yml(self):
        class Klass(BaseModel):
            num: int

        with tempfile.TemporaryDirectory() as temp_dir:
            kls = Klass(num=5)
            dumped_to = dump_yml(Path(temp_dir) / "file", kls)
            restored = load_yml(dumped_to, Klass)
            # assert_type(restored, Klass)
            assert isinstance(restored, Klass)


class DummyCSVEntry(NamedTuple):
    kind: str
    idx: int
    time: str


class TestCSVWriter:
    def test_parallel(self):
        """Simulate parallel writes to the same file with index and timestamp."""

        # Verify the file content
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_output_dir = Path(tmpdir)
            csv_writer = CSVWriter[DummyCSVEntry](output_dir=temp_output_dir, file_name="test")

            def write_row(index):
                row = DummyCSVEntry(
                    kind="test",
                    idx=index,
                    time=datetime.now().isoformat(),
                )
                csv_writer.write_row(row)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(write_row, i) for i in range(10)]
                for future in futures:
                    future.result()
                csv_file = temp_output_dir / "test.csv"
                with csv_file.open("r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    assert reader[0] == list(DummyCSVEntry._fields)
                    assert len(reader) == 11

                    indexes = []
                    for row in reader[1:]:
                        assert row[0] == "test"
                        assert len(row[2]) > 0
                        indexes.append(row[1])
                    # Order may differ due to concurrent writing
                    assert list(map(str, range(10))) == sorted(indexes)


class TestModelCheckpointing:
    class Model(torch.nn.Module):  # noqa
        def __init__(self, num_embs: int, emb_dim: int):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embs, emb_dim)
            self.seq = torch.nn.Sequential(
                torch.nn.Embedding(num_embs, emb_dim),
                torch.nn.Linear(emb_dim, num_embs),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.emb(x)
            return self.seq(x)

    class OldModel(torch.nn.Module):  # noqa
        def __init__(self):
            super().__init__()
            self.keep = torch.nn.Parameter(torch.randn((1, 1)))
            self.old_param = torch.nn.Parameter(torch.ones((1, 1)))
            self.old_param_x = torch.nn.Parameter(torch.ones((1, 1)))

    class NewModel(torch.nn.Module):  # noqa
        def __init__(self):
            super().__init__()
            self.keep = torch.nn.Parameter(torch.randn((1, 1)))
            self.new_param = torch.nn.Parameter(torch.zeros((1, 1)))
            self.new_param_x = torch.nn.Parameter(torch.zeros((1, 1)))

    @pytest.mark.parametrize(
        "src_emb_size,tgt_emb_size",
        [
            (4, 4),  # same size
            (2, 3),  # slightly larger
            (4, 10),  # much larger
        ],
    )
    @torch.no_grad()
    def test_load_map_state(self, src_emb_size: int, tgt_emb_size: int):
        assert src_emb_size <= tgt_emb_size, "Invalid test"

        model_src = self.Model(src_emb_size, 3)
        model_tgt = self.Model(tgt_emb_size, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, chkpoint = save_model_state(tmpdir, "checkpoint", model_src, 0)
            model_tgt, _ = load_model_state(
                chkpoint,
                model_tgt,
                map_extend_embeddings={
                    "emb.weight",
                    "seq.0.weight",
                    "seq.1.weight",
                    "seq.1.bias",
                },
            )
        assert model_tgt.emb.weight.size(0) == tgt_emb_size
        assert model_tgt.emb.weight[:src_emb_size].equal(model_src.emb.weight)
        assert model_tgt.seq[1].weight[:src_emb_size].equal(model_src.seq[1].weight)
        assert model_tgt.seq[1].bias[:src_emb_size].equal(model_src.seq[1].bias)

        # make sure that the first part of the logits is equal
        max_token_id = src_emb_size - 1
        input_both = torch.tensor([max_token_id]).long()
        src_logits = model_src.forward(input_both)
        tgt_logits = model_tgt.forward(input_both)

        assert tgt_logits[:, :src_emb_size].equal(src_logits)

    @torch.no_grad()
    def test_load_map_state_mbml(self):
        def create_model(num_tokens: int):
            return MBLM(
                MBLMModelConfig(
                    num_tokens=num_tokens,
                    pad_token_id=0,
                    hidden_dims=(1024, 512),
                    num_layers=(1, 1),
                    seq_lens=(8192, 8),
                    train_checkpoint_chunks=None,
                    block=TransformerBlock(
                        attn_head_dims=64,
                        attn_num_heads=8,
                        attn_use_rot_embs=True,
                        pos_emb_type=None,
                    ),
                )
            )

        num_src_emb, num_tgt_emb = 5, 6
        model_src = create_model(num_src_emb)
        model_tgt = create_model(num_tgt_emb)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, chkpoint = save_model_state(tmpdir, "checkpoint", model_src, 0)
            model_tgt, _ = load_model_state(
                chkpoint,
                model_tgt,
                map_extend_embeddings=MBLM_TOKEN_EMB_MIGRATION,
            )
            """
            This is the structure of the embeddings we're migrating:

                    (token_embs_rev): ModuleList(
            case 1:     (0): Embedding(255, 512, padding_idx=0)
                        (1): Sequential(
            case 2:         (0): Embedding(255, 512, padding_idx=0)
                            (1): Rearrange('... r d -> ... (r d)')
                            (2): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
                            (3): Linear(in_features=4096, out_features=1024, bias=True)
                            (4): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                )
                ...

            cases 3/4: (to_logits): Linear(in_features=512, out_features=255, bias=True)
            )
            """
            src_emb = cast(torch.nn.Embedding, model_src.token_embs_rev[0])
            src_emb_seq = cast(torch.nn.Sequential, model_src.token_embs_rev[1])
            tgt_emb = cast(torch.nn.Embedding, model_tgt.token_embs_rev[0])
            tgt_emb_seq = cast(torch.nn.Sequential, model_tgt.token_embs_rev[1])

            # case 1, base embedding
            assert tgt_emb.num_embeddings == num_tgt_emb
            assert tgt_emb.weight[:num_src_emb].equal(src_emb.weight)
            # case 2, embedding in sequential
            assert tgt_emb_seq[0].num_embeddings == num_tgt_emb
            assert tgt_emb_seq[0].weight[:num_src_emb].equal(src_emb_seq[0].weight)
            # case 3/4, logits
            assert model_tgt.to_logits.weight.size(0) == num_tgt_emb
            assert model_tgt.to_logits.bias.size(0) == num_tgt_emb
            assert model_tgt.to_logits.weight[:num_src_emb].equal(model_src.to_logits.weight)
            assert model_tgt.to_logits.bias[:num_src_emb].equal(model_src.to_logits.bias)

            # check if new token id works
            max_new_token_id = num_tgt_emb - 1
            input_for_tgt_model_only = torch.tensor([[max_new_token_id]]).long()
            with pytest.raises(Exception):
                # should fail for old model
                model_src.forward(input_for_tgt_model_only)
            try:
                # should work for migrated model
                model_tgt.forward(input_for_tgt_model_only)
            except Exception as error:
                pytest.fail(f"Forward pass should work: {error}")

    @pytest.mark.parametrize("rename_from,rename_to", [("old_param", "new_param")])
    def test_load_map_state_rename(self, rename_from: str, rename_to: str):
        new_model = self.NewModel()
        old_model = self.OldModel()
        assert not new_model.new_param.all()  # before migration
        assert old_model.old_param.all()

        with tempfile.TemporaryDirectory() as tmpdir:
            _, chkpoint = save_model_state(tmpdir, "checkpoint", old_model, 0)

            new_model, _ = load_model_state(
                chkpoint,
                new_model,
                map_rename_modules=((rename_from, rename_to),),
            )

            assert new_model.new_param.equal(old_model.old_param)
            # make sure all modules with prefix are renamed
            assert new_model.new_param_x.equal(old_model.old_param_x)

    @pytest.mark.parametrize(
        "src_emb_size,tgt_emb_size,src_emb_dim,tgt_emb_dim,err_msg",
        [
            (3, 2, 4, 4, "Mapping to a smaller number of embeddings"),  # shrinking
            (3, 3, 4, 5, "Mapping to a smaller embedding dimension"),  # incompatible emb dim
        ],
    )
    def test_load_map_state_unsupported_mapping(
        self,
        src_emb_size: int,
        tgt_emb_size: int,
        src_emb_dim: int,
        tgt_emb_dim: int,
        err_msg: str,
    ):
        src_mod = self.Model(src_emb_size, src_emb_dim)
        tgt_mod = self.Model(tgt_emb_size, tgt_emb_dim)
        with tempfile.TemporaryDirectory() as tmpdir:
            _, chkpoint = save_model_state(tmpdir, "checkpoint", src_mod, 0)
            with pytest.raises(ValueError) as exc_info:
                load_model_state(
                    chkpoint,
                    tgt_mod,
                    map_extend_embeddings={
                        "emb.weight",
                        "seq.0.weight",
                        "seq.1.weight",
                        "seq.1.bias",
                    },
                )

            assert err_msg in str(exc_info.value)

    def test_load_map_state_unsupported_different_modules(self):
        src_mod = self.OldModel()
        tgt_mod = self.NewModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            _, chkpoint = save_model_state(tmpdir, "checkpoint", src_mod, 0)
            with pytest.raises(ValueError) as exc_info:
                load_model_state(
                    chkpoint,
                    tgt_mod,
                    map_extend_embeddings={"keep"},
                )

            assert "Expected source and target state dict to match" in str(exc_info.value)


class TestNDJSONWriter:
    class MyClass(BaseModel):  # noqa: D106
        data: str

    first_entry = MyClass(data="a")
    temp_entry = MyClass(data="bbbbbbb")  # long entry
    second_entry = MyClass(data="c")

    def test_write_and_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "file.jsonl"
            writer = NDJSONWriter[TestNDJSONWriter.MyClass](file)
            writer.write_line(self.first_entry)
            writer.write_line(self.temp_entry)
            writer.remove_last_line()
            writer.write_line(self.second_entry)

            result = read_jsonl(file, parse_lines_to=self.MyClass)
            assert len(result) == 2
            assert result[0] == self.first_entry
            assert result[1] == self.second_entry

    def test_write_and_remove_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "file.jsonl"
            writer = NDJSONWriter[TestNDJSONWriter.MyClass](file)
            writer.write_line(self.first_entry)
            writer.write_line(self.temp_entry)
            writer.remove_last_line()
            writer.remove_last_line()
            writer.remove_last_line()

            result = read_jsonl(file, parse_lines_to=self.MyClass)
            assert len(result) == 0
