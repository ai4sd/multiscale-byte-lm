__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import math
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Iterable, Literal, get_args

import polars as pl
from pydantic import ValidationError

from mblm.data.types import ModelMode
from mblm.trainer.mblm import TrainOutputConfig
from mblm.utils.io import load_yml

ModelType = Literal["SSM", "Transformer", "Mixed"]

# weirdly enough, we cannot use this enum in the schema definition directly and
# need to cast after the df has been created
_mode_enum = pl.Enum([ModelMode.TRAIN.value, ModelMode.VALID.value, ModelMode.TEST.value])
_model_type_enum = pl.Enum(get_args(ModelType))

_gpu_df_default_schema = dict(
    cum_batch=pl.Int32,
    num_items=pl.Int16,
    kind=pl.String,  # renamed to mode
    fw_time=pl.Float32,
    bw_time=pl.Float32,
    allocated=pl.Float32,
    allocated_max=pl.Float32,
    reserved=pl.Float32,
    reserved_max=pl.Float32,
    total=pl.Float32,
)
# we have added new fields since, check Aggregator
_loss_df_default_schema = dict(
    timestamp=pl.Datetime,
    elements_seen=pl.Int64,
    kind=pl.String,  # renamed to mode
    epoch=pl.Int64,
    batch=pl.Int64,
    cum_batch=pl.Int64,
    loss=pl.Float64,
)
_bpb_scale_factor = math.log2(math.e)


@dataclass
class Filter:
    mode: ModelMode | None = None
    num_stages_gte: int | None = None
    num_stages_lte: int | None = None


class Aggregator:
    def __init__(self, folders: Iterable[tuple[str, str]]):
        self.configs, self.df_exp, self._df_train, df_gpu = self._read_experiments(folders)
        self.exp_names = [folder_name[1] for folder_name in folders]

        # drop the very first train batch measurement, it includes the warmup times
        self._df_gpu = df_gpu.filter(
            ~((pl.col("mode") == "train") & (pl.col("cum_batch") == 0)),
        )

    def df_train(self, filter: Filter = Filter()) -> pl.DataFrame:
        return self._filter_df(self._df_train, filter).pipe(self._rename_values_for_plot)

    def df_gpu(self, filter: Filter = Filter()) -> pl.DataFrame:
        return self._filter_df(self._df_gpu, filter).pipe(self._rename_values_for_plot)

    def _filter_df(self, df_with_size: pl.DataFrame, filter: Filter) -> pl.DataFrame:
        f = filter
        return (
            df_with_size.lazy()
            .filter((pl.col("mode").eq(f.mode)) if f.mode else True)
            .filter((pl.col("stages").ge(f.num_stages_gte)) if f.num_stages_gte else True)
            .filter((pl.col("stages").le(f.num_stages_lte)) if f.num_stages_lte else True)
            .collect()
        )

    def _rename_values_for_plot(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("kind").replace(["train", "valid", "test"], ["Training", "Validation", "Test"]),
            pl.col("stages")
            .cast(pl.String)
            .replace(["1", "2", "3"], ["1D, 8k ctx", "2D, 100k ctx", "3D, 1m ctx"]),
        )

    def _df_with_common_props(
        self,
        df: pl.DataFrame,
        config: TrainOutputConfig,
        exp_name: str,
        model_type: str,
    ):
        return df.with_columns(
            mode=pl.col("kind").cast(_mode_enum),
            name=pl.lit(exp_name),
            model_type=pl.lit(model_type).cast(_model_type_enum),
            stages=pl.lit(len(config.params.hidden_dims)).cast(pl.Int8),
            seq_len=pl.lit(config.params.input_seq_len),
        )

    def _determine_model_type(self, config: TrainOutputConfig) -> ModelType:
        block_ids = [block.block_type for block in config.params.stage_blocks]
        if all([block_id == "mamba2" for block_id in block_ids]):
            return "SSM"
        if all([block_id == "transformer" for block_id in block_ids]):
            return "Transformer"
        return "Mixed"

    def _read_experiments(
        self, folders: Iterable[tuple[str, str]]
    ) -> tuple[list[TrainOutputConfig], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        dfs_exp: list[pl.DataFrame] = []
        dfs_train: list[pl.DataFrame] = []
        dfs_gpu: list[pl.DataFrame] = []
        configs: list[TrainOutputConfig] = []
        for folder, exp_name in folders:
            path = Path(folder)
            try:
                config = load_yml(path / "config.yaml", TrainOutputConfig, try_yaml_suffixes=True)

                grad_acc_every = config.train.gradient_accumulate_every
                model_type = self._determine_model_type(config)

                # static df with experiment/model details
                df_exp = pl.DataFrame().with_columns(
                    name=pl.lit(exp_name),
                    model_type=pl.lit(model_type).cast(_model_type_enum),
                    ctx_size_total=pl.lit(config.params.input_seq_len),
                    ctx_sizes=pl.lit(config.params.seq_lens),
                    elements_trained=pl.lit(config.train.target_elements),
                    params_m=pl.lit(config.summary.parameter_count).truediv(1e6).round(0),
                    num_layers=pl.lit(config.params.num_layers),
                    num_tokens=pl.lit(config.params.num_tokens),
                    checkpoints=pl.lit(config.params.train_checkpoint_chunks),
                    patch_pos_emb=pl.lit(
                        [b.pos_emb_type or "none" for b in config.params.stage_blocks]
                    ),
                    lr=pl.lit(config.train.learning_rate),
                    batch_size=pl.lit(config.train.batch_size),
                    grad_step=pl.lit(grad_acc_every * config.train.batch_size),
                    grad_clip=pl.lit(config.train.gradient_clipping),
                    num_gpus=pl.lit(config.summary.num_workers),
                    # dataset_args=pl.lit(config.io.dataset_args),
                    comment=pl.lit(config.io.description),
                    training_finished=pl.lit(bool(config.summary.training_end)),
                    error=pl.lit(config.summary.error),
                )

                dfs_exp.append(df_exp)

                df_train = (
                    self._try_read_csv(path / "loss.csv", _loss_df_default_schema, True)
                    .pipe(
                        self._df_with_common_props,
                        config=config,
                        exp_name=exp_name,
                        model_type=model_type,
                    )
                    .with_columns(
                        pl.col("elements_seen") * config.summary.num_workers,  # TODO check
                        bpb=pl.col("loss") * _bpb_scale_factor,
                        cum_batch=(pl.col("cum_batch") + 1),
                    )
                    .with_columns(step=((pl.col("cum_batch")) // grad_acc_every).cast(pl.Int32))
                )
                dfs_train.append(df_train)

                df_gpu = (
                    self._try_read_csv(path / "timemem.csv", _gpu_df_default_schema)
                    .pipe(
                        self._df_with_common_props,
                        config=config,
                        exp_name=exp_name,
                        model_type=model_type,
                    )
                    .rename({"num_items": "batch_size"})
                )
                dfs_gpu.append(df_gpu)
            except ValidationError as e:
                print(folder)
                raise e

            configs.append(config)
        df_exp = reduce(lambda new_df, df: df.vstack(new_df), dfs_exp)
        df_train = reduce(lambda new_df, df: df.vstack(new_df), dfs_train)
        df_gpu = reduce(lambda new_df, df: df.vstack(new_df), dfs_gpu)
        return configs, df_exp, df_train, df_gpu

    def _try_read_csv(self, csv_path: str | Path, schema: dict, is_loss_csv: bool = False):
        try:
            df = pl.read_csv(
                csv_path,
                schema_overrides=schema,
                raise_if_empty=False,
                null_values="nan",
            )
        except Exception:
            df = pl.DataFrame(schema=schema)

        # we added these three fields later in the project, hence, they don't exist
        # for all existing csv files - interpolate
        if is_loss_csv:
            if "lr" not in df.columns:
                df = df.with_columns(
                    lr=pl.lit(-1, pl.Float64),
                    avg_grad=pl.lit(-1, pl.Float64),
                    avg_grad_clipped=pl.lit(-1, pl.Float64),
                )
            if "gpu_rank" not in df.columns:
                df = df.with_columns(gpu_rank=pl.lit(0, pl.Int16))

            df = df.with_columns(
                pl.col("avg_grad", "avg_grad_clipped").cast(pl.Float64),
                pl.col("gpu_rank").cast(pl.Int16),
            )

        return df

    def _convert_utc_timezones(
        self,
        df: pl.DataFrame,
        columns: list[str],
        to_timezone: str,
    ) -> pl.DataFrame:
        utc_fmt = "%Y-%m-%dT%H:%M:%S%.f"
        exprs = [
            pl.col(col)
            .str.to_datetime(format=utc_fmt, time_zone="UTC", time_unit="ms")
            .dt.replace_time_zone(to_timezone)
            for col in columns
        ]
        return df.with_columns(*exprs)

    def df_train_times(self) -> pl.DataFrame:
        df = pl.DataFrame(
            [
                [
                    name,
                    cfg.summary.training_start,
                    cfg.summary.training_end or datetime.now().isoformat(),
                    bool(cfg.summary.training_end),
                ]
                for cfg, name in zip(self.configs, self.exp_names)
            ],
            orient="row",
            schema=["name", "train_start", "train_end", "has_finished"],
        )
        return (
            df.pipe(
                self._convert_utc_timezones,
                columns=["train_start", "train_end"],
                to_timezone="Europe/Brussels",
            )
            .with_columns(
                train_dur=(pl.col("train_end") - pl.col("train_start")),
            )
            .with_columns(
                train_dur_h=pl.col("train_dur").dt.total_hours(),
                train_dur_str=pl.col("train_dur").dt.total_days().cast(pl.String)
                + "d "
                + (pl.col("train_dur").dt.total_hours() % 24).cast(pl.String)
                + "h "
                + (pl.col("train_dur").dt.total_minutes() % 60).cast(pl.String)
                + "m",
            )
            .sort(pl.col("train_dur"))
        )
