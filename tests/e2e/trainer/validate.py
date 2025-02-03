from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

import polars as pl

from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericOutputConfig,
)
from mblm.utils.io import load_yml


class TrainOutputConfig(GenericOutputConfig[CoreModelParams, CoreTrainConfig, CoreIoConfig]):
    pass


def ensure_no_error_logs(log_file: Path):
    with Path.open(log_file) as log:
        for line in log.readlines():
            _, _, level, msg = line.split(" - ")
            if level == "CRITICAL" or level == "ERROR":
                raise AssertionError(msg)


def assert_on_model_run_output(output_dir: Path) -> None:
    checkpoints = list(output_dir.rglob("*.pth"))
    csv_loss_file = output_dir / "loss.csv"
    yml_config_file = output_dir / "config.yaml"
    log_file = output_dir / "train.log"

    assert csv_loss_file.exists(), "Expected a CSV loss file"
    assert yml_config_file.exists(), "Expected a YAML config file"
    assert log_file.exists(), "Expected a log file"

    ensure_no_error_logs(log_file)

    run_config = load_yml(yml_config_file, parse_to=TrainOutputConfig)

    assert len(checkpoints) == run_config.io.num_models_to_save, (
        f"Expected {run_config.io.num_models_to_save} model checkpoints"
    )

    # assert that we get the specified number of training loss logs
    csv_log = pl.read_csv(csv_loss_file)

    num_train_loss_entries = csv_log.filter(pl.col("kind") == "train").select(pl.len()).item()
    assert num_train_loss_entries == run_config.io.log_train_loss_amount, (
        f"Expected {run_config.io.log_train_loss_amount} train loss entries, received {num_train_loss_entries}"
    )

    # assert that we get the specified number of validation loss logs
    num_valid_loss_entries = csv_log.filter(pl.col("kind") == "valid").select(pl.len()).item()
    assert num_valid_loss_entries == run_config.io.validate_amount, (
        f"Expected {run_config.io.validate_amount} validation loss entries, received {num_valid_loss_entries}"
    )

    # assert that we the test loss is logged
    num_test_loss_entries = csv_log.filter(pl.col("kind") == "test").select(pl.len()).item()
    assert num_test_loss_entries == 1

    # assert we log valid dates in the summary
    try:
        datetime.fromisoformat(run_config.summary.training_start)
        datetime.fromisoformat(run_config.summary.training_end)
    except ValueError:
        raise AssertionError("Failed to parse training start/end dates")


def assert_on_chained_logs(csv_loss_files: list[Path], assert_equal_epochs: bool):
    for file_idx in range(0, len(csv_loss_files) - 1):
        file_1 = csv_loss_files[file_idx]
        file_2 = csv_loss_files[file_idx + 1]

        prev = pl.read_csv(file_1).filter(pl.col("kind") == "train")
        curr = pl.read_csv(file_2).filter(pl.col("kind") == "train")

        # every single run was exactly one epoch
        if assert_equal_epochs:
            # this test also ensures the dfs are the same length
            prev.select(pl.col("epoch") + 1).equals(
                curr.select(pl.col("epoch")),
            )

        assert prev.select(pl.col("cum_batch").max() + 1).equals(
            curr.select(pl.col("cum_batch").min())
        ), "Training did not exactly resume from previous training"

        assert prev.select(pl.col("elements_seen").sum()).equals(
            curr.select(pl.col("elements_seen").sum())
        ), "Same configuration saw different number of train items"

        for df in [prev, curr]:
            # the chained test is designed to run for around 3 epochs, adjust if necessary
            assert df.select(pl.col("epoch").max()).item() < 4, "Expected less than 3 epochs"


def assert_on_grad_acc(csv_loss_files: list[Path]):
    for file_idx in range(0, len(csv_loss_files) - 1):
        file_1 = csv_loss_files[file_idx]
        file_2 = csv_loss_files[file_idx + 1]

        prev = pl.read_csv(file_1).sort("timestamp")
        curr = pl.read_csv(file_2).sort("timestamp")

        # make sure same loss on test
        assert (
            curr.select(pl.col("loss").last().round(3)).item()
            == prev.select(pl.col("loss").last().round(3)).item()
        )


def run_test():
    parser = ArgumentParser()
    parser.add_argument("--check-output", type=Path, dest="check_output")
    parser.add_argument("--check-chained-csv", action="append", type=Path, dest="check_chained_csv")
    parser.add_argument(
        "--check-grad-acc-csv", action="append", type=Path, dest="check_grad_acc_csv"
    )
    parser.add_argument("--assert-equal-epochs", action=BooleanOptionalAction, default=False)
    args = parser.parse_args()

    check_output: Path | None = args.check_output
    check_chained_csv: list[Path] | None = args.check_chained_csv
    check_grad_acc_csv: list[Path] | None = args.check_grad_acc_csv
    assert_equal_epochs: bool = args.assert_equal_epochs

    if check_output and check_chained_csv and check_grad_acc_csv:
        raise AssertionError(
            "Speficy either --check-output, --check-chained-log or --check-grad-acc"
        )
    if check_output:
        print(f"Validating {check_output}")
        return assert_on_model_run_output(check_output)
    if check_chained_csv:
        print(f"Validating {len(check_chained_csv)} chained runs")
        return assert_on_chained_logs(check_chained_csv, assert_equal_epochs)
    if check_grad_acc_csv:
        print("Asserting on gradient accumulation")
        return assert_on_grad_acc(check_grad_acc_csv)
    raise AssertionError("No tests ran")


if __name__ == "__main__":
    run_test()
