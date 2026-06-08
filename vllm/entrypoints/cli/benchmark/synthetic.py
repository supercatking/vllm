# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from vllm.benchmarks.synthetic import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkSyntheticSubcommand(BenchmarkSubcommandBase):
    """The `synthetic` subcommand for `vllm bench`."""

    name = "synthetic"
    help = "Benchmark host scheduler latency with synthetic GPU execution."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
