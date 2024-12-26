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

from typing import Literal, Sequence, TypeVar

from rouge_score import rouge_scorer, scoring

from mblm.data.utils.bytes import Bytes

T = TypeVar("T")


def token_accuracy(a: Sequence[T], b: Sequence[T], allow_var_lens: bool) -> float:
    if len(a) != len(b) and not allow_var_lens:
        raise ValueError("Sequences must have the same length")
    if len(a) == 0 or len(b) == 0:
        # opinionated edge case
        return 0
    correct = 0
    for item_a, item_b in zip(a, b):
        correct += int(item_a == item_b)
    return correct / len(a)


rouge_scorers = dict(
    rouge1=rouge_scorer.RougeScorer(["rouge1"]), rougeL=rouge_scorer.RougeScorer(["rougeL"])
)


def rouge_score_from_bytes(
    target: list[int], predicted: list[int], which: Literal["rouge1", "rougeL"]
) -> float:
    try:
        t = Bytes.byte_list_to_str(target)
        p = Bytes.byte_list_to_str(predicted)
        score: scoring.Score = rouge_scorers[which].score(t, p)[which]
        return score.fmeasure
    except Exception:
        return -1
