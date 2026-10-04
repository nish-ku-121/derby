from collections import deque

from pipeline.run_sweep import MAX_STDERR_TAIL, _append_output_tail


def test_output_tail_keeps_tail_of_single_oversized_line() -> None:
    tail: deque[str] = deque()
    size = _append_output_tail(tail, 0, "x" * (MAX_STDERR_TAIL * 2))

    assert size == MAX_STDERR_TAIL
    assert "".join(tail) == "x" * MAX_STDERR_TAIL
