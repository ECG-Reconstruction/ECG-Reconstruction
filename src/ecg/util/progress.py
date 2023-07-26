"""Defines the `ProgressCounter` class for tracking the progress of a task."""

import datetime
import io
import sys
import time
from collections.abc import Iterable, Iterator
from typing import Any, Generic, Optional, TypeVar

_T = TypeVar("_T")


class ProgressCounter(Generic[_T]):
    """
    A counter that wraps an iterable and tracks the progress of a task.

    We define this custom class instead of using `tqdm` because the progress bars take
    much space and are visually distracting.
    """

    def __init__(
        self,
        iterable: Iterable[_T],
        prefix: str,
        length: Optional[int] = None,
        stream: io.TextIOBase = sys.stderr,
    ) -> None:
        """
        Initializes the progress counter.

        Args
            iterable: An iterable to wrap.
            prefix: A prefix to print before the progress counter.
            length: If not `None`, it is the number of items to be taken from the
              iterable.
            stream: The stream to print the progress counter to.
        """
        self._length = len(iterable) if length is None else length
        self._iterator = iter(iterable)
        self._prefix = prefix
        self._stream = stream
        self.postfix: dict[str, Any] = {}
        """The key-value pairs to print after the progress counter."""

    def __len__(self) -> int:
        """Returns the number of items to be yielded by the progress counter."""
        return self._length

    def __iter__(self) -> Iterator[_T]:
        """Yields the items of the wrapped iterable while tracking the progress."""
        previous_line_width: Optional[int] = None

        def write_line(line: str) -> None:
            nonlocal previous_line_width
            if previous_line_width is None:
                self._stream.write(line)
            else:
                self._stream.write("\r" + line.ljust(previous_line_width))
            previous_line_width = len(line)
            self._stream.flush()

        def format_postfix_value(value):
            if isinstance(value, float):
                return f"{value:.4g}"
            return value

        t_start = time.time()
        write_line(
            f"{self._prefix} 0/{self._length} [{datetime.timedelta(seconds=0)}<?, ?s/it]"
        )
        for i, item in zip(range(1, self._length + 1), self._iterator):
            yield item
            t_current = time.time()
            t_elapsed = t_current - t_start
            seconds_per_it = t_elapsed / i
            t_remaining = seconds_per_it * (self._length - i)
            message = (
                f"{self._prefix} {i}/{self._length} "
                f"[{datetime.timedelta(seconds=round(t_elapsed))}<"
                f"{datetime.timedelta(seconds=round(t_remaining))}, "
                f"{seconds_per_it:.3g}s/it"
            )
            if self.postfix:
                message += "".join(
                    f", {k}={format_postfix_value(v)}" for k, v in self.postfix.items()
                )
            message += "]"
            write_line(message)
        self._stream.write("\n")
        self._stream.flush()
