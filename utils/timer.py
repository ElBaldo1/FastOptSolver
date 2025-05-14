import time
from typing import Optional

class Timer:
    """A context manager for timing code execution using time.perf_counter.

    Attributes
    ----------
    elapsed : Optional[float]
        The elapsed time in seconds between entering and exiting the context.
        None if the context is still active or hasn't been entered.

    Examples
    --------
    >>> with Timer() as timer:
    ...     # Code to time goes here
    ...     pass
    >>> print(f"Elapsed time: {timer.elapsed:.6f} seconds")
    """

    def __init__(self):
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> 'Timer':
        """Start the timer when entering the context.

        Returns
        -------
        Timer
            The Timer instance itself.
        """
        self._start_time = time.perf_counter()
        self._end_time = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer when exiting the context."""
        self._end_time = time.perf_counter()

    @property
    def elapsed(self) -> Optional[float]:
        """Get the elapsed time in seconds.

        Returns
        -------
        Optional[float]
            The elapsed time in seconds if the timer has completed,
            None otherwise.
        """
        if self._start_time is None:
            return None
        end_time = self._end_time if self._end_time is not None else time.perf_counter()
        return end_time - self._start_time