import time


class LapTimer:
    """Tracks lap times and race progress."""

    def __init__(self):
        self._current_lap = 1
        self._lap_start_time = None
        self._current_lap_time = 0.0
        self._best_lap_time = float('inf')
        self._last_lap_time = 0.0
        self._total_race_time = 0.0
        self._race_started = False
        self._lap_times_history = []

    def start_race(self):
        """Start race timer."""
        self._lap_start_time = time.time()
        self._race_started = True
        self._current_lap = 1
        self._total_race_time = 0.0
        self._lap_times_history = []

    def update(self):
        """Update current lap time."""
        if self._race_started and self._lap_start_time:
            self._current_lap_time = time.time() - self._lap_start_time
            self._total_race_time += self._current_lap_time

    def complete_lap(self):
        """Complete current lap and start new one."""
        if not self._race_started:
            return None

        lap_time = self._current_lap_time
        self._last_lap_time = lap_time
        self._lap_times_history.append(lap_time)

        is_best = False
        if lap_time < self._best_lap_time and self._current_lap > 0:
            self._best_lap_time = lap_time
            is_best = True

        lap_info = {
            'lap_number': self._current_lap,
            'time': lap_time,
            'is_best': is_best
        }

        # Start new lap
        self._current_lap += 1
        self._lap_start_time = time.time()
        self._current_lap_time = 0.0

        return lap_info

    def reset(self):
        """Reset all timer values."""
        self._current_lap = 1
        self._lap_start_time = None
        self._current_lap_time = 0.0
        self._best_lap_time = float('inf')
        self._last_lap_time = 0.0
        self._total_race_time = 0.0
        self._race_started = False
        self._lap_times_history = []

    @property
    def current_lap(self):
        return self._current_lap

    @property
    def current_lap_time(self):
        return self._current_lap_time

    @property
    def best_lap_time(self):
        return self._best_lap_time if self._best_lap_time != float('inf') else 0.0

    @property
    def last_lap_time(self):
        return self._last_lap_time

    @property
    def lap_history(self):
        return self._lap_times_history.copy()

    def format_time(self, seconds):
        """Format time as MM:SS.mmm"""
        if seconds == 0 or seconds == float('inf'):
            return "--:--.---"

        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

