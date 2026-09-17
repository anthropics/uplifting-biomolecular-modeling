"""stand-in jax.monitoring (tests only): listeners are registered and never called (nothing compiles here)."""
_events, _durations = [], []


def register_event_listener(f):
    _events.append(f)


def register_event_duration_secs_listener(f):
    _durations.append(f)
