import logging
from rich.logging import RichHandler

LOGGING_FORMAT = "%(name)s - %(levelname)s - %(message)s"


def maybe_initialize_root_logger():
    """
    Note: this is no-op if someone already called basicConfig() before.
    """
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
        format=LOGGING_FORMAT,
    )


maybe_initialize_root_logger()

activations_file_handler = logging.FileHandler("activations_debug.log")
activations_file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))


def initialize_activations_logger():
    """
    Activations logger is available to call any time, but by default
    actual activation prints are disabled (i.e. only errors are printed.)

    To enable activations, call `enable_activations_logging()`.
    """

    # Essentially "turned off" by default as activations use info/debug levels.
    # We still allow errors via activation logger, though.
    level = logging.ERROR

    logger = logging.getLogger("activations_logger")
    logger.setLevel(level)

    if activations_file_handler not in logger.handlers:
        # Attach to root logger to make sure file captures all logs, not only
        # activations. Easier to correlate and debug.
        logging.getLogger().addHandler(activations_file_handler)
        # By default, make the file only save errors. This reduces file size
        # growth when activations debugging is disabled.
        activations_file_handler.setLevel(level)
    return logger


activations_logger = initialize_activations_logger()


def enable_activations_logging():
    # This enables [up to] debug-level printouts to the console.
    activations_logger.setLevel(logging.DEBUG)
    # And to the file, too. This is needed since the file is attached to the
    # root logger, not to activations_logger. (And attaching to root logger was
    # needed to capture all logs into the activations file.)
    activations_file_handler.setLevel(logging.DEBUG)
