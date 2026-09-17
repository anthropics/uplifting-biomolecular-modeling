"""Runtime context for Genie3 CLI invocations."""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from types import TracebackType
from typing import Any

from genie3.config import ExperimentRunConfig, load_experiment_config
from genie3.runtime.logging import close_logger, configure_run_logger
from genie3.runtime.profile import RuntimeProfile
from genie3.runtime.progress import ProgressReporter, set_active_reporter


@dataclass(frozen=True)
class RunInvocation:
    """User-facing invocation options shared by CLI commands."""

    config_path: Path
    command: str
    verbose: bool = False
    log_dir: Path | None = None


@dataclass(frozen=True)
class RunLoggingConfig:
    """Resolved logging destinations for a run."""

    log_root: Path
    run_dir: Path
    log_file: Path
    metadata_file: Path
    workers_dir: Path


@dataclass
class RunContext(AbstractContextManager["RunContext"]):
    """Shared per-command runtime state."""

    invocation: RunInvocation
    config: ExperimentRunConfig
    logging_config: RunLoggingConfig
    logger: logging.Logger
    progress: ProgressReporter
    profile: RuntimeProfile
    _root_handlers: list[logging.Handler] | None = None
    _root_level: int | None = None
    _root_file_handler: logging.Handler | None = None
    _stdout: TextIOWrapper | None = None
    _stderr: TextIOWrapper | None = None
    _captured_stream: TextIOWrapper | None = None
    _env_updates: dict[str, str | None] | None = None

    @property
    def run_dir(self) -> Path:
        return self.logging_config.run_dir

    @property
    def log_file(self) -> Path:
        return self.logging_config.log_file

    def __enter__(self) -> "RunContext":
        self._configure_quiet_runtime_environment()
        set_active_reporter(self.progress)
        if not self.invocation.verbose:
            self._start_console_capture()
        return self

    def write_metadata(self, *, status: str, error: str | None = None) -> None:
        """Write metadata for this invocation."""
        metadata: dict[str, Any] = {
            "command": self.invocation.command,
            "config_path": str(self.invocation.config_path),
            "verbose": self.invocation.verbose,
            "log_root": str(self.logging_config.log_root),
            "run_dir": str(self.logging_config.run_dir),
            "log_file": str(self.logging_config.log_file),
            "workers_dir": str(self.logging_config.workers_dir),
            "status": status,
            "experiment": {
                "name": self.config.experiment.name,
                "seed": self.config.experiment.seed,
            },
            "paths": {
                "rootdir": self.config.paths.rootdir,
                "dataset": self.config.paths.dataset,
            },
            "runtime": {
                "num_devices": self.config.runtime.num_devices,
                "device": self.config.runtime.device,
                "workers": self.config.runtime.workers,
            },
            "profile": self.profile.to_dict(),
        }
        if error is not None:
            metadata["error"] = error

        with self.logging_config.metadata_file.open("w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        del exc_type, traceback
        self.progress.clear_status()
        if exc_value is None:
            self.logger.info("🎉 %s completed", self.invocation.command.capitalize())
            self.write_metadata(status="completed")
        else:
            self.logger.exception("❌ %s failed", self.invocation.command.capitalize())
            self.write_metadata(status="failed", error=str(exc_value))
        self.progress.close()
        self._stop_console_capture()
        self._restore_runtime_environment()
        set_active_reporter(None)
        close_logger(self.logger)
        return None

    def _configure_quiet_runtime_environment(self) -> None:
        """Reduce third-party runtime chatter in default terminal mode."""
        updates: dict[str, str | None] = {}
        for key, value in {
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "GENIE3_RUN_DIR": str(self.logging_config.run_dir),
            "GENIE3_WORKERS_DIR": str(self.logging_config.workers_dir),
            "GENIE3_RUN_LOG": str(self.logging_config.log_file),
            "GENIE3_VERBOSE": "1" if self.invocation.verbose else "0",
        }.items():
            updates[key] = os.environ.get(key)
            os.environ[key] = value
        self._env_updates = updates

    def _restore_runtime_environment(self) -> None:
        """Restore environment variables temporarily changed for this run."""
        if self._env_updates is None:
            return
        for key, previous in self._env_updates.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        self._env_updates = None

    def _start_console_capture(self) -> None:
        """Route third-party console noise to the run log in non-verbose mode."""
        root_logger = logging.getLogger()
        self._root_handlers = list(root_logger.handlers)
        self._root_level = root_logger.level

        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        self._root_file_handler = logging.FileHandler(self.log_file)
        self._root_file_handler.setLevel(logging.DEBUG)
        self._root_file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        root_logger.addHandler(self._root_file_handler)
        root_logger.setLevel(logging.INFO)

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._captured_stream = self.log_file.open("a")
        sys.stdout = self._captured_stream
        sys.stderr = self._captured_stream

    def _stop_console_capture(self) -> None:
        """Restore root logging and raw console streams after capture."""
        if self._captured_stream is not None:
            self._captured_stream.flush()

        if self._stdout is not None:
            sys.stdout = self._stdout
            self._stdout = None
        if self._stderr is not None:
            sys.stderr = self._stderr
            self._stderr = None

        root_logger = logging.getLogger()
        if self._root_file_handler is not None:
            self._root_file_handler.flush()
            self._root_file_handler.close()
            root_logger.removeHandler(self._root_file_handler)
            self._root_file_handler = None

        if self._root_handlers is not None:
            for handler in self._root_handlers:
                root_logger.addHandler(handler)
            self._root_handlers = None
        if self._root_level is not None:
            root_logger.setLevel(self._root_level)
            self._root_level = None

        if self._captured_stream is not None:
            self._captured_stream.close()
            self._captured_stream = None


def create_run_context(
    *,
    command: str,
    config_path: str,
    verbose: bool = False,
    log_dir: str | None = None,
) -> RunContext:
    """Create a run context with log directory, logger, metadata, and profile."""
    config = load_experiment_config(config_path)
    resolved_config_path = Path(config.config_path)
    reused_run_dir = os.environ.get("GENIE3_RUN_DIR")
    log_root = Path(log_dir or "logs/runs").expanduser().resolve()
    if reused_run_dir:
        run_dir = Path(reused_run_dir).expanduser().resolve()
        log_root = run_dir.parent
    else:
        run_dir = _create_run_dir(log_root=log_root, command=command, experiment=config.experiment.name)
    log_file = run_dir / "run.log"
    metadata_file = run_dir / ("metadata.json" if not reused_run_dir else f"{command}_metadata.json")
    workers_dir = run_dir / "workers"
    workers_dir.mkdir(exist_ok=True)

    logger = configure_run_logger(log_file=log_file, verbose=verbose)
    invocation = RunInvocation(
        config_path=resolved_config_path,
        command=command,
        verbose=verbose,
        log_dir=Path(log_dir).expanduser().resolve() if log_dir is not None else None,
    )
    logging_config = RunLoggingConfig(
        log_root=log_root,
        run_dir=run_dir,
        log_file=log_file,
        metadata_file=metadata_file,
        workers_dir=workers_dir,
    )
    context = RunContext(
        invocation=invocation,
        config=config,
        logging_config=logging_config,
        logger=logger,
        progress=ProgressReporter(),
        profile=RuntimeProfile(),
    )
    context.write_metadata(status="started")
    if not reused_run_dir:
        logger.info("📝 Detailed log: %s", log_file)
    logger.debug("Run directory: %s", run_dir)
    logger.debug("Config path: %s", resolved_config_path)
    return context


def _create_run_dir(*, log_root: Path, command: str, experiment: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_experiment = _safe_path_part(experiment)
    base_name = f"{timestamp}_{_safe_path_part(command)}_{safe_experiment}"

    log_root.mkdir(parents=True, exist_ok=True)
    for suffix in [None, *range(2, 1000)]:
        name = base_name if suffix is None else f"{base_name}_{suffix}"
        run_dir = log_root / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique run directory under {log_root}")


def _safe_path_part(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.strip().lower()
    )
    return cleaned.strip("-") or "run"
