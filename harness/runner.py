"""harness.runner — Execute a compiled kernel."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from harness.models import RunResult, Status
from harness.spec_loader import resolve_paths

log = logging.getLogger(__name__)


def _parse_gnu_time(output: str) -> float | None:
    """Parse /usr/bin/time -v output and return user+system time in seconds.

    Parameters
    ----------
    output:
        Full text of the ``/usr/bin/time -v`` stderr/file output.

    Returns
    -------
    float | None
        Sum of ``User time (seconds)`` and ``System time (seconds)`` lines,
        or ``None`` if parsing fails.
    """
    user_time: float | None = None
    system_time: float | None = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("User time (seconds):"):
            try:
                user_time = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("System time (seconds):"):
            try:
                system_time = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    if user_time is None or system_time is None:
        return None
    return user_time + system_time


def run_spec(
    spec: dict[str, Any],
    project_root: Path,
    configuration: str = "correctness",
    *,
    verbose: bool = False,
    measure_cpu_time: bool = False,
) -> RunResult:
    """Run a compiled kernel for a single input configuration.

    Parameters
    ----------
    spec:
        Parsed spec dict.
    project_root:
        Absolute path to the *parbench/* project root.
    configuration:
        Name of the entry in ``run.input_configurations`` to use
        (e.g. ``"correctness"`` or ``"performance"``).
    verbose:
        If *True*, log stdout/stderr.
    measure_cpu_time:
        If *True*, prefix the command with ``/usr/bin/time -v`` to capture
        user+system CPU time.  The result is stored in
        ``RunResult.cpu_time_seconds``.  Requires GNU time (Linux).
        Defaults to *False* to preserve existing behaviour.

    Returns
    -------
    RunResult
    """
    resolved = resolve_paths(spec, project_root)["_resolved"]
    working_dir: Path = resolved["working_dir"]

    run_section = spec.get("run", {})
    input_configs = run_section.get("input_configurations", {})
    timeout_seconds = run_section.get("timeout_seconds", 300)

    # Look up the requested configuration
    config = input_configs.get(configuration)
    if config is None:
        available = list(input_configs.keys())
        return RunResult(
            status=Status.ERROR,
            configuration=configuration,
            duration_seconds=0.0,
            exit_code=-1,
            stdout="",
            stderr=(
                f"Configuration '{configuration}' not found in spec. "
                f"Available: {available}"
            ),
        )

    # Build the command as a list (no shell=True for run)
    executable = run_section.get("executable", "./main")
    exe_path = working_dir / executable

    arguments: list[str] = config.get("arguments", [])
    # Use the spec's executable name (e.g. "./heartwall.out") as argv[0],
    # not the resolved absolute path. On POSIX, subprocess resolves relative
    # paths in cmd[0] against cwd. This avoids bugs in programs that parse
    # argv[0] (heartwall's main.c:71 starts at i=0, matching 'h' in /home/...).
    cmd: list[str] = [executable] + [str(a) for a in arguments]

    # Environment
    env = os.environ.copy()
    env_vars = run_section.get("environment_variables") or {}
    env.update(env_vars)

    # Optionally wrap with /usr/bin/time -v for CPU timing
    time_output_file: str | None = None
    if measure_cpu_time:
        # Create a temp file for /usr/bin/time -v to write its report into.
        # We use -o <file> so that the timing output does not interleave with
        # the program's own stderr.
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".gnu_time", delete=False
        )
        tmp.close()
        time_output_file = tmp.name
        cmd = ["/usr/bin/time", "-v", "-o", time_output_file] + cmd

    log.debug("Running: %s  (cwd=%s, timeout=%ds)", cmd, working_dir, timeout_seconds)

    start = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            shell=False,
            cwd=str(working_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.monotonic() - start

        if verbose:
            if proc.stdout:
                log.info("[stdout] %s", proc.stdout.rstrip())
            if proc.stderr:
                log.info("[stderr] %s", proc.stderr.rstrip())

        status = Status.PASS if proc.returncode == 0 else Status.FAIL

        # Parse CPU time from /usr/bin/time -v output file
        cpu_time: float | None = None
        if time_output_file is not None:
            try:
                time_text = Path(time_output_file).read_text(encoding="utf-8")
                cpu_time = _parse_gnu_time(time_text)
                if verbose and cpu_time is not None:
                    log.info("[cpu_time] %.3fs", cpu_time)
            except OSError:
                pass
            finally:
                try:
                    os.unlink(time_output_file)
                except OSError:
                    pass

        return RunResult(
            status=status,
            configuration=configuration,
            duration_seconds=round(elapsed, 3),
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            cpu_time_seconds=round(cpu_time, 6) if cpu_time is not None else None,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        if time_output_file is not None:
            try:
                os.unlink(time_output_file)
            except OSError:
                pass
        return RunResult(
            status=Status.TIMEOUT,
            configuration=configuration,
            duration_seconds=round(elapsed, 3),
            exit_code=-1,
            stdout="",
            stderr=f"Execution timed out after {timeout_seconds}s",
        )
    except FileNotFoundError:
        elapsed = time.monotonic() - start
        if time_output_file is not None:
            try:
                os.unlink(time_output_file)
            except OSError:
                pass
        return RunResult(
            status=Status.ERROR,
            configuration=configuration,
            duration_seconds=round(elapsed, 3),
            exit_code=-1,
            stdout="",
            stderr=f"Executable not found: {exe_path}",
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        if time_output_file is not None:
            try:
                os.unlink(time_output_file)
            except OSError:
                pass
        return RunResult(
            status=Status.ERROR,
            configuration=configuration,
            duration_seconds=round(elapsed, 3),
            exit_code=-1,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


def run_all_configurations(
    spec: dict[str, Any],
    project_root: Path,
    *,
    verbose: bool = False,
    measure_cpu_time: bool = False,
) -> dict[str, RunResult]:
    """Run every input configuration defined in the spec.

    Parameters
    ----------
    spec:
        Parsed spec dict.
    project_root:
        Absolute path to the project root.
    verbose:
        Forward to :func:`run_spec`.
    measure_cpu_time:
        Forward to :func:`run_spec`.

    Returns
    -------
    dict[str, RunResult]:
        Mapping of ``configuration_name`` → ``RunResult``.
    """
    run_section = spec.get("run", {})
    input_configs = run_section.get("input_configurations", {})

    results: dict[str, RunResult] = {}
    for config_name in input_configs:
        results[config_name] = run_spec(
            spec,
            project_root,
            configuration=config_name,
            verbose=verbose,
            measure_cpu_time=measure_cpu_time,
        )
    return results
