"""harness.builder — Compile a kernel from its spec."""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from harness.models import BuildResult, Status
from harness.spec_loader import resolve_paths

log = logging.getLogger(__name__)

# Default timeout for build commands (10 minutes)
BUILD_TIMEOUT_SECONDS = 600


def _substitute_variables(command: str, variables: dict[str, Any] | None) -> str:
    """Replace ``${VAR}`` placeholders in *command* with default values.

    Parameters
    ----------
    command:
        Raw build command string (may contain ``${VAR}`` tokens).
    variables:
        The ``build.variables`` dict from the spec.  Each key maps to an
        object with at least a ``default`` field.

    Returns
    -------
    str:
        Command with substitutions applied.
    """
    if not variables:
        return command

    def _replacer(m: re.Match[str]) -> str:
        var_name = m.group(1)
        var_def = variables.get(var_name)
        if var_def is not None and isinstance(var_def, dict):
            return var_def.get("default", m.group(0))
        return m.group(0)

    return re.sub(r"\$\{(\w+)\}", _replacer, command)


def build_spec(
    spec: dict[str, Any],
    project_root: Path,
    *,
    timeout: int = BUILD_TIMEOUT_SECONDS,
    verbose: bool = False,
) -> BuildResult:
    """Build a single kernel according to its spec.

    Steps
    -----
    1. Resolve ``working_directory`` to an absolute path.
    2. Run ``commands.clean`` (ignore errors).
    3. If ``commands.configure`` exists, run it.
    4. Substitute ``${VARIABLE}`` tokens in ``commands.build``.
    5. Run ``commands.build`` via :func:`subprocess.run`.
    6. Check that ``outputs.executable`` exists after build.
    7. Return a :class:`BuildResult`.

    Parameters
    ----------
    spec:
        Parsed spec dict.
    project_root:
        Absolute path to the *parbench/* project root.
    timeout:
        Max seconds for each subprocess call.
    verbose:
        If *True*, log stdout/stderr for every command.

    Returns
    -------
    BuildResult
    """
    resolved = resolve_paths(spec, project_root)["_resolved"]
    working_dir: Path = resolved["working_dir"]
    build = spec.get("build", {})
    commands = build.get("commands", {})
    variables = build.get("variables")
    outputs = build.get("outputs", {})
    executable_rel = outputs.get("executable", "")

    # Collect combined stdout / stderr across all steps
    all_stdout: list[str] = []
    all_stderr: list[str] = []

    if not working_dir.exists():
        msg = f"Working directory does not exist: {working_dir}"
        log.error(msg)
        return BuildResult(
            status=Status.ERROR,
            duration_seconds=0.0,
            stdout="",
            stderr=msg,
        )

    # Environment variables from run section (some builds also need them)
    env_vars = spec.get("run", {}).get("environment_variables") or {}

    start = time.monotonic()

    try:
        # ---- clean (optional, errors ignored) ----
        clean_cmd = commands.get("clean")
        if clean_cmd:
            log.debug("Running clean: %s", clean_cmd)
            _run_shell(
                clean_cmd,
                working_dir,
                env_vars,
                timeout,
                verbose,
                all_stdout,
                all_stderr,
                ignore_errors=True,
            )

        # ---- configure (optional) ----
        configure_cmd = commands.get("configure")
        if configure_cmd:
            log.debug("Running configure: %s", configure_cmd)
            rc = _run_shell(
                configure_cmd,
                working_dir,
                env_vars,
                timeout,
                verbose,
                all_stdout,
                all_stderr,
            )
            if rc != 0:
                elapsed = time.monotonic() - start
                return BuildResult(
                    status=Status.FAIL,
                    duration_seconds=round(elapsed, 3),
                    stdout="\n".join(all_stdout),
                    stderr="\n".join(all_stderr),
                )

        # ---- build ----
        build_cmd = commands.get("build", "")
        if not build_cmd:
            elapsed = time.monotonic() - start
            return BuildResult(
                status=Status.ERROR,
                duration_seconds=round(elapsed, 3),
                stdout="",
                stderr="No build command specified in spec",
            )

        build_cmd = _substitute_variables(build_cmd, variables)
        log.debug("Running build: %s", build_cmd)
        rc = _run_shell(
            build_cmd, working_dir, env_vars, timeout, verbose, all_stdout, all_stderr
        )

        elapsed = time.monotonic() - start

        if rc != 0:
            return BuildResult(
                status=Status.FAIL,
                duration_seconds=round(elapsed, 3),
                stdout="\n".join(all_stdout),
                stderr="\n".join(all_stderr),
            )

        # ---- check executable exists ----
        exe_path = working_dir / executable_rel if executable_rel else None
        if exe_path and not exe_path.exists():
            return BuildResult(
                status=Status.FAIL,
                duration_seconds=round(elapsed, 3),
                stdout="\n".join(all_stdout),
                stderr=(
                    "\n".join(all_stderr)
                    + f"\nExpected executable not found: {exe_path}"
                ),
            )

        return BuildResult(
            status=Status.PASS,
            duration_seconds=round(elapsed, 3),
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr),
            executable_path=str(exe_path) if exe_path else None,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return BuildResult(
            status=Status.TIMEOUT,
            duration_seconds=round(elapsed, 3),
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr) + "\nBuild timed out",
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        return BuildResult(
            status=Status.ERROR,
            duration_seconds=round(elapsed, 3),
            stdout="\n".join(all_stdout),
            stderr="\n".join(all_stderr) + f"\n{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _run_shell(
    command: str,
    cwd: Path,
    env_vars: dict[str, str],
    timeout: int,
    verbose: bool,
    stdout_acc: list[str],
    stderr_acc: list[str],
    *,
    ignore_errors: bool = False,
) -> int:
    """Run a shell command, appending output to accumulators.

    Returns the process exit code (0 on success).
    """
    import os

    env = os.environ.copy()
    env.update(env_vars)

    proc = subprocess.run(
        command,
        shell=True,  # Build commands may contain pipes / &&
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if proc.stdout:
        stdout_acc.append(proc.stdout)
        if verbose:
            log.info("[stdout] %s", proc.stdout.rstrip())
    if proc.stderr:
        stderr_acc.append(proc.stderr)
        if verbose:
            log.info("[stderr] %s", proc.stderr.rstrip())

    if proc.returncode != 0 and not ignore_errors:
        log.warning("Command exited %d: %s", proc.returncode, command)

    return proc.returncode
