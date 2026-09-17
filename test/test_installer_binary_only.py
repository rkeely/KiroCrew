"""The installers resolve dependencies from prebuilt wheels only.

Left to itself, pip treats a dependency with no wheel for the host as something
to BUILD from its sdist. On an old-glibc distro the newest numpy / Pillow wheels
carry a manylinux floor the host does not meet, so pip compiled them -- and that
needs GCC >= 10 and libjpeg headers the host was never required to have. The
failure surfaces deep inside a compiler run, after ``cli.sh`` has already moved
the working venv aside, and the transactional rebuild rolls back on every retry.

``--only-binary=:all:`` changes both halves: pip resolves the newest release of
each dependency that publishes a wheel the host can run, and when no release
does, it fails BEFORE any build starts with ``No matching distribution found``,
which the installers turn into a supported-platform message. These tests drive
the real ``cli.sh`` through the signed-manifest harness with the install step
faked, so they pin the pip/pipx invocation and the failure report rather than a
paraphrase of either.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from installer_test_helpers import run_bounded
from test_cli_manifest_signature import (  # noqa: F401  (fixtures are looked up by name)
    CDN_BASE,
    WHEEL_NAME,
    SigningKey,
    _build_manifest,
    _openssl_bin,
    _openssl_on_path,
    _patched_installer,
    _stage_cdn,
    test_key,
)

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "cli.sh"
INSTALL_SH = ROOT / "install.sh"


@pytest.fixture(scope="module")
def signing_key(test_key: SigningKey) -> SigningKey:  # noqa: F811 -- pytest fixture by name
    """The manifest harness's signing key, under a name no test parameter shadows."""
    return test_key


ONLY_BINARY = "--only-binary=:all:"
OPT_IN_ENV = "KIROCREW_ALLOW_SOURCE_BUILDS"

# pip's exact wording when binary-only resolution finds no usable release, as
# printed for a dependency the wheel pulls in (captured from a real run).
NO_WHEEL_LOG = (
    "ERROR: Could not find a version that satisfies the requirement numpy>=1.21,<3 "
    "(from kirocrew) (from versions: 1.26.0, 1.26.4, 2.0.2, 2.2.6)\n"
    "ERROR: No matching distribution found for numpy>=1.21,<3\n"
)
NETWORK_LOG = (
    "WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) "
    "after connection broken by 'NewConnectionError': /simple/numpy/\n"
    "ERROR: Could not install packages due to an OSError: [Errno 101] Network is unreachable\n"
)
# The same two lines when the index carries NO release of the package (a
# private mirror missing it): not a platform verdict.
NO_RELEASE_LOG = (
    "ERROR: Could not find a version that satisfies the requirement numpy>=1.21,<3 "
    "(from kirocrew) (from versions: none)\n"
    "ERROR: No matching distribution found for numpy>=1.21,<3\n"
)

_FAKE_CURL = """#!/bin/sh
set -eu
out=""
url=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
case "$url" in
  https://fixtures.invalid/*) rel=${url#https://fixtures.invalid/} ;;
  *) echo "unexpected URL: $url" >&2; exit 9 ;;
esac
[ -n "$out" ] || exit 10
cp "$FAKE_CDN_ROOT/$rel" "$out"
"""

# Records every argument (one per line) and, when FAKE_PIP_FAIL_WITH names a
# file, replays that file as pip's output and fails -- the shape of a real pip
# error reaching the installer's failure branch. Like real pipx (its install
# path calls venv.remove_venv() on any exception), a failed install DELETES
# the package venv; a successful one lays a fresh venv down.
_FAKE_PIPX = """#!/bin/sh
set -eu
case "$1" in
  install)
    printf '%s\\n' "$@" > "$FAKE_ARGV_FILE"
    rm -rf "$FAKE_PIPX_VENVS/kirocrew"
    if [ -n "${FAKE_PIP_FAIL_WITH:-}" ]; then cat "$FAKE_PIP_FAIL_WITH" >&2; exit 1; fi
    mkdir -p "$FAKE_PIPX_VENVS/kirocrew/bin"
    printf 'home = fresh\\n' > "$FAKE_PIPX_VENVS/kirocrew/pyvenv.cfg"
    printf 'fresh\\n' > "$FAKE_PIPX_VENVS/kirocrew/bin/kirocrew" ;;
  environment)
    case "${3:-}" in
      PIPX_LOCAL_VENVS) printf '%s\\n' "$FAKE_PIPX_VENVS" ;;
      *) printf '%s\\n' "$HOME/.local/bin" ;;
    esac ;;
  *) exit 11 ;;
esac
"""

_FAKE_PIP = """#!/bin/sh
set -eu
# The best-effort `pip install --upgrade pip` refresh is not the install step.
case " $* " in
  *" --upgrade pip "*) exit 0 ;;
esac
printf '%s\\n' "$@" > "$FAKE_ARGV_FILE"
if [ -n "${FAKE_PIP_FAIL_WITH:-}" ]; then cat "$FAKE_PIP_FAIL_WITH" >&2; exit 1; fi
"""

# A `python3` that answers `-m venv DIR` by laying out a venv whose pip is the
# recorder above, and hands every other invocation (the signature and digest
# checks, the symlink helper) to the real interpreter. cli.sh's venv branch is
# otherwise unreachable without a network-facing pip.
_FAKE_PYTHON = """#!/bin/sh
set -eu
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "venv" ]; then
  mkdir -p "$3/bin"
  printf 'home = %s\\n' "$FAKE_REAL_PYTHON" > "$3/pyvenv.cfg"
  cp "$FAKE_PIP_SCRIPT" "$3/bin/pip"
  chmod 755 "$3/bin/pip"
  ln -sf "$FAKE_REAL_PYTHON" "$3/bin/python"
  ln -sf "$FAKE_REAL_PYTHON" "$3/bin/python3"
  exit 0
fi
exec "$FAKE_REAL_PYTHON" "$@"
"""


def _write_tools(root: Path, *, with_pipx: bool) -> tuple[Path, Path]:
    """PATH prefix for one installer run: fake curl, optional fake pipx, and an
    interpreter ladder that either points at the real interpreter (pipx branch)
    or at the venv-faking wrapper (venv branch)."""
    tools = root / "tools"
    tools.mkdir(parents=True)
    argv_file = root / "install-argv"
    (tools / "curl").write_text(_FAKE_CURL, encoding="utf-8")
    (tools / "curl").chmod(0o755)
    if with_pipx:
        (tools / "pipx").write_text(_FAKE_PIPX, encoding="utf-8")
        (tools / "pipx").chmod(0o755)
    pip_script = root / "fake-pip"
    pip_script.write_text(_FAKE_PIP, encoding="utf-8")
    pip_script.chmod(0o755)
    # Shadow EVERY candidate in cli.sh's ladder (same reasoning as the manifest
    # harness: a host shim under an empty HOME wedges instead of answering).
    for name in ("python3.13", "python3.12", "python3"):
        if with_pipx:
            (tools / name).symlink_to(sys.executable)
        else:
            (tools / name).write_text(_FAKE_PYTHON, encoding="utf-8")
            (tools / name).chmod(0o755)
    return tools, argv_file


def _run_installer(
    case: Path,
    key: SigningKey,
    *,
    with_pipx: bool,
    fail_with: str | None = None,
    extra_env: dict[str, str] | None = None,
    existing_pipx_venv: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    if os.name == "nt":
        pytest.skip("cli.sh is supported on macOS and Linux only")
    case.mkdir(exist_ok=True)
    wheel = case / WHEEL_NAME
    wheel.write_bytes(b"verified wheel")
    manifest = _build_manifest(case, key, wheel)
    cdn = _stage_cdn(case, manifest, wheel)
    script = _patched_installer(case, key)
    run_root = case / "run"
    tools, argv_file = _write_tools(run_root, with_pipx=with_pipx)
    pipx_venvs = run_root / "pipx-venvs"
    pipx_venvs.mkdir()
    if existing_pipx_venv:
        # A working install from an earlier run: pyvenv.cfg is what cli.sh
        # keys on, bin/kirocrew is the launcher pipx's symlink points at.
        (pipx_venvs / "kirocrew" / "bin").mkdir(parents=True)
        (pipx_venvs / "kirocrew" / "pyvenv.cfg").write_text("home = old\n", encoding="utf-8")
        (pipx_venvs / "kirocrew" / "bin" / "kirocrew").write_text("old\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop(OPT_IN_ENV, None)
    # A closed PATH: the fakes, the openssl the harness resolved (linked in on
    # its own -- its directory may be a package-manager prefix that also holds
    # pipx), and the system directories that hold sh/awk/sed/tar. A host pipx
    # would otherwise win `command -v pipx` and turn the venv-branch cases into
    # pipx-branch runs.
    openssl = shutil.which("openssl")
    assert openssl is not None, "the manifest harness needs openssl on PATH"
    (tools / "openssl").symlink_to(openssl)
    path = os.pathsep.join([str(tools), "/usr/bin", "/bin", "/usr/sbin", "/sbin"])
    if not with_pipx and shutil.which("pipx", path=path) is not None:
        pytest.skip("a system-directory pipx shadows cli.sh's venv branch on this host")
    env.update(
        {
            "PATH": path,
            "HOME": str(run_root / "home"),
            "KIROCREW_HOME": str(run_root / "data-home"),
            "FAKE_CDN_ROOT": str(cdn),
            "FAKE_ARGV_FILE": str(argv_file),
            "FAKE_PIPX_VENVS": str(pipx_venvs),
            "FAKE_PIP_SCRIPT": str(run_root / "fake-pip"),
            "FAKE_REAL_PYTHON": sys.executable,
        }
    )
    if fail_with is not None:
        log = run_root / "pip-failure.txt"
        log.write_text(fail_with, encoding="utf-8")
        env["FAKE_PIP_FAIL_WITH"] = str(log)
    if extra_env:
        env.update(extra_env)
    result = run_bounded(["sh", str(script), "--cdn", CDN_BASE], env, cwd=str(run_root))
    argv = argv_file.read_text(encoding="utf-8").splitlines() if argv_file.exists() else []
    return result, argv


# ---------------------------------------------------------------------------
# The invocation: binary-only reaches pip on both install branches
# ---------------------------------------------------------------------------


def test_pipx_branch_forwards_binary_only_to_pip(tmp_path: Path, signing_key: SigningKey) -> None:
    result, argv = _run_installer(tmp_path / "case", signing_key, with_pipx=True)

    assert result.returncode == 0, result.stderr
    assert argv[0] == "install"
    # pipx does not resolve dependencies itself; --pip-args is the only way the
    # policy reaches the pip it drives, and it must be ONE argument.
    assert f"--pip-args={ONLY_BINARY}" in argv, argv
    assert argv[-1].endswith(WHEEL_NAME), argv


def test_venv_branch_passes_binary_only_before_the_wheel(
    tmp_path: Path, signing_key: SigningKey
) -> None:
    result, argv = _run_installer(tmp_path / "case", signing_key, with_pipx=False)

    assert result.returncode == 0, result.stderr
    assert argv[:2] == ["install", "--quiet"], argv
    assert argv[-2] == ONLY_BINARY, argv
    assert argv[-1].endswith(WHEEL_NAME), argv
    # `$PIP_BINARY_ONLY` is expanded unquoted so that an empty value vanishes;
    # the non-empty value must still arrive as exactly one word.
    assert argv.count(ONLY_BINARY) == 1


@pytest.mark.parametrize("with_pipx", [True, False], ids=["pipx", "venv"])
def test_the_opt_in_restores_the_compile_fallback(
    tmp_path: Path, signing_key: SigningKey, with_pipx: bool
) -> None:
    """KIROCREW_ALLOW_SOURCE_BUILDS=1 removes the flag and adds nothing in its
    place: no empty `--pip-args=` for pipx, no empty word for pip."""
    result, argv = _run_installer(
        tmp_path / "case", signing_key, with_pipx=with_pipx, extra_env={OPT_IN_ENV: "1"}
    )

    assert result.returncode == 0, result.stderr
    assert not any(ONLY_BINARY in word for word in argv), argv
    assert not any(word.startswith("--pip-args") for word in argv), argv
    assert "" not in argv, argv
    assert argv[-1].endswith(WHEEL_NAME), argv


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_only_the_literal_one_opts_in(tmp_path: Path, signing_key: SigningKey, value: str) -> None:
    result, argv = _run_installer(
        tmp_path / "case", signing_key, with_pipx=True, extra_env={OPT_IN_ENV: value}
    )

    assert result.returncode == 0, result.stderr
    assert f"--pip-args={ONLY_BINARY}" in argv, argv


# ---------------------------------------------------------------------------
# The failure report: a missing wheel names the platform and the way out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_pipx", [True, False], ids=["pipx", "venv"])
def test_no_wheel_failure_names_platform_packages_and_remedy(
    tmp_path: Path, signing_key: SigningKey, with_pipx: bool
) -> None:
    result, _argv = _run_installer(
        tmp_path / "case", signing_key, with_pipx=with_pipx, fail_with=NO_WHEEL_LOG
    )

    assert result.returncode == 1
    err = result.stderr
    # pip's own words are replayed first, so nothing the user could act on is hidden.
    assert "No matching distribution found for numpy>=1.21,<3" in err
    # The report names the host, the packages and both remedies.
    assert "no prebuilt wheel exists for this platform" in err
    # The same `uname` the installer's child shell runs, so an arch-translated
    # shell (Rosetta) cannot make the expectation disagree with the report.
    uname = subprocess.run(
        ["uname", "-s", "-m"], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.strip()
    assert uname in err, (uname, err)
    assert "for: numpy>=1.21,<3" in err
    assert "never compiles a dependency" in err
    assert "not a supported platform" in err
    assert f"{OPT_IN_ENV}=1" in err
    # And still ends on the branch's own failure line, so the exit path is unchanged.
    assert "kirocrew-install: installing the wheel" in err
    assert "Installed kirocrew" not in result.stdout


@pytest.mark.parametrize("with_pipx", [True, False], ids=["pipx", "venv"])
@pytest.mark.parametrize(
    ("log", "echoed"),
    [(NETWORK_LOG, "Network is unreachable"), (NO_RELEASE_LOG, "from versions: none")],
    ids=["network", "no-release"],
)
def test_other_pip_failures_keep_the_generic_message(
    tmp_path: Path, signing_key: SigningKey, with_pipx: bool, log: str, echoed: str
) -> None:
    """A network error, or an index that carries no release of the package at
    all, is not a platform verdict: the tail is replayed, the supported-platform
    paragraph stays out, and the run still fails."""
    result, _argv = _run_installer(
        tmp_path / "case", signing_key, with_pipx=with_pipx, fail_with=log
    )

    assert result.returncode == 1
    assert echoed in result.stderr
    assert "no prebuilt wheel exists" not in result.stderr
    assert "not a supported platform" not in result.stderr
    assert "kirocrew-install: installing the wheel" in result.stderr


def test_the_no_wheel_report_is_silent_when_compiling_was_opted_in(
    tmp_path: Path, signing_key: SigningKey
) -> None:
    """Under the opt-in the same pip text does not mean "no wheel": pip builds
    an sdist there, so a `No matching distribution` is a real resolution
    failure and the platform paragraph would mislead."""
    result, _argv = _run_installer(
        tmp_path / "case",
        signing_key,
        with_pipx=True,
        fail_with=NO_WHEEL_LOG,
        extra_env={OPT_IN_ENV: "1"},
    )

    assert result.returncode == 1
    assert "No matching distribution found for numpy>=1.21,<3" in result.stderr
    assert "no prebuilt wheel exists" not in result.stderr


def test_venv_failure_still_restores_the_previous_install(
    tmp_path: Path, signing_key: SigningKey
) -> None:
    """The new capture-and-report path sits inside the transactional rebuild:
    a binary-only refusal must leave the pre-rebuild venv back in place, exactly
    as any other pip failure does."""
    case = tmp_path / "case"
    data_home = case / "run" / "data-home"
    venv = case / "run" / "data-home-venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /previous\n", encoding="utf-8")
    (venv / "bin").mkdir()
    (venv / "bin" / "kirocrew").write_text("#!/bin/sh\necho previous\n", encoding="utf-8")
    data_home.mkdir(parents=True)

    result, _argv = _run_installer(case, signing_key, with_pipx=False, fail_with=NO_WHEEL_LOG)

    assert result.returncode == 1
    assert "The previous install was restored" in result.stderr
    assert (venv / "pyvenv.cfg").read_text(encoding="utf-8") == "home = /previous\n"
    assert not list(case.glob("run/data-home-venv.pre-rebuild.*"))


def test_pipx_failure_restores_the_previous_install(
    tmp_path: Path, signing_key: SigningKey
) -> None:
    """pipx's own install path deletes the package venv on any failure, so a
    binary-only refusal over an existing install would otherwise take the
    working `kirocrew` down with it. The installer moves the venv aside first
    and puts it back when pipx fails."""
    case = tmp_path / "case"
    result, _argv = _run_installer(
        case, signing_key, with_pipx=True, fail_with=NO_WHEEL_LOG, existing_pipx_venv=True
    )

    venv = case / "run" / "pipx-venvs" / "kirocrew"
    assert result.returncode == 1
    assert "The previous install was restored" in result.stderr
    assert (venv / "pyvenv.cfg").read_text(encoding="utf-8") == "home = old\n"
    assert (venv / "bin" / "kirocrew").read_text(encoding="utf-8") == "old\n"
    assert not list(case.glob("run/pipx-venvs/kirocrew.pre-rebuild.*"))


def test_pipx_success_drops_the_backup(tmp_path: Path, signing_key: SigningKey) -> None:
    """The move-aside is a transaction, not a copy: once pipx has built the new
    venv the backup is removed and only the fresh tree remains."""
    case = tmp_path / "case"
    result, _argv = _run_installer(case, signing_key, with_pipx=True, existing_pipx_venv=True)

    venv = case / "run" / "pipx-venvs" / "kirocrew"
    assert result.returncode == 0, result.stderr
    assert (venv / "pyvenv.cfg").read_text(encoding="utf-8") == "home = fresh\n"
    assert not list(case.glob("run/pipx-venvs/kirocrew.pre-rebuild.*"))


def test_pipx_first_install_has_nothing_to_back_up(tmp_path: Path, signing_key: SigningKey) -> None:
    """No existing venv: no move-aside, no restore wording, the plain failure."""
    case = tmp_path / "case"
    result, _argv = _run_installer(case, signing_key, with_pipx=True, fail_with=NO_WHEEL_LOG)

    assert result.returncode == 1
    assert "The previous install was restored" not in result.stderr
    assert "installing the wheel with pipx failed." in result.stderr
    assert not list(case.glob("run/pipx-venvs/*"))


# ---------------------------------------------------------------------------
# install.sh: the same policy on the editable install's dependency resolution
# ---------------------------------------------------------------------------


def _install_sh_pip_line() -> str:
    body = INSTALL_SH.read_text(encoding="utf-8")
    lines = [ln for ln in body.splitlines() if '"$_venv/bin/pip" install' in ln and " -e " in ln]
    assert len(lines) == 1, lines
    return lines[0]


def test_install_sh_editable_install_is_binary_only() -> None:
    line = _install_sh_pip_line()
    assert "$_pip_binary_only" in line, line
    body = INSTALL_SH.read_text(encoding="utf-8")
    assert f'_pip_binary_only="{ONLY_BINARY}"' in body
    assert f'"${{{OPT_IN_ENV}:-0}}" = "1"' in body
    # The flag precedes `-e`: it is a resolution option, not part of the target.
    assert line.index("$_pip_binary_only") < line.index(" -e ")


def test_install_sh_flag_expands_to_one_word_or_nothing() -> None:
    """The variable is expanded unquoted on purpose (a quoted empty value would
    hand pip a bare "" argument); the non-empty value has no whitespace to split."""
    line = _install_sh_pip_line()
    assert re.search(r"\s\$_pip_binary_only\s", line), line
    assert shlex.split(ONLY_BINARY) == [ONLY_BINARY]


def test_install_sh_reports_a_missing_wheel_as_a_platform_verdict() -> None:
    body = INSTALL_SH.read_text(encoding="utf-8")
    report = body.split("No matching distribution found for", 1)
    assert len(report) == 2, "install.sh does not classify pip's no-wheel failure"
    tail = report[1]
    assert "No prebuilt wheel exists for this platform" in tail
    assert "uname -s" in tail and "uname -m" in tail
    assert f"{OPT_IN_ENV}=1" in tail
    # The verdict is gated on the policy being ON, since under the opt-in the same
    # pip text is an ordinary resolution failure.
    gate = body[: body.index("No prebuilt wheel exists")].rsplit("if ", 1)[1]
    assert '-n "$_pip_binary_only"' in gate
    # ... and on releases being listed: "(from versions: none)" is the index
    # lacking the package, which the platform paragraph must not claim.
    assert "from versions: [0-9]" in gate


def test_cli_sh_help_documents_the_opt_in() -> None:
    help_text = INSTALLER.read_text(encoding="utf-8").split("cat <<'EOF'", 1)[1].split("EOF", 1)[0]
    assert f"{OPT_IN_ENV}=1" in help_text
