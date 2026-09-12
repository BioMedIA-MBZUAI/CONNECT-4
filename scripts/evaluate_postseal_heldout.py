#!/usr/bin/env python3
"""Run or verify fixed sealed CONNECT-4 post-seal evidence.

The staged production evaluator enforces a cohort-wide two-pass barrier: all 34
predictions are authenticated first, followed by the frozen execution/Stage-B
authority and all 34 native target pairs.  Metric/model modules are imported
only after that complete barrier succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any, Sequence

_EXPECTED_RUNTIME_FILES = (
    "architecture_contract.py",
    "eval/__init__.py",
    "eval/postseal.py",
    "eval/postseal_execution.py",
    "eval/postseal_metrics.py",
    "eval/quality.py",
    "scripts/__init__.py",
    "scripts/check_4d_quality.py",
    "scripts/evaluate_postseal_heldout.py",
    "scripts/run_postseal_heldout_evaluation.slurm",
    "scripts/visualize_4d_comparison.py",
    "scripts/visualize_texture_audit.py",
    "utils/__init__.py",
    "utils/source_provenance.py",
    "utils/spatial_detail.py",
)
_EXCLUDED_BAD_NODES = ("gpu-05", "gpu-50", "gpu-51", "gpu-56")


class BootstrapError(RuntimeError):
    """Raised before any CONNECT-4 package is allowed to be imported."""


def _authenticate_closed_dependency_runtime(
    arguments: argparse.Namespace, source_runtime: dict[str, Any]
) -> dict[str, Any]:
    """Authenticate a closed no-site dependency tree before project imports."""

    authority_argument = getattr(arguments, "dependency_runtime_authority", None)
    authority_pin_argument = getattr(
        arguments, "dependency_runtime_authority_sha256", None
    )
    if authority_argument is None or authority_pin_argument is None:
        raise BootstrapError(
            "closed dependency runtime authority v2 is required; site.main(), .pth "
            "execution, mutable import hooks, and source-only runtime v1 are forbidden"
        )
    authority = Path(authority_argument)
    if (
        not authority.is_absolute()
        or authority != Path(os.path.abspath(authority))
        or authority.is_symlink()
        or authority.resolve(strict=True) != authority
    ):
        raise BootstrapError(
            "dependency runtime authority path is not absolute/lexical"
        )
    authority_pin = _require_sha256(
        authority_pin_argument, label="dependency runtime authority pin"
    )
    authority_payload, authority_metadata = _read_stable_regular_file(
        authority,
        label="dependency runtime authority",
        require_single_link=True,
        maximum_bytes=256 * 1024,
    )
    if (
        hashlib.sha256(authority_payload).hexdigest() != authority_pin
        or stat.S_IMODE(authority_metadata.st_mode) != 0o444
    ):
        raise BootstrapError("dependency runtime authority bytes or mode differ")
    try:
        record = json.loads(authority_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("dependency runtime authority is not ASCII JSON") from exc
    expected_fields = {
        "schema",
        "status",
        "dependency_root",
        "tree_manifest",
        "import_roots",
        "python_executable_sha256",
        "site_enabled",
        "pth_execution",
        "environment_indirection",
        "record_sha256",
    }
    if type(record) is not dict or set(record) != expected_fields:
        raise BootstrapError("dependency runtime authority fields differ")
    unsigned = dict(record)
    recorded_sha256 = unsigned.pop("record_sha256", None)
    canonical = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    if (
        record.get("schema") != "connect4-postseal-closed-dependency-runtime-v2"
        or record.get("status") != "QUALIFIED_IMMUTABLE_NO_SITE_NO_PTH"
        or recorded_sha256 != canonical
        or record.get("site_enabled") is not False
        or record.get("pth_execution") is not False
        or record.get("environment_indirection") is not False
    ):
        raise BootstrapError("dependency runtime authority identity differs")
    if (
        record.get("python_executable_sha256")
        != source_runtime["python_executable"]["sha256"]
    ):
        raise BootstrapError("dependency runtime Python binding differs")
    raw_root = record.get("dependency_root")
    if type(raw_root) is not str or "$" in raw_root or "~" in raw_root:
        raise BootstrapError("dependency root uses indirection")
    root = Path(raw_root)
    if (
        not root.is_absolute()
        or root != Path(os.path.abspath(root))
        or root.is_symlink()
        or root.resolve(strict=True) != root
        or stat.S_IMODE(root.lstat().st_mode) != 0o555
    ):
        raise BootstrapError("dependency root is aliased or mutable")
    manifest_descriptor = record.get("tree_manifest")
    if type(manifest_descriptor) is not dict or set(manifest_descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise BootstrapError("dependency tree-manifest descriptor differs")
    manifest = Path(str(manifest_descriptor.get("path", "")))
    if (
        not manifest.is_absolute()
        or manifest.parent != root
        or manifest.is_symlink()
        or manifest.resolve(strict=True) != manifest
    ):
        raise BootstrapError("dependency tree manifest escaped its root")
    manifest_payload, manifest_metadata = _read_stable_regular_file(
        manifest,
        label="dependency tree manifest",
        require_single_link=True,
        maximum_bytes=64 * 1024 * 1024,
    )
    if (
        hashlib.sha256(manifest_payload).hexdigest()
        != _require_sha256(
            manifest_descriptor.get("sha256"),
            label="dependency tree-manifest SHA-256",
        )
        or manifest_metadata.st_size != manifest_descriptor.get("size_bytes")
        or stat.S_IMODE(manifest_metadata.st_mode) != 0o444
    ):
        raise BootstrapError("dependency tree manifest bytes or mode differ")
    try:
        manifest_text = manifest_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BootstrapError("dependency tree manifest is not ASCII") from exc
    bindings: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous = ""
    forbidden_names = {"sitecustomize.py", "usercustomize.py"}
    for line in manifest_text.splitlines():
        parts = line.split("  ")
        if len(parts) != 2:
            raise BootstrapError("dependency tree-manifest binding is malformed")
        digest = _require_sha256(parts[0], label="dependency file SHA-256")
        relative = parts[1]
        relative_path = Path(relative)
        if (
            not relative
            or relative <= previous
            or relative in seen
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.name.endswith(".pth")
            or relative_path.name in forbidden_names
        ):
            raise BootstrapError("dependency tree-manifest order/path differs")
        previous = relative
        seen.add(relative)
        path = root / relative_path
        if path.resolve(strict=True) != path:
            raise BootstrapError("dependency tree contains an alias")
        payload, metadata = _read_stable_regular_file(
            path,
            label=f"dependency file {relative}",
            require_single_link=True,
        )
        if (
            hashlib.sha256(payload).hexdigest() != digest
            or stat.S_IMODE(metadata.st_mode) != 0o444
        ):
            raise BootstrapError(f"dependency file changed: {relative}")
        bindings.append(
            {"relative_path": relative, "sha256": digest, "size_bytes": len(payload)}
        )
    if not bindings:
        raise BootstrapError("dependency tree manifest is empty")
    observed_files: set[str] = set()
    for directory, child_directories, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if (
            directory_path.is_symlink()
            or stat.S_IMODE(directory_path.lstat().st_mode) != 0o555
        ):
            raise BootstrapError(
                "dependency tree contains an aliased/mutable directory"
            )
        for child in child_directories:
            if (directory_path / child).is_symlink():
                raise BootstrapError("dependency tree contains a directory symlink")
        for file_name in file_names:
            file_path = directory_path / file_name
            if file_path.is_symlink():
                raise BootstrapError("dependency tree contains a file symlink")
            observed_files.add(file_path.relative_to(root).as_posix())
    if observed_files != seen | {manifest.name}:
        raise BootstrapError("dependency tree has a missing or extra file")
    raw_import_roots = record.get("import_roots")
    if type(raw_import_roots) is not list or not raw_import_roots:
        raise BootstrapError("dependency import roots are missing")
    import_roots: list[str] = []
    for raw_relative in raw_import_roots:
        if type(raw_relative) is not str:
            raise BootstrapError("dependency import-root type differs")
        relative = Path(raw_relative)
        candidate = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or candidate.resolve(strict=True) != candidate
            or not candidate.is_dir()
            or candidate.is_symlink()
        ):
            raise BootstrapError("dependency import root escaped its authority")
        import_roots.append(str(candidate))
    if len(import_roots) != len(set(import_roots)):
        raise BootstrapError("dependency import roots contain duplicates")
    return {
        "schema": "connect4-postseal-closed-dependency-runtime-evidence-v2",
        "authority": {
            "path": str(authority),
            "raw_sha256": authority_pin,
            "record_sha256": recorded_sha256,
            "size_bytes": authority_metadata.st_size,
        },
        "dependency_root": str(root),
        "tree_manifest": dict(manifest_descriptor),
        "file_count": len(bindings),
        "files_sha256": hashlib.sha256(
            json.dumps(
                bindings,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest(),
        "import_roots": import_roots,
        "site_enabled": False,
        "pth_execution": False,
        "environment_indirection": False,
    }


def _read_stable_regular_file(
    path: Path,
    *,
    label: str,
    require_single_link: bool,
    maximum_bytes: int | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read one regular file without following a final-component symlink."""
    try:
        initial = path.lstat()
    except OSError as exc:
        raise BootstrapError(f"{label} cannot be inspected") from exc
    if (
        stat.S_ISLNK(initial.st_mode)
        or not stat.S_ISREG(initial.st_mode)
        or (require_single_link and initial.st_nlink != 1)
    ):
        raise BootstrapError(f"{label} must be one regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BootstrapError(f"{label} cannot be opened safely") from exc
    chunks: list[bytes] = []
    size = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
            raise BootstrapError(f"{label} changed while opening")
        while block := os.read(descriptor, 1024 * 1024):
            size += len(block)
            if maximum_bytes is not None and size > maximum_bytes:
                raise BootstrapError(f"{label} exceeds its byte limit")
            chunks.append(block)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise BootstrapError(f"{label} changed after reading") from exc
    identity = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    if identity != (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    ) or identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    ):
        raise BootstrapError(f"{label} changed while reading")
    payload = b"".join(chunks)
    if len(payload) != initial.st_size:
        raise BootstrapError(f"{label} size changed while reading")
    return payload, initial


def _sha256(path: Path, *, label: str = "file") -> str:
    payload, _ = _read_stable_regular_file(
        path,
        label=label,
        require_single_link=False,
    )
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise BootstrapError(f"{label} is not a full SHA-256")
    return digest


def _parse_tres(value: str) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in value.split(",")
        if item.count("=") == 1
    }


def _memory_bytes(value: str) -> int | None:
    normalized = str(value).strip().upper()
    if normalized.endswith(("N", "C")):
        normalized = normalized[:-1]
    if not normalized:
        return None
    multipliers = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }
    suffix = normalized[-1]
    if suffix not in multipliers or not normalized[:-1].isdigit():
        return None
    return int(normalized[:-1]) * multipliers[suffix]


def _one_gpu_tres(tres: dict[str, str]) -> bool:
    general = tres.get("gres/gpu")
    typed = {key: value for key, value in tres.items() if key.startswith("gres/gpu:")}
    try:
        if general is not None:
            return int(general) == 1 and all(
                int(value) <= 1 for value in typed.values()
            )
        return bool(typed) and sum(int(value) for value in typed.values()) == 1
    except ValueError:
        return False


def _one_gpu_per_node(value: str) -> bool:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    gpu_tokens = [token for token in tokens if "gpu" in token.lower()]
    if len(gpu_tokens) != 1:
        return False
    token = gpu_tokens[0].replace("=", ":")
    pieces = token.split(":")
    return pieces[-1] == "1" and token.startswith(("gres:gpu:", "gres/gpu:"))


def _one_gpu_environment(value: str) -> bool:
    normalized = str(value).strip()
    if normalized == "1":
        return True
    pieces = normalized.replace("=", ":").split(":")
    return len(pieces) >= 2 and pieces[-1] == "1"


def _split_hostlist(value: str) -> list[str]:
    tokens: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                return []
        elif character == "," and depth == 0:
            tokens.append(value[start:index])
            start = index + 1
    if depth != 0:
        return []
    tokens.append(value[start:])
    return [token for token in tokens if token]


def _expand_fixed_slurm_hostlist(value: str) -> set[str]:
    """Expand the simple numeric hostlists used by CIAI exclusion evidence."""
    expanded: set[str] = set()
    for token in _split_hostlist(str(value).strip()):
        if "[" not in token:
            expanded.add(token)
            continue
        if token.count("[") != 1 or not token.endswith("]"):
            return set()
        prefix, members = token[:-1].split("[", 1)
        if not prefix or not members:
            return set()
        for member in members.split(","):
            if "-" in member:
                bounds = member.split("-", 1)
                if len(bounds) != 2 or not all(item.isdigit() for item in bounds):
                    return set()
                lower, upper = (int(item) for item in bounds)
                if upper < lower or upper - lower > 1024:
                    return set()
                width = max(len(item) for item in bounds)
                expanded.update(
                    f"{prefix}{number:0{width}d}" for number in range(lower, upper + 1)
                )
            elif member.isdigit():
                expanded.add(f"{prefix}{member}")
            else:
                return set()
    return expanded


def _require_ciai_gpu_allocation() -> dict[str, Any]:
    evidence = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_ntasks": os.environ.get("SLURM_NTASKS", ""),
        "slurm_job_num_nodes": os.environ.get("SLURM_JOB_NUM_NODES", ""),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE", ""),
        "hostname": socket.gethostname(),
    }
    job_id = evidence["slurm_job_id"]
    if not job_id.isdigit():
        raise BootstrapError("paired evaluation has no Slurm job allocation")
    try:
        query = subprocess.run(
            ["/usr/bin/scontrol", "show", "job", "-o", job_id],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        ).stdout.strip()
        gpu_inventory = subprocess.run(
            ["/usr/bin/nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BootstrapError(
            "CIAI scheduler/GPU allocation could not be authenticated"
        ) from exc
    fields = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in query.split()
        if "=" in item
    }
    evidence["slurm_qos"] = fields.get("QOS", "")
    evidence["scheduler_state"] = fields.get("JobState", "")
    evidence["scheduler_node_list"] = fields.get("NodeList", "")
    evidence["scheduler_excluded_nodes"] = fields.get("ExcNodeList", "")
    evidence["scheduler_tres"] = fields.get("TRES", fields.get("AllocTRES", ""))
    evidence["scheduler_tres_per_node"] = fields.get(
        "TresPerNode", fields.get("TRESPerNode", "")
    )
    evidence["gpu_inventory_sha256"] = hashlib.sha256(
        gpu_inventory.encode("utf-8")
    ).hexdigest()
    tres = _parse_tres(str(evidence["scheduler_tres"]))
    memory = tres.get("mem", "")
    exact_memory = _memory_bytes(memory) == 128 * 1024**3
    exact_minimum_memory = _memory_bytes(fields.get("MinMemoryNode", "")) == (
        128 * 1024**3
    )
    allocated_node = str(evidence["scheduler_node_list"])
    allocated_nodes = _expand_fixed_slurm_hostlist(allocated_node)
    allocated_short_names = {node.split(".", 1)[0] for node in allocated_nodes}
    excluded_nodes = _expand_fixed_slurm_hostlist(
        str(evidence["scheduler_excluded_nodes"])
    )
    excluded_short_names = {node.split(".", 1)[0] for node in excluded_nodes}
    cuda_devices = str(evidence["cuda_visible_devices"])
    if (
        fields.get("JobId") != job_id
        or fields.get("Partition") != "cscc-gpu-p"
        or evidence["slurm_qos"] != "cscc-gpu-qos"
        or evidence["scheduler_state"] != "RUNNING"
        or evidence["slurm_partition"] != "cscc-gpu-p"
        or fields.get("NumNodes") != "1"
        or fields.get("NumCPUs") != "16"
        or fields.get("NumTasks") != "1"
        or fields.get("CPUs/Task") != "16"
        or not exact_minimum_memory
        or not _one_gpu_per_node(str(evidence["scheduler_tres_per_node"]))
        or tres.get("cpu") != "16"
        or tres.get("node") != "1"
        or not _one_gpu_tres(tres)
        or not exact_memory
        or evidence["slurm_cpus_per_task"] != "16"
        or evidence["slurm_ntasks"] != "1"
        or evidence["slurm_job_num_nodes"] != "1"
        or not _one_gpu_environment(str(evidence["slurm_gpus_on_node"]))
        or not cuda_devices.strip()
        or "," in cuda_devices
        or "GPU " not in gpu_inventory
        or "login" in evidence["hostname"].lower()
        or not allocated_node
        or len(allocated_nodes) != 1
        or bool(allocated_short_names.intersection(_EXCLUDED_BAD_NODES))
        or not set(_EXCLUDED_BAD_NODES).issubset(excluded_short_names)
    ):
        raise BootstrapError(
            "paired evaluation requires the exact CIAI one-node/one-GPU, "
            "16-CPU/128-GiB cscc-gpu-p/cscc-gpu-qos allocation; login and "
            "excluded bad nodes are forbidden"
        )
    evidence["requested_excluded_bad_nodes"] = list(_EXCLUDED_BAD_NODES)
    evidence["allocated_node_not_excluded"] = True
    evidence["scheduler_excluded_bad_nodes_authenticated"] = True
    evidence["exact_one_gpu_tres_authenticated"] = True
    evidence["scontrol_output_sha256"] = hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()
    return evidence


def _authenticate_runtime(arguments: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CONNECT4_RUNTIME_PREIMPORT_AUTHENTICATED") != "1":
        raise BootstrapError(
            "immutable runtime was not authenticated by the shell boundary before import"
        )
    if os.environ.get("PYTHONPATH") is not None:
        raise BootstrapError("inherited PYTHONPATH was not scrubbed")
    if not sys.flags.isolated or not sys.flags.no_site or not sys.dont_write_bytecode:
        raise BootstrapError("Python must start with -I -S -B")
    if any(
        name == root or name.startswith(f"{root}.")
        for name in sys.modules
        for root in ("eval", "scripts", "utils")
    ):
        raise BootstrapError(
            "CONNECT-4 code was imported before runtime authentication"
        )
    manifest = arguments.runtime_manifest
    if not manifest.is_absolute() or manifest != Path(os.path.abspath(manifest)):
        raise BootstrapError("runtime manifest path is not absolute/lexical")
    if manifest.is_symlink() or manifest.resolve(strict=True) != manifest:
        raise BootstrapError("runtime manifest aliases another path")
    expected_manifest_sha256 = _require_sha256(
        arguments.runtime_manifest_sha256, label="runtime manifest pin"
    )
    manifest_payload, manifest_metadata = _read_stable_regular_file(
        manifest,
        label="runtime manifest",
        require_single_link=True,
        maximum_bytes=64 * 1024,
    )
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise BootstrapError("runtime manifest changed after shell authentication")
    if stat.S_IMODE(manifest_metadata.st_mode) != 0o444:
        raise BootstrapError("runtime manifest is not read-only mode 0444")
    root = manifest.parent
    root_metadata = root.lstat()
    if (
        root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root.resolve(strict=True) != root
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
    ):
        raise BootstrapError("runtime root is aliased or not read-only mode 0555")
    bindings = []
    seen = set()
    try:
        manifest_text = manifest_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BootstrapError("runtime manifest is not ASCII") from exc
    for line in manifest_text.splitlines():
        parts = line.split("  ")
        if len(parts) != 2:
            raise BootstrapError("runtime manifest binding is malformed")
        digest = _require_sha256(parts[0], label="runtime source binding")
        relative = parts[1]
        if relative in seen or relative not in _EXPECTED_RUNTIME_FILES:
            raise BootstrapError("runtime manifest file set differs")
        seen.add(relative)
        path = root / relative
        if path.resolve(strict=True) != path:
            raise BootstrapError(f"runtime source aliases another path: {relative}")
        payload, metadata = _read_stable_regular_file(
            path,
            label=f"runtime source {relative}",
            require_single_link=True,
            maximum_bytes=8 * 1024 * 1024,
        )
        if (
            stat.S_IMODE(metadata.st_mode) != 0o444
            or hashlib.sha256(payload).hexdigest() != digest
        ):
            raise BootstrapError(f"runtime source changed: {relative}")
        bindings.append(
            {
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": metadata.st_size,
            }
        )
    if tuple(item["relative_path"] for item in bindings) != _EXPECTED_RUNTIME_FILES:
        raise BootstrapError("runtime manifest order/set differs")
    observed_files: set[str] = set()
    for directory, child_directories, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_metadata = directory_path.lstat()
        if (
            stat.S_ISLNK(directory_metadata.st_mode)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o555
        ):
            raise BootstrapError("runtime tree has an aliased or mutable directory")
        for child in child_directories:
            child_path = directory_path / child
            if child_path.is_symlink():
                raise BootstrapError("runtime tree contains a symbolic link")
        for file_name in file_names:
            file_path = directory_path / file_name
            if file_path.is_symlink():
                raise BootstrapError("runtime tree contains a symbolic link")
            observed_files.add(file_path.relative_to(root).as_posix())
    if observed_files != {*_EXPECTED_RUNTIME_FILES, manifest.name}:
        raise BootstrapError("runtime tree has a missing or extra file")
    launcher_sha256 = _require_sha256(
        arguments.slurm_launcher_sha256, label="Slurm launcher pin"
    )
    launcher_binding = next(
        item
        for item in bindings
        if item["relative_path"] == "scripts/run_postseal_heldout_evaluation.slurm"
    )
    if launcher_binding["sha256"] != launcher_sha256:
        raise BootstrapError("Slurm launcher/runtime binding differs")
    cli_source = Path(os.path.abspath(__file__))
    if cli_source.is_symlink() or cli_source.resolve(strict=True) != cli_source:
        raise BootstrapError("CLI source aliases another path")
    if cli_source != root / "scripts" / "evaluate_postseal_heldout.py":
        raise BootstrapError("CLI did not execute from the staged runtime")
    cli_binding = next(
        item
        for item in bindings
        if item["relative_path"] == "scripts/evaluate_postseal_heldout.py"
    )
    if _sha256(cli_source, label="CLI source") != cli_binding["sha256"]:
        raise BootstrapError("executing CLI source differs from its runtime binding")
    expected_python_sha256 = _require_sha256(
        arguments.python_executable_sha256,
        label="Python executable pin",
    )
    python_entry = Path(sys.executable)
    if not python_entry.is_absolute():
        raise BootstrapError("Python executable path is not absolute")
    python_executable = python_entry.resolve(strict=True)
    if not python_executable.is_file() or python_executable.is_symlink():
        raise BootstrapError("resolved Python executable is not a regular file")
    python_payload, python_metadata = _read_stable_regular_file(
        python_executable,
        label="Python executable",
        require_single_link=False,
        maximum_bytes=256 * 1024 * 1024,
    )
    observed_python_sha256 = hashlib.sha256(python_payload).hexdigest()
    if observed_python_sha256 != expected_python_sha256:
        raise BootstrapError("Python executable differs from its external pin")
    return {
        "schema": "connect4-postseal-immutable-runtime-evidence-v1",
        "runtime_root": str(root),
        "manifest": {
            "path": str(manifest),
            "sha256": expected_manifest_sha256,
            "size_bytes": manifest_metadata.st_size,
        },
        "slurm_launcher_sha256": launcher_sha256,
        "python_executable": {
            "invoked_path": str(python_entry),
            "resolved_path": str(python_executable),
            "sha256": observed_python_sha256,
            "size_bytes": python_metadata.st_size,
        },
        "files": bindings,
        "preimport_shell_authentication": True,
        "runtime_tree_read_only_modes": True,
        "python_executable_sha256_authenticated": True,
        "python_isolated_mode": True,
        "python_no_site_bootstrap": True,
        "python_bytecode_writes_disabled": True,
        "connect4_imports_absent_during_authentication": True,
        "inherited_pythonpath_scrubbed": True,
    }


def _add_hash_bindings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prediction-set-seal-sha256", required=True)
    parser.add_argument("--synthesis-checkpoint-sha256", required=True)
    parser.add_argument("--training-run-artifact-identity-sha256", required=True)
    parser.add_argument("--training-target-artifact-identities-sha256", required=True)
    parser.add_argument("--spatial-authority-sha256", required=True)


def _add_runtime_bindings(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--slurm-launcher-sha256", required=True)
    parser.add_argument("--python-executable-sha256", required=True)
    parser.add_argument("--dependency-runtime-authority", type=Path)
    parser.add_argument("--dependency-runtime-authority-sha256")
    parser.add_argument("--execution-authority", type=Path)
    parser.add_argument("--execution-authority-sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prediction-root", type=Path, required=True)
    evaluate.add_argument("--stage-b-completed-set", type=Path, required=True)
    evaluate.add_argument("--stage-b-completed-set-sha256", required=True)
    _add_runtime_bindings(evaluate)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--fps", type=int, default=8)
    evaluate.add_argument("--slimbrain-authority", type=Path)
    evaluate.add_argument("--slimbrain-authority-sha256")
    _add_hash_bindings(evaluate)
    verify = commands.add_parser("verify-publication")
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--expected-evaluation-sha256", required=True)
    verify.add_argument("--expected-publication-receipt-sha256", required=True)
    verify.add_argument("--expected-prediction-set-seal-path", required=True)
    verify.add_argument("--expected-prediction-set-seal-sha256", required=True)
    verify.add_argument("--expected-stage-b-completed-set-path", required=True)
    verify.add_argument("--expected-stage-b-completed-set-sha256", required=True)
    verify.add_argument("--expected-stage-b-verifier-source-path", required=True)
    verify.add_argument("--expected-stage-b-verifier-source-sha256", required=True)
    verify.add_argument(
        "--expected-stage-b-verifier-dependency-path",
        action="append",
        required=True,
    )
    verify.add_argument(
        "--expected-stage-b-verifier-dependency-sha256",
        action="append",
        required=True,
    )
    verify.add_argument("--slimbrain-authority-sha256")
    _add_runtime_bindings(verify)
    return parser


def _pins(arguments: argparse.Namespace, evaluation_pins: type[Any]) -> Any:
    return evaluation_pins(
        prediction_set_seal_sha256=arguments.prediction_set_seal_sha256,
        synthesis_checkpoint_sha256=arguments.synthesis_checkpoint_sha256,
        training_run_artifact_identity_sha256=(
            arguments.training_run_artifact_identity_sha256
        ),
        training_target_artifact_identities_sha256=(
            arguments.training_target_artifact_identities_sha256
        ),
        spatial_authority_sha256=arguments.spatial_authority_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    immutable_runtime = _authenticate_runtime(arguments)
    if arguments.command == "evaluate":
        _require_ciai_gpu_allocation()
    closed_dependency_runtime = _authenticate_closed_dependency_runtime(
        arguments, immutable_runtime
    )
    runtime_root = Path(str(immutable_runtime["runtime_root"]))
    sys.path[:0] = [str(runtime_root), *closed_dependency_runtime["import_roots"]]
    from eval.postseal import (  # pylint: disable=import-outside-toplevel
        EvaluationPins,
        run_postseal_evaluation,
        verify_evaluation_publication,
    )

    if arguments.command == "verify-publication":
        record = verify_evaluation_publication(
            arguments.output_root,
            expected_evaluation_sha256=arguments.expected_evaluation_sha256,
            expected_receipt_sha256=(arguments.expected_publication_receipt_sha256),
            expected_prediction_set_seal_path=(
                arguments.expected_prediction_set_seal_path
            ),
            expected_prediction_set_seal_sha256=(
                arguments.expected_prediction_set_seal_sha256
            ),
            expected_stage_b_completed_set_path=(
                arguments.expected_stage_b_completed_set_path
            ),
            expected_stage_b_completed_set_sha256=(
                arguments.expected_stage_b_completed_set_sha256
            ),
            expected_stage_b_verifier_source_path=(
                arguments.expected_stage_b_verifier_source_path
            ),
            expected_stage_b_verifier_source_sha256=(
                arguments.expected_stage_b_verifier_source_sha256
            ),
            expected_stage_b_verifier_dependency_paths=(
                arguments.expected_stage_b_verifier_dependency_path
            ),
            expected_stage_b_verifier_dependency_sha256s=(
                arguments.expected_stage_b_verifier_dependency_sha256
            ),
            expected_execution_authority_path=str(arguments.execution_authority),
            expected_execution_authority_sha256=(arguments.execution_authority_sha256),
            expected_slimbrain_authority_sha256=(arguments.slimbrain_authority_sha256),
        )
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "scan_count": record["scan_count"],
                    "record_sha256": record["record_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if (arguments.slimbrain_authority is None) != (
        arguments.slimbrain_authority_sha256 is None
    ):
        raise BootstrapError(
            "SLIM-Brain authority path and external SHA-256 pin must be supplied together"
        )
    record = run_postseal_evaluation(
        prediction_root=arguments.prediction_root,
        pins=_pins(arguments, EvaluationPins),
        stage_b_completed_set=arguments.stage_b_completed_set,
        stage_b_completed_set_sha256=arguments.stage_b_completed_set_sha256,
        output_root=arguments.output_root,
        execution_authority_path=arguments.execution_authority,
        execution_authority_sha256=arguments.execution_authority_sha256,
        fps=arguments.fps,
        slimbrain_authority=arguments.slimbrain_authority,
        slimbrain_authority_sha256=arguments.slimbrain_authority_sha256,
        slimbrain_dependency_authority_sha256=(
            closed_dependency_runtime["authority"]["raw_sha256"]
            if arguments.slimbrain_authority is not None
            else None
        ),
        slimbrain_python_executable_sha256=(
            immutable_runtime["python_executable"]["sha256"]
            if arguments.slimbrain_authority is not None
            else None
        ),
        slimbrain_device=(
            "cuda" if arguments.slimbrain_authority is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "status": record["status"],
                "scan_count": record["scan_count"],
                "record_sha256": record["record_sha256"],
                "output_root": str(arguments.output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
