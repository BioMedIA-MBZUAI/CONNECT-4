#!/usr/bin/env python3
"""Atomically stage the minimal immutable CONNECT-4 post-seal runtime.

This utility opens every source with ``O_NOFOLLOW``, rejects hard links and
mutation while copying, writes an exact SHA-256 manifest, makes the staged tree
read-only, and publishes the directory with an atomic no-replace rename.  It
does not import CONNECT-4 code and never accesses predictions or targets.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile


RUNTIME_MANIFEST = "postseal_runtime.sha256"
RUNTIME_FILES = (
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


class RuntimeStagingError(RuntimeError):
    """Raised when the immutable evaluator runtime cannot be staged safely."""


class RuntimePublicationUncertainError(RuntimeStagingError):
    """Raised after the no-replace rename committed but durability is uncertain.

    The destination is deliberately preserved for forensic review.  Callers
    must never treat this as an ordinary pre-commit staging failure or reuse
    the destination.
    """


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise RuntimeStagingError(f"{label} must be an absolute lexical path")
    metadata = requested.lstat()
    resolved = requested.resolve(strict=True)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeStagingError(f"{label} must be a non-symlink directory")
    if resolved != requested:
        raise RuntimeStagingError(f"{label} aliases another path")
    return requested


def _destination(path: Path) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute() or requested != Path(os.path.abspath(requested)):
        raise RuntimeStagingError("destination must be an absolute lexical path")
    parent = _canonical_existing_directory(requested.parent, label="destination parent")
    if requested.parent != parent or requested.name in {"", ".", ".."}:
        raise RuntimeStagingError("destination aliases another path")
    try:
        requested.lstat()
    except FileNotFoundError:
        return requested
    raise FileExistsError(f"runtime destination already exists: {requested}")


def _copy_stable_file(source: Path, destination: Path) -> tuple[str, int]:
    initial = source.lstat()
    if (
        not stat.S_ISREG(initial.st_mode)
        or stat.S_ISLNK(initial.st_mode)
        or initial.st_nlink != 1
    ):
        raise RuntimeStagingError(f"invalid runtime source file: {source}")
    read_descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    write_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(read_descriptor)
        if (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino):
            raise RuntimeStagingError(f"runtime source changed while opening: {source}")
        write_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        while block := os.read(read_descriptor, 1024 * 1024):
            digest.update(block)
            size += len(block)
            view = memoryview(block)
            while view:
                written = os.write(write_descriptor, view)
                view = view[written:]
        os.fsync(write_descriptor)
        final = os.fstat(read_descriptor)
        current = source.lstat()
        identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
        )
        if identity != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ) or identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise RuntimeStagingError(f"runtime source mutated while copying: {source}")
    finally:
        if write_descriptor >= 0:
            os.close(write_descriptor)
        os.close(read_descriptor)
    if size != initial.st_size:
        raise RuntimeStagingError(f"runtime source size changed: {source}")
    return digest.hexdigest(), size


def _write_manifest(path: Path, bindings: list[tuple[str, str]]) -> str:
    payload = "".join(
        f"{digest}  {relative}\n" for relative, digest in sorted(bindings)
    ).encode("ascii")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _tree_record(
    relative_path: str,
    metadata: os.stat_result,
    *,
    kind: str,
    sha256: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "relative_path": relative_path,
        "kind": kind,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": stat.S_IMODE(metadata.st_mode),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
        "link_count": int(metadata.st_nlink),
    }
    if sha256 is not None:
        record["sha256"] = sha256
    return record


def _capture_runtime_tree(root: Path) -> tuple[dict[str, object], ...]:
    """Capture one descriptor-anchored, mutation-stable runtime inventory."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(root, flags)
    records: list[dict[str, object]] = []

    def capture_directory(directory_descriptor: int, relative: str) -> None:
        initial = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(initial.st_mode):
            raise RuntimeStagingError(f"runtime entry is not a directory: {relative}")
        records.append(_tree_record(relative, initial, kind="directory"))
        with os.scandir(directory_descriptor) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            if name in {"", ".", ".."} or "/" in name:
                raise RuntimeStagingError("runtime tree contains an invalid entry name")
            child_relative = name if relative == "." else f"{relative}/{name}"
            child_metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(child_metadata.st_mode):
                raise RuntimeStagingError(
                    f"runtime tree contains a symbolic link: {child_relative}"
                )
            if stat.S_ISDIR(child_metadata.st_mode):
                child_descriptor = os.open(name, flags, dir_fd=directory_descriptor)
                try:
                    opened = os.fstat(child_descriptor)
                    if (opened.st_dev, opened.st_ino) != (
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                    ):
                        raise RuntimeStagingError(
                            f"runtime directory changed while opening: {child_relative}"
                        )
                    capture_directory(child_descriptor, child_relative)
                    final = os.fstat(child_descriptor)
                    current = os.stat(
                        name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    identity = (
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                        child_metadata.st_size,
                        child_metadata.st_mtime_ns,
                        stat.S_IMODE(child_metadata.st_mode),
                        child_metadata.st_nlink,
                    )
                    if identity != (
                        final.st_dev,
                        final.st_ino,
                        final.st_size,
                        final.st_mtime_ns,
                        stat.S_IMODE(final.st_mode),
                        final.st_nlink,
                    ) or identity != (
                        current.st_dev,
                        current.st_ino,
                        current.st_size,
                        current.st_mtime_ns,
                        stat.S_IMODE(current.st_mode),
                        current.st_nlink,
                    ):
                        raise RuntimeStagingError(
                            f"runtime directory mutated while inspecting: {child_relative}"
                        )
                finally:
                    os.close(child_descriptor)
                continue
            if not stat.S_ISREG(child_metadata.st_mode) or child_metadata.st_nlink != 1:
                raise RuntimeStagingError(
                    f"runtime tree contains an invalid file: {child_relative}"
                )
            file_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            try:
                opened = os.fstat(file_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    child_metadata.st_dev,
                    child_metadata.st_ino,
                ):
                    raise RuntimeStagingError(
                        f"runtime file changed while opening: {child_relative}"
                    )
                digest = hashlib.sha256()
                size = 0
                while block := os.read(file_descriptor, 1024 * 1024):
                    digest.update(block)
                    size += len(block)
                final = os.fstat(file_descriptor)
                current = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                identity = (
                    child_metadata.st_dev,
                    child_metadata.st_ino,
                    child_metadata.st_size,
                    child_metadata.st_mtime_ns,
                    stat.S_IMODE(child_metadata.st_mode),
                    child_metadata.st_nlink,
                )
                if (
                    identity
                    != (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_size,
                        opened.st_mtime_ns,
                        stat.S_IMODE(opened.st_mode),
                        opened.st_nlink,
                    )
                    or identity
                    != (
                        final.st_dev,
                        final.st_ino,
                        final.st_size,
                        final.st_mtime_ns,
                        stat.S_IMODE(final.st_mode),
                        final.st_nlink,
                    )
                    or identity
                    != (
                        current.st_dev,
                        current.st_ino,
                        current.st_size,
                        current.st_mtime_ns,
                        stat.S_IMODE(current.st_mode),
                        current.st_nlink,
                    )
                ):
                    raise RuntimeStagingError(
                        f"runtime file mutated while hashing: {child_relative}"
                    )
                if size != child_metadata.st_size:
                    raise RuntimeStagingError(
                        f"runtime file size changed while hashing: {child_relative}"
                    )
                records.append(
                    _tree_record(
                        child_relative,
                        child_metadata,
                        kind="file",
                        sha256=digest.hexdigest(),
                    )
                )
            finally:
                os.close(file_descriptor)
        os.fsync(directory_descriptor)
        final_directory = os.fstat(directory_descriptor)
        initial_identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            stat.S_IMODE(initial.st_mode),
            initial.st_nlink,
        )
        if initial_identity != (
            final_directory.st_dev,
            final_directory.st_ino,
            final_directory.st_size,
            final_directory.st_mtime_ns,
            stat.S_IMODE(final_directory.st_mode),
            final_directory.st_nlink,
        ):
            raise RuntimeStagingError(
                f"runtime directory mutated while traversing: {relative}"
            )

    try:
        capture_directory(root_descriptor, ".")
    finally:
        os.close(root_descriptor)
    return tuple(sorted(records, key=lambda item: str(item["relative_path"])))


def _freeze_runtime_tree(
    root: Path,
    *,
    bindings: list[tuple[str, str]],
    sizes: dict[str, int],
    manifest_sha256: str,
) -> tuple[dict[str, object], ...]:
    expected_files = set(RUNTIME_FILES) | {RUNTIME_MANIFEST}
    inventory = _capture_runtime_tree(root)
    actual_files = {
        str(item["relative_path"]) for item in inventory if item["kind"] == "file"
    }
    expected_directories = {"."}
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_directories = {
        str(item["relative_path"]) for item in inventory if item["kind"] == "directory"
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeStagingError("staged runtime inventory differs from its contract")
    expected_hashes = dict(bindings) | {RUNTIME_MANIFEST: manifest_sha256}
    for item in inventory:
        if item["mode"] != (0o444 if item["kind"] == "file" else 0o555):
            raise RuntimeStagingError(
                f"staged runtime mode differs: {item['relative_path']}"
            )
        if item["kind"] == "file":
            relative = str(item["relative_path"])
            if item["sha256"] != expected_hashes[relative]:
                raise RuntimeStagingError(f"staged runtime hash differs: {relative}")
            if relative != RUNTIME_MANIFEST and item["size_bytes"] != sizes[relative]:
                raise RuntimeStagingError(f"staged runtime size differs: {relative}")
    return inventory


def _reauthenticate_runtime_tree(
    root: Path,
    expected: tuple[dict[str, object], ...],
) -> None:
    if _capture_runtime_tree(root) != expected:
        raise RuntimeStagingError(
            f"runtime tree identity or bytes changed before publication: {root}"
        )


def _fsync_parent_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(source_bytes, destination_bytes, 0x00000004)
    else:  # pragma: no cover
        raise RuntimeStagingError("platform lacks atomic no-replace directory rename")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"runtime destination exists: {destination}")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _rename_directory_noreplace(
    source: Path,
    destination: Path,
    *,
    expected_tree: tuple[dict[str, object], ...],
) -> None:
    source_descriptor = os.open(
        source,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        source_identity = os.fstat(source_descriptor)
        _reauthenticate_runtime_tree(source, expected_tree)
        _atomic_rename_noreplace(source, destination)
        try:
            published = destination.lstat()
            if stat.S_ISLNK(published.st_mode) or (
                published.st_dev,
                published.st_ino,
            ) != (source_identity.st_dev, source_identity.st_ino):
                raise RuntimeStagingError(
                    "published runtime root is not the frozen staged directory"
                )
            _reauthenticate_runtime_tree(destination, expected_tree)
            _fsync_parent_directory(destination.parent)
        except Exception as exc:
            raise RuntimePublicationUncertainError(
                "runtime rename committed no-replace, but destination verification "
                f"or parent-directory fsync failed; preserve for review: {destination}"
            ) from exc
    finally:
        os.close(source_descriptor)


def _remove_staged_tree(staged: Path) -> None:
    if not staged.exists():
        return
    for directory, _child_directories, _file_names in os.walk(staged):
        try:
            descriptor = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fchmod(descriptor, 0o700)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    shutil.rmtree(staged)


def stage_runtime(source_root: Path, destination: Path) -> dict[str, object]:
    source = _canonical_existing_directory(source_root, label="source root")
    output = _destination(destination)
    staged: Path | None = None
    bindings: list[tuple[str, str]] = []
    sizes: dict[str, int] = {}
    try:
        staged = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.staged-", dir=output.parent)
        )
        os.chmod(staged, 0o700)
        for relative in RUNTIME_FILES:
            source_path = source / relative
            if source_path.resolve(strict=True) != source_path:
                raise RuntimeStagingError(
                    f"runtime source aliases another path: {relative}"
                )
            destination_path = staged / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest, size = _copy_stable_file(source_path, destination_path)
            bindings.append((relative, digest))
            sizes[relative] = size
        manifest_sha256 = _write_manifest(staged / RUNTIME_MANIFEST, bindings)
        for directory, child_directories, file_names in os.walk(staged, topdown=False):
            for file_name in file_names:
                os.chmod(Path(directory) / file_name, 0o444)
            for child in child_directories:
                os.chmod(Path(directory) / child, 0o555)
            os.chmod(directory, 0o555)
        frozen_tree = _freeze_runtime_tree(
            staged,
            bindings=bindings,
            sizes=sizes,
            manifest_sha256=manifest_sha256,
        )
        receipt = {
            "status": "STAGED_IMMUTABLE_NO_REPLACE",
            "runtime_root": str(output),
            "manifest": str(output / RUNTIME_MANIFEST),
            "manifest_sha256": manifest_sha256,
            "file_count": len(RUNTIME_FILES),
            "files": [
                {
                    "relative_path": relative,
                    "sha256": digest,
                    "size_bytes": sizes[relative],
                }
                for relative, digest in sorted(bindings)
            ],
        }
        _rename_directory_noreplace(
            staged,
            output,
            expected_tree=frozen_tree,
        )
        return receipt
    except Exception:
        if staged is not None and staged.exists():
            _remove_staged_tree(staged)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            stage_runtime(arguments.source_root, arguments.destination),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
