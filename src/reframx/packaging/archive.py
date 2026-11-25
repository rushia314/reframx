"""Low-level atomic ZIP archive reader/writer for ``.rfx`` files."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from reframx.core.exceptions import ResearchFrameworkError
from reframx.packaging.manifest import RFX_EXTENSION


class ArchiveError(ResearchFrameworkError):
    """Raised when an ``.rfx`` archive is corrupted, invalid, or unsafe."""


def normalize_rfx_path(path: str | Path) -> Path:
    """Ensure the destination path ends with ``.rfx``."""
    target = Path(path)
    if target.suffix != RFX_EXTENSION:
        target = target.with_suffix(target.suffix + RFX_EXTENSION)
    return target


def _validate_member_name(name: str) -> str:
    """Reject Zip Slip (absolute paths or ``..`` traversal) in archive entries."""
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ArchiveError(f"Unsafe archive member path: {name!r}")
    return str(posix)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RFXArchiveWriter:
    """Stages archive entries and writes an ``.rfx`` ZIP file atomically."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}
        self._checksums: dict[str, str] = {}

    @property
    def checksums(self) -> dict[str, str]:
        return dict(self._checksums)

    def add_bytes(self, member_name: str, data: bytes) -> str:
        clean = _validate_member_name(member_name)
        self._entries[clean] = data
        digest = sha256_bytes(data)
        self._checksums[clean] = digest
        return digest

    def add_json(self, member_name: str, payload: Any) -> str:
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return self.add_bytes(member_name, encoded)

    def add_file(self, member_name: str, source_path: str | Path) -> str:
        src = Path(source_path)
        if not src.is_file():
            raise ArchiveError(f"Cannot pack missing file: {src}")
        return self.add_bytes(member_name, src.read_bytes())

    def add_directory(self, member_prefix: str, source_dir: str | Path) -> None:
        root = Path(source_dir)
        if not root.is_dir():
            raise ArchiveError(f"Cannot pack missing directory: {root}")
        for file_path in sorted(root.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(root).as_posix()
                self.add_file(f"{member_prefix}/{rel}", file_path)

    def write(
        self,
        destination: str | Path,
        *,
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> Path:
        """Atomically write, verify, and move the ``.rfx`` archive into place."""
        target = normalize_rfx_path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")

        try:
            with zipfile.ZipFile(tmp_path, mode="w", compression=compression) as zf:
                for member_name in sorted(self._entries.keys()):
                    zf.writestr(member_name, self._entries[member_name])

            # Verify archive integrity before committing into place.
            with zipfile.ZipFile(tmp_path, mode="r") as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise ArchiveError(
                        f"Archive integrity check failed on member '{bad_member}'."
                    )

            os.replace(tmp_path, target)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        return target


class RFXArchiveReader:
    """Safe reader for ``.rfx`` ZIP archives with checksum and Zip Slip validation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ArchiveError(f"Archive file not found: {self.path}")
        if not zipfile.is_zipfile(self.path):
            raise ArchiveError(f"File is not a valid .rfx ZIP archive: {self.path}")

    def list_members(self) -> tuple[str, ...]:
        with zipfile.ZipFile(self.path, mode="r") as zf:
            return tuple(_validate_member_name(name) for name in zf.namelist())

    def has_member(self, member_name: str) -> bool:
        clean = _validate_member_name(member_name)
        with zipfile.ZipFile(self.path, mode="r") as zf:
            return clean in zf.namelist()

    def read_bytes(self, member_name: str) -> bytes:
        clean = _validate_member_name(member_name)
        with zipfile.ZipFile(self.path, mode="r") as zf:
            try:
                return zf.read(clean)
            except KeyError as exc:
                raise ArchiveError(
                    f"Archive '{self.path}' is missing member '{clean}'."
                ) from exc

    def read_json(self, member_name: str) -> Any:
        raw = self.read_bytes(member_name)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ArchiveError(
                f"Failed to parse JSON member '{member_name}' in '{self.path}': {exc}"
            ) from exc

    def verify_checksums(self, expected: Mapping[str, str]) -> list[str]:
        """Verify SHA-256 digests and return a list of any mismatch messages."""
        errors: list[str] = []
        with zipfile.ZipFile(self.path, mode="r") as zf:
            names = set(zf.namelist())
            for member_name, expected_digest in expected.items():
                clean = _validate_member_name(member_name)
                if clean not in names:
                    errors.append(f"Missing expected archive member: '{clean}'")
                    continue
                actual = sha256_bytes(zf.read(clean))
                if actual != expected_digest:
                    errors.append(
                        f"Checksum mismatch for '{clean}': "
                        f"expected {expected_digest[:12]}, got {actual[:12]}"
                    )
        return errors

    def extract_prefix(self, prefix: str, destination_dir: str | Path) -> list[Path]:
        """Safely extract all members under ``prefix/`` into ``destination_dir``."""
        dest_root = Path(destination_dir).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        normalized_prefix = prefix.strip("/") + "/"
        extracted: list[Path] = []

        with zipfile.ZipFile(self.path, mode="r") as zf:
            for raw_name in zf.namelist():
                clean = _validate_member_name(raw_name)
                if not clean.startswith(normalized_prefix):
                    continue
                rel = clean[len(normalized_prefix) :]
                if not rel or rel.endswith("/"):
                    continue
                out_path = (dest_root / rel).resolve()
                if dest_root not in out_path.parents:
                    raise ArchiveError(f"Unsafe extraction target for '{clean}'.")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(zf.read(clean))
                extracted.append(out_path)

        return extracted
