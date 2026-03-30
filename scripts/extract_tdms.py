# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from nptdms import TdmsFile


TDMS_TIME_PATTERN = re.compile(r"_UTC_(\d{8}_\d{6}\.\d+)\.tdms$", re.IGNORECASE)
DATETIME_FORMATS = (
    "%Y-%m-%d  %H:%M:%S",
    "%Y/%m/%d  %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d  %H:%M:%S.%f",
    "%Y/%m/%d  %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S.%f",
)


@dataclass(frozen=True)
class TdmsIndex:
    path: Path
    start_utc: datetime
    nominal_end_utc: datetime


@dataclass(frozen=True)
class ExtractionTask:
    name: str
    csv_path: Path
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime


def parse_datetime(text: str) -> datetime:
    s = text.strip().replace("T", " ")
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime format: {text!r}")


def read_csv_start_end(csv_path: Path, encoding: str = "utf-8-sig") -> Tuple[datetime, datetime]:
    first_dt: Optional[datetime] = None
    last_dt: Optional[datetime] = None
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {csv_path}")
        if not str(header[0]).strip().lower().startswith("datetime"):
            raise ValueError(f"First column is not datetime in {csv_path}: {header[0]!r}")
        for row in reader:
            if not row:
                continue
            raw = row[0].strip()
            if not raw:
                continue
            dt = parse_datetime(raw)
            if first_dt is None:
                first_dt = dt
            last_dt = dt
    if first_dt is None or last_dt is None:
        raise ValueError(f"No datetime rows found in {csv_path}")
    return first_dt, last_dt


def build_task(name: str, airtag_csv: Path, offset_hours: float, encoding: str = "utf-8-sig") -> ExtractionTask:
    start_local, end_local = read_csv_start_end(airtag_csv, encoding=encoding)
    offset = timedelta(hours=offset_hours)
    start_utc = start_local - offset
    end_utc = end_local - offset
    if end_utc < start_utc:
        raise ValueError(f"End earlier than start in {airtag_csv}")
    return ExtractionTask(
        name=name,
        csv_path=airtag_csv,
        start_local=start_local,
        end_local=end_local,
        start_utc=start_utc,
        end_utc=end_utc,
    )


def parse_tdms_start(path: Path) -> datetime:
    m = TDMS_TIME_PATTERN.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse UTC timestamp from TDMS name: {path.name}")
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S.%f")


def build_tdms_index(tdms_dir: Path, default_duration_sec: float = 60.0) -> List[TdmsIndex]:
    tdms_files = sorted(tdms_dir.glob("*.tdms"), key=lambda p: p.name)
    if not tdms_files:
        raise ValueError(f"No TDMS files found in {tdms_dir}")
    starts = [(p, parse_tdms_start(p)) for p in tdms_files]
    starts.sort(key=lambda x: x[1])

    index = []
    for i, (path, start) in enumerate(starts):
        nominal_end = starts[i + 1][1] if i + 1 < len(starts) else start + timedelta(seconds=default_duration_sec)
        index.append(TdmsIndex(path=path, start_utc=start, nominal_end_utc=nominal_end))
    return index


def choose_overlapping_files(index: Sequence[TdmsIndex], start_utc: datetime, end_utc: datetime) -> List[TdmsIndex]:
    return [item for item in index if item.start_utc <= end_utc and item.nominal_end_utc >= start_utc]


def list_channels(tdms_file: TdmsFile):
    channels = []
    for group in tdms_file.groups():
        for channel in group.channels():
            channels.append(channel)
    return channels


def extraction_indices(
    file_start_utc: datetime,
    fs: float,
    n_samples: int,
    seg_start_utc: datetime,
    seg_end_utc: datetime,
) -> Tuple[int, int]:
    dt0 = (seg_start_utc - file_start_utc).total_seconds()
    dt1 = (seg_end_utc - file_start_utc).total_seconds()
    start_idx = int(math.ceil(dt0 * fs - 1e-9))
    end_exclusive = int(math.floor(dt1 * fs + 1e-9)) + 1
    start_idx = max(0, start_idx)
    end_exclusive = min(n_samples, end_exclusive)
    end_exclusive = max(end_exclusive, start_idx)
    return start_idx, end_exclusive


def write_array_csv(path: Path, data: np.ndarray, write_header: bool, n_channels: int, append: bool, channel_offset: int = 0):
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        if write_header:
            header = ",".join([f"ch_{i + channel_offset}" for i in range(n_channels)])
            f.write(header + "\n")
        if data.size > 0:
            if np.issubdtype(data.dtype, np.integer):
                np.savetxt(f, data, delimiter=",", fmt="%d")
            else:
                np.savetxt(f, data, delimiter=",", fmt="%.10g")


def extract_one_task(
    task: ExtractionTask,
    tdms_index: Sequence[TdmsIndex],
    output_dir: Path,
    fs: float,
    overwrite: bool,
    dry_run: bool,
    skip_channels: int = 0,
) -> Tuple[Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{task.name}.csv"
    overlaps = choose_overlapping_files(tdms_index, task.start_utc, task.end_utc)
    print(
        f"[Extract] {task.name}: {task.start_local} -> {task.end_local} "
        f"(UTC {task.start_utc} -> {task.end_utc}), matched TDMS files: {len(overlaps)}"
    )
    if not overlaps:
        raise FileNotFoundError(f"No overlapping TDMS files found for {task.name}.")
    if out_csv.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output exists, use overwrite=True: {out_csv}")

    if not dry_run and out_csv.exists() and overwrite:
        out_csv.unlink()

    total_rows = 0
    output_channels = 0
    has_written = False
    n_channels_ref: Optional[int] = None

    for item in overlaps:
        tdms = TdmsFile.read(item.path)
        channels = list_channels(tdms)
        if not channels:
            continue
        n_channels = len(channels)
        if n_channels_ref is None:
            n_channels_ref = n_channels
        elif n_channels != n_channels_ref:
            raise ValueError(
                f"Channel count mismatch for {task.name}: expected {n_channels_ref}, got {n_channels} in {item.path.name}"
            )

        n_samples = len(channels[0])
        start_idx, end_exclusive = extraction_indices(
            file_start_utc=item.start_utc,
            fs=fs,
            n_samples=n_samples,
            seg_start_utc=task.start_utc,
            seg_end_utc=task.end_utc,
        )
        row_count = end_exclusive - start_idx
        if row_count <= 0:
            continue
        total_rows += row_count
        channels_to_use = channels[skip_channels:] if skip_channels > 0 else channels
        output_channels = len(channels_to_use)
        print(f"  [Use] {item.path.name}: samples [{start_idx}, {end_exclusive}) => {row_count} rows")
        if dry_run:
            continue
        data = np.column_stack([np.asarray(ch[start_idx:end_exclusive]) for ch in channels_to_use])
        write_array_csv(
            path=out_csv,
            data=data,
            write_header=not has_written,
            n_channels=output_channels,
            append=has_written,
            channel_offset=skip_channels,
        )
        has_written = True

    if not dry_run and not has_written:
        raise RuntimeError(f"No overlapping samples were written for {task.name}.")
    return out_csv, total_rows, output_channels


def extract_name_to_csv(
    *,
    name: str,
    airtag_csv_dir: str,
    tdms_dir: str,
    output_dir: str,
    fs: float = 2000.0,
    csv_utc_offset_hours: float = 8.0,
    skip_channels: int = 18,
    overwrite: bool = False,
    dry_run: bool = False,
    encoding: str = "utf-8-sig",
):
    name = name.lower().strip()
    csv_dir = Path(airtag_csv_dir).expanduser().resolve()
    tdms_dir_path = Path(tdms_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    airtag_csv = csv_dir / f"{name}.csv"
    if not airtag_csv.exists():
        raise FileNotFoundError(f"Airtag CSV not found: {airtag_csv}")
    if not tdms_dir_path.exists():
        raise FileNotFoundError(f"TDMS dir not found: {tdms_dir_path}")

    task = build_task(name=name, airtag_csv=airtag_csv, offset_hours=csv_utc_offset_hours, encoding=encoding)
    index = build_tdms_index(tdms_dir_path)
    csv_path, rows, channels = extract_one_task(
        task=task,
        tdms_index=index,
        output_dir=out_dir,
        fs=fs,
        overwrite=overwrite,
        dry_run=dry_run,
        skip_channels=skip_channels,
    )
    return {
        "csv_path": csv_path,
        "rows": rows,
        "channels": channels,
        "task": task,
    }
