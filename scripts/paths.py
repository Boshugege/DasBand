"""Where the raw data and the other workspace repositories live on this machine.

Every location comes from one per-machine JSON file, by default
``~/.config/wyh-footstep/paths.json`` (set ``WYH_FOOTSTEP_PATHS`` to use another
file)::

    {
      "data_root": "/path/to/wyh-footstep/data",
      "datasets": {"folded-2024-11-26": "/elsewhere/folded"},
      "repos": {"footstep-events": "/path/to/wyh-footstep/footstep-events"}
    }

A dataset resolves to ``datasets[name]`` when listed, otherwise to
``data_root/name``. Code asks for datasets and repositories by name and never
builds absolute paths or assumes where a sibling repository sits. Provenance
records store ``label(path)`` (``data:<dataset>/<file>`` or
``repo:<name>/<file>``) instead of an absolute path, and ``resolve`` turns a
label back into a path on the current machine. The same file is copied into
each repository; the template and notes live in the wyh-footstep-notes
repository (setup/paths.md).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

ENV = 'WYH_FOOTSTEP_PATHS'
DEFAULT = Path('~/.config/wyh-footstep/paths.json')

FOLDED = 'folded-2024-11-26'   # folded-fibre recordings (formerly wangdikai-data)
B202 = 'b202-2023-11-15'       # B202 straight-fibre recordings with Airtag and audio
YUAN2025 = 'yuan2025-srl'      # public TDMS of Yuan et al. (2025)


def config_path() -> Path:
    return Path(os.environ.get(ENV, str(DEFAULT))).expanduser()


@lru_cache(maxsize=None)
def _read(path: str) -> dict:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(
            f'{file} does not exist. Create it from paths.example.json in the '
            f'wyh-footstep-notes repository (setup/paths.md), or point {ENV} at a paths file.')
    return json.loads(file.read_text(encoding='utf-8'))


def config() -> dict:
    return _read(str(config_path()))


def dataset(name: str) -> Path:
    """Root folder of a raw dataset, e.g. ``dataset(FOLDED) / 'csv_time_ranges.csv'``."""
    cfg = config()
    listed = cfg.get('datasets', {}).get(name)
    if listed:
        return Path(listed).expanduser()
    if 'data_root' not in cfg:
        raise KeyError(f'{config_path()} has neither datasets[{name!r}] nor data_root')
    return Path(cfg['data_root']).expanduser() / name


def repo(name: str) -> Path:
    """Root folder of another workspace repository, by its directory name."""
    try:
        return Path(config()['repos'][name]).expanduser()
    except KeyError:
        raise KeyError(f'repository {name!r} is not listed under "repos" in {config_path()}') from None


def _roots() -> list[tuple[str, Path]]:
    """(label prefix, folder) pairs, most specific folder first."""
    cfg = config()
    roots = [(f'data:{name}', Path(p).expanduser()) for name, p in cfg.get('datasets', {}).items()]
    if 'data_root' in cfg:
        roots.append(('data:', Path(cfg['data_root']).expanduser()))
    roots += [(f'repo:{name}', Path(p).expanduser()) for name, p in cfg.get('repos', {}).items()]
    roots = [(prefix, folder.resolve()) for prefix, folder in roots]
    return sorted(roots, key=lambda r: len(r[1].parts), reverse=True)


def label(path: str | Path) -> str:
    """Machine-independent name of a file under a dataset or repository.

    Falls back to the absolute path for files outside every configured root.
    """
    target = Path(path).expanduser().resolve()
    for prefix, folder in _roots():
        if target == folder or folder in target.parents:
            rest = target.relative_to(folder).as_posix()
            if prefix == 'data:':
                return prefix + rest
            return prefix + ('/' + rest if rest != '.' else '')
    return str(target)


def resolve(name: str) -> Path:
    """Inverse of ``label``; absolute paths are returned unchanged."""
    if name.startswith('data:'):
        dataset_name, _, rest = name[5:].partition('/')
        return dataset(dataset_name) / rest if rest else dataset(dataset_name)
    if name.startswith('repo:'):
        repo_name, _, rest = name[5:].partition('/')
        return repo(repo_name) / rest if rest else repo(repo_name)
    return Path(name)
