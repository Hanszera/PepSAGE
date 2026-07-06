"""pepsage dataset utilities."""

from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import joblib
import lmdb
import numpy as np
import torch
from Bio.PDB import PDBExceptions
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from core.models.torsion import get_torsion_angle
from core.modules.protein.constants import BBHeavyAtom
from core.modules.protein.parsers import parse_pdb
from core.utils.data import PaddingCollate

LOGGER = logging.getLogger(__name__)
root_path = Path(__file__).resolve().parents[2]


def _resolve_local_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = root_path / path
    return path.resolve()


class ReferenceArray:
    def __init__(self, path, dtype=np.float32, shape=(360, 1001, 3, 3, 1000)):
        self.data = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def lookup(self, x, t, sample_idx):
        data = torch.from_numpy(self.data[x, t, :, :, sample_idx])
        return data


class RotReferenceArray:
    def __init__(self, path, dtype=np.float32, shape=(4000, 10000, 3, 3)):
        self.data = np.memmap(path, dtype=dtype, mode="r", shape=shape)

    def lookup(self, x, sample_idx):
        data = torch.from_numpy(self.data[x, sample_idx])
        return data


def _load_excluded_ids(dataset_dir: Path) -> set[str]:
    names_path = dataset_dir / "names.txt"
    if not names_path.exists():
        return set()
    return {line.strip() for line in names_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def preprocess_structure(task: dict[str, Any]) -> dict[str, Any] | None:
    try:
        sample_id = str(task["id"])
        excluded_ids = set(task.get("excluded_ids", set()))
        if sample_id in excluded_ids:
            raise ValueError(f"{sample_id} is excluded by names.txt")

        pdb_path = Path(task["pdb_path"])
        pep, _ = parse_pdb(str(pdb_path / "peptide.pdb"))
        rec, _ = parse_pdb(str(pdb_path / "pocket.pdb"))
        if pep is None or rec is None:
            raise ValueError("Failed to parse peptide or pocket structure")

        center_mask = pep["mask_heavyatom"][:, BBHeavyAtom.CA]
        center = pep["pos_heavyatom"][center_mask, BBHeavyAtom.CA].mean(dim=0)

        pep["pos_heavyatom"] = pep["pos_heavyatom"] - center.view(1, 1, 3)
        pep["torsion_angle"], pep["torsion_angle_mask"] = get_torsion_angle(pep["pos_heavyatom"], pep["aa"])
        if pep["aa"].numel() < 3 or pep["aa"].numel() > 25:
            raise ValueError("peptide length not in [3, 25]")

        rec["pos_heavyatom"] = rec["pos_heavyatom"] - center.view(1, 1, 3)
        rec["torsion_angle"], rec["torsion_angle_mask"] = get_torsion_angle(rec["pos_heavyatom"], rec["aa"])
        rec["chain_nb"] = rec["chain_nb"] + 1

        data = {
            "id": sample_id,
            "generate_mask": torch.cat([torch.zeros_like(rec["aa"]), torch.ones_like(pep["aa"])], dim=0).bool(),
        }
        for key, rec_value in rec.items():
            pep_value = pep[key]
            if torch.is_tensor(rec_value):
                data[key] = torch.cat([rec_value, pep_value], dim=0)
            elif isinstance(rec_value, list):
                data[key] = rec_value + pep_value
            else:
                raise TypeError(f"Unsupported field type for {key}: {type(rec_value)!r}")
        return data
    except (PDBExceptions.PDBConstructionException, KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("[%s] %s: %s", task.get("id", "<unknown>"), exc.__class__.__name__, exc)
        return None


class PepDataset(Dataset):
    def __init__(
        self,
        structure_dir: str | Path,
        dataset_dir: str | Path,
        name: str = "pep",
        transform=None,
        reset: bool = False,
    ):
        super().__init__()
        self.structure_dir = _resolve_local_path(structure_dir)
        self.dataset_dir = _resolve_local_path(dataset_dir)
        self.transform = transform
        self.name = name
        self.excluded_ids = _load_excluded_ids(self.dataset_dir)

        self.db_ids: list[str] | None = None
        self.sample_paths: dict[str, Path] = {}
        self._load_structures(reset)

    @property
    def _cache_dir(self) -> Path:
        return self.dataset_dir / f"{self.name}_structure_cache"

    @property
    def _cache_index_path(self) -> Path:
        return self._cache_dir / "index.pkl"

    @property
    def _cache_samples_dir(self) -> Path:
        return self._cache_dir / "samples"

    @property
    def _legacy_cache_db_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_structure_cache.lmdb"

    @property
    def _legacy_cache_lock_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_structure_cache.lmdb-lock"

    def _sample_cache_path(self, sample_id: str) -> Path:
        return self._cache_samples_dir / f"{sample_id}.pkl"

    def _remove_local_cache_files(self) -> None:
        self._close_db()
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)

    def _connect_db(self) -> None:
        if self.db_ids is not None and self.sample_paths:
            return

        if not self._cache_index_path.exists():
            raise FileNotFoundError(f"Missing cache index: {self._cache_index_path}")

        with self._cache_index_path.open("rb") as handle:
            cache_index = pickle.load(handle)
        self.db_ids = list(cache_index["ids"])
        self.sample_paths = {
            sample_id: self._cache_dir / relative_path
            for sample_id, relative_path in cache_index["files"].items()
        }

    def _close_db(self) -> None:
        self.db_ids = None
        self.sample_paths = {}

    def _open_legacy_lmdb(self):
        if not self._legacy_cache_db_path.exists():
            raise FileNotFoundError(f"Missing LMDB cache: {self._legacy_cache_db_path}")
        return lmdb.open(
            str(self._legacy_cache_db_path),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def _import_from_legacy_lmdb(self) -> None:
        LOGGER.info("Importing official pepsage LMDB cache from %s", self._legacy_cache_db_path)
        self._remove_local_cache_files()
        self._cache_samples_dir.mkdir(parents=True, exist_ok=True)

        cache_ids: list[str] = []
        cache_files: dict[str, str] = {}
        env = self._open_legacy_lmdb()
        try:
            with env.begin(buffers=True) as txn:
                cursor = txn.cursor()
                for key_bytes in tqdm(cursor.iternext(values=False), dynamic_ncols=True, desc="Import LMDB"):
                    key = bytes(key_bytes)
                    sample_id = key.decode("utf-8")
                    value = txn.get(key)
                    if value is None:
                        continue
                    data = pickle.loads(bytes(value))
                    sample_path = self._sample_cache_path(sample_id)
                    with sample_path.open("wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    cache_ids.append(sample_id)
                    cache_files[sample_id] = str(sample_path.relative_to(self._cache_dir))
        finally:
            env.close()

        with self._cache_index_path.open("wb") as handle:
            pickle.dump({"ids": cache_ids, "files": cache_files}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_structures(self, reset: bool) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        if reset:
            self._remove_local_cache_files()

        if self._cache_index_path.exists():
            LOGGER.info("Loading pepsage split %s from local pkl cache %s", self.name, self._cache_dir)
            self._connect_db()
            self._close_db()
            return

        if self._legacy_cache_db_path.exists():
            self._import_from_legacy_lmdb()
            return

        if not self.structure_dir.exists():
            raise FileNotFoundError(
                f"Neither local pkl cache nor official LMDB cache exists for split {self.name}, "
                f"and structure directory was not found: {self.structure_dir}"
            )

        all_pdbs = sorted(entry.name for entry in self.structure_dir.iterdir() if entry.is_dir())
        if not all_pdbs:
            raise FileNotFoundError(f"No structure directories found in {self.structure_dir}")

        LOGGER.warning(
            "Official LMDB cache for split %s was not found. Falling back to rebuilding from raw structures in %s.",
            self.name,
            self.structure_dir,
        )
        self._preprocess_structures(all_pdbs)

    def _preprocess_structures(self, pdb_list: list[str]) -> None:
        tasks = [
            {
                "id": pdb_name,
                "pdb_path": self.structure_dir / pdb_name,
                "excluded_ids": self.excluded_ids,
            }
            for pdb_name in pdb_list
        ]

        data_list = joblib.Parallel(n_jobs=max(joblib.cpu_count() // 2, 1))(
            joblib.delayed(preprocess_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc="Preprocess")
        )

        self._cache_samples_dir.mkdir(parents=True, exist_ok=True)
        cache_ids: list[str] = []
        cache_files: dict[str, str] = {}
        if self._cache_index_path.exists():
            with self._cache_index_path.open("rb") as handle:
                cache_index = pickle.load(handle)
            cache_ids = list(cache_index["ids"])
            cache_files = dict(cache_index["files"])

        seen_ids = set(cache_ids)
        for data in tqdm(data_list, dynamic_ncols=True, desc="Write cache"):
            if data is None:
                continue
            sample_id = data["id"]
            sample_path = self._sample_cache_path(sample_id)
            with sample_path.open("wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if sample_id not in seen_ids:
                cache_ids.append(sample_id)
                seen_ids.add(sample_id)
            cache_files[sample_id] = str(sample_path.relative_to(self._cache_dir))

        with self._cache_index_path.open("wb") as handle:
            pickle.dump({"ids": cache_ids, "files": cache_files}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self) -> int:
        self._connect_db()
        return len(self.db_ids or [])

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._connect_db()
        sample_id = self.db_ids[index]
        sample_path = self.sample_paths[sample_id]
        with sample_path.open("rb") as handle:
            data = pickle.load(handle)
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    dataset = PepDataset(
        structure_dir="../data/pepsage/crude",
        dataset_dir="../data/pepsage/processed",
        name="pep_pocket_test",
        transform=None,
        reset=False,
    )
    print(len(dataset))
    print(dataset[0]["id"])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=PaddingCollate(eight=False))
    batch = next(iter(dataloader))
    print(batch["torsion_angle"].shape)
    print(batch["torsion_angle_mask"].shape)
