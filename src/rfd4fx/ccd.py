"""Chemical Component Dictionary access.

gemmi-backed CCD lookup with a per-entry cache, resolved from (1) an
atomworks-style mirror (``$CCD_MIRROR_PATH``), (2) a cached
``components.cif.gz``, or (3) a one-time RCSB download."""

from __future__ import annotations

import os
import pathlib
import urllib.request

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO / ".ccd_cache"
CCD_DOWNLOAD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

_BOND_ORDER_MAP = {b"SING": 1, b"DOUB": 2, b"TRIP": 3}


class CCDRecord:
    """One CCD entry: atoms and bonds as numpy arrays."""

    __slots__ = ("ccd_id", "atom_names", "elements", "charges", "aromatic_atom",
                 "leaving_atom", "stereo", "bond_i", "bond_j", "bond_order",
                 "bond_aromatic")

    def __init__(self, ccd_id: str, **arrays):
        self.ccd_id = ccd_id
        for key, value in arrays.items():
            setattr(self, key, value)

    def atom_index(self, name: str) -> int | None:
        hits = np.flatnonzero(self.atom_names == name)
        return int(hits[0]) if len(hits) else None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"CCDRecord({self.ccd_id}: {len(self.atom_names)} atoms, "
                f"{len(self.bond_i)} bonds)")


def cache_dir() -> pathlib.Path:
    d = pathlib.Path(os.environ.get("RFD4FX_CCD_CACHE", str(DEFAULT_CACHE_DIR)))
    (d / "entries").mkdir(parents=True, exist_ok=True)
    return d


def _mirror_component_path(ccd_id: str) -> pathlib.Path | None:
    mirror = os.environ.get("CCD_MIRROR_PATH")
    if not mirror:
        return None
    p = pathlib.Path(mirror) / ccd_id[0] / ccd_id / f"{ccd_id}.cif"
    return p if p.exists() else None


def ensure_ccd_source(*, force_cache_update: bool = False) -> pathlib.Path:
    """Locate or download the full CCD file; returns its path."""
    d = cache_dir()
    source = d / "components.cif.gz"
    if source.exists() and not force_cache_update:
        return source
    if force_cache_update and source.exists():
        source.unlink()
        for stale in (d / "entries").glob("*.npz"):
            stale.unlink()  # entries extracted from a superseded source
    print(f"[ccd] downloading full CCD ({CCD_DOWNLOAD_URL}) ...", flush=True)
    tmp = source.with_suffix(".tmp")
    urllib.request.urlretrieve(CCD_DOWNLOAD_URL, tmp)
    tmp.rename(source)
    return source


_DOC = None  # process-level gemmi document (parsed once)


def _load_ccd_document(*, force_cache_update: bool = False):
    global _DOC
    if _DOC is not None and not force_cache_update:
        return _DOC
    import gemmi

    mirror = os.environ.get("CCD_MIRROR_PATH")
    if mirror and (pathlib.Path(mirror) / "components.cif").exists():
        path = pathlib.Path(mirror) / "components.cif"
    elif mirror and (pathlib.Path(mirror) / "components.cif.gz").exists():
        path = pathlib.Path(mirror) / "components.cif.gz"
    else:
        path = ensure_ccd_source(force_cache_update=force_cache_update)
    print(f"[ccd] parsing CCD with gemmi: {path} (once per process) ...", flush=True)
    _DOC = gemmi.cif.read(str(path))
    return _DOC


def get_ccd_entry(ccd_id: str, *, force_cache_update: bool = False) -> CCDRecord:
    """CCD entry by ID: cached per-entry npz, extracted with gemmi on miss/update.

    A per-ID mirror file (`$CCD_MIRROR_PATH/.../<ID>.cif`) takes precedence over the
    full-CCD search.
    """
    ccd_id = ccd_id.upper()
    entry_path = cache_dir() / "entries" / f"{ccd_id}.npz"
    if entry_path.exists() and not force_cache_update:
        z = np.load(entry_path)
        return CCDRecord(ccd_id, **{k: z[k] for k in z.files})

    import gemmi

    mirror_file = _mirror_component_path(ccd_id)
    if mirror_file is not None:
        block = gemmi.cif.read(str(mirror_file)).sole_block()
    else:
        doc = _load_ccd_document(force_cache_update=force_cache_update)
        block = doc.find_block(ccd_id)
        if block is None:
            raise KeyError(f"CCD-ID {ccd_id!r} not found in the CCD")

    names, elements, charges, arom, leaving, stereo = [], [], [], [], [], []
    tab = block.find(
        "_chem_comp_atom.",
        ["atom_id", "type_symbol", "charge", "pdbx_aromatic_flag",
         "pdbx_leaving_atom_flag", "pdbx_stereo_config"],
    )
    for row in tab:
        # row.str(i) returns the UNQUOTED value — manual quote-stripping would
        # also strip trailing primes from atom names like "O5'" (raw `'O5'''`)
        names.append(row.str(0).strip())
        elements.append(row.str(1).strip())
        try:
            charges.append(int(float(row.str(2))))
        except ValueError:
            charges.append(0)
        arom.append(row.str(3).strip().upper() == "Y")
        leaving.append(row.str(4).strip().upper() == "Y")
        stereo.append(row.str(5).strip())

    name_to_idx = {n: i for i, n in enumerate(names)}
    bi, bj, border, barom = [], [], [], []
    bond_tab = block.find(
        "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"]
    )
    for row in bond_tab:
        a1, a2 = row.str(0).strip(), row.str(1).strip()
        if a1 not in name_to_idx or a2 not in name_to_idx:
            continue
        order = _BOND_ORDER_MAP.get(row.str(2).strip().upper().encode(), 0)
        bi.append(name_to_idx[a1])
        bj.append(name_to_idx[a2])
        border.append(order)
        barom.append(row.str(3).strip().upper() == "Y")

    rec = CCDRecord(
        ccd_id,
        atom_names=np.array(names, dtype="S8"),
        elements=np.array(elements, dtype="S4"),
        charges=np.array(charges, dtype=np.int16),
        aromatic_atom=np.array(arom, dtype=bool),
        leaving_atom=np.array(leaving, dtype=bool),
        stereo=np.array(stereo, dtype="S2"),
        bond_i=np.array(bi, dtype=np.int32),
        bond_j=np.array(bj, dtype=np.int32),
        bond_order=np.array(border, dtype=np.int8),
        bond_aromatic=np.array(barom, dtype=bool),
    )
    np.savez_compressed(
        entry_path,
        atom_names=rec.atom_names, elements=rec.elements, charges=rec.charges,
        aromatic_atom=rec.aromatic_atom, leaving_atom=rec.leaving_atom,
        stereo=rec.stereo, bond_i=rec.bond_i, bond_j=rec.bond_j,
        bond_order=rec.bond_order, bond_aromatic=rec.bond_aromatic,
    )
    return rec


def get_ccd_entries(ccd_ids, *, force_cache_update: bool = False) -> dict[str, CCDRecord]:
    """Batch lookup sharing one gemmi parse of the CCD source."""
    return {cid: get_ccd_entry(cid, force_cache_update=force_cache_update)
            for cid in dict.fromkeys(i.upper() for i in ccd_ids)}
