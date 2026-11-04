"""Ligand interaction-role autofeaturization (C_PISTACK / C_HB / C_SALT).

Ports the native AnnotateInteractions ligand heuristics: aromatic-ring
pi-stacking atoms, hydrogen-bond donor/acceptor roles (acceptor = any N/O/S,
donor = N/O with H), and formal-charge salt roles (1 = positive, 2 =
negative)."""

from __future__ import annotations

import numpy as np

# SaltBridgeEnum / HBondEnum (rfproteina.conditions)
HB_NONE, HB_DONOR, HB_ACCEPTOR, HB_BOTH = 0, 1, 2, 3
SALT_NONE, SALT_POSITIVE, SALT_NEGATIVE = 0, 1, 2

_ACCEPTOR_ELEMENTS = frozenset((7, 8, 16))   # N, O, S
_DONOR_ELEMENTS = frozenset((7, 8))          # N, O (with >= 1 H)


def _rdkit_mol_from_graph(elements: np.ndarray, charges: np.ndarray,
                          bonds: np.ndarray):
    """Build a sanitized RDKit RWMol from the LigandSpec heavy-atom graph.

    `charges` (k,) formal charges are set per atom (without them RDKit would
    re-protonate anionic O / neutralize cationic N and miscount implicit H).
    `bonds` is the LigandSpec convention: (B, 4) rows of [i, j, biotite bond
    type id, aromatic flag] (featurize_atoms._build_bonds). Aromatic bond
    variants (5=AROMATIC_SINGLE, 6=AROMATIC_DOUBLE, 7=AROMATIC_TRIPLE) map to
    RDKit AROMATIC bonds with aromatic atom flags; 1/2/3 map to
    SINGLE/DOUBLE/TRIPLE.
    """
    from rdkit import Chem

    mol = Chem.RWMol()
    for z, q in zip(elements, charges):
        atom = Chem.Atom(int(z))
        atom.SetFormalCharge(int(q))
        mol.AddAtom(atom)
    aromatic = set()
    for i, j, bt, _ar in bonds:
        i, j = int(i), int(j)
        bt = int(bt)
        if bt in (5, 6, 7):
            bond = Chem.BondType.AROMATIC
            aromatic.update((i, j))
        else:
            bond = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE}.get(bt, Chem.BondType.SINGLE)
        mol.AddBond(i, j, bond)
    # RDKit must not invent bonds; keep the graph exactly as given. Implicit-H
    # counts come from standard valence rules after sanitization.
    mol = mol.GetMol()
    for idx in aromatic:
        mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(False)
    Chem.SanitizeMol(mol)
    return mol


def ligand_roles(elements: np.ndarray, charges: np.ndarray,
                 bonds: np.ndarray) -> dict[str, np.ndarray]:
    """Per-atom interaction roles for one ligand (native ligand semantics).

    Args:
      elements: (k,) int32 atomic numbers.
      charges: (k,) int32 formal charges.
      bonds: (B, 4) LigandSpec bond table.

    Returns {"hbond": int8 (k,) in {0,1,2,3}, "pistack": bool (k,),
    "salt": int8 (k,) in {0,1,2}}.
    """
    from rdkit import Chem

    elements = np.asarray(elements, np.int32)
    charges = np.asarray(charges, np.int32)
    k = elements.shape[0]
    mol = _rdkit_mol_from_graph(elements, charges, bonds)

    hbond = np.full(k, HB_NONE, np.int8)
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in _ACCEPTOR_ELEMENTS:
            hbond[atom.GetIdx()] |= HB_ACCEPTOR
        if z in _DONOR_ELEMENTS and atom.GetTotalNumHs() > 0:
            hbond[atom.GetIdx()] |= HB_DONOR
    hbond[hbond == (HB_DONOR | HB_ACCEPTOR)] = HB_BOTH

    # aromatic 5/6-membered rings (RDKit SSSR + aromaticity; all ring atoms)
    pistack = np.zeros(k, bool)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if not (4 < len(ring) <= 6):
            continue
        if not all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in ring):
            continue
        pistack[np.fromiter(ring, np.int64)] = True

    salt = np.full(k, SALT_NONE, np.int8)
    salt[charges > 0] = SALT_POSITIVE
    salt[charges < 0] = SALT_NEGATIVE

    assert hbond.min() >= 0 and hbond.max() <= 3
    assert salt.min() >= 0 and salt.max() <= 2
    assert not Chem.MolToSmiles(mol) is None  # sanity: mol survived sanitization
    return {"hbond": hbond, "pistack": pistack, "salt": salt}
