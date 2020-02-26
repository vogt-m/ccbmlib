#   Copyright 2020 Martin Vogt, Antonio de la Vega de Leon
#   Copyright  (C) 2012-2016 by Greg Landrum, Creative Commons Attribution-ShareAlike 4.0 License
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial
#  portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#  WITH  THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import AllChem

import logging

logger = logging.getLogger(__name__)


def wash(mol, remove_stereo=False):
    """
    Perform a series of modifications to standardize a molecule.
    :param mol: an RDKit molecule
    :param remove_stereo: if True stereochemical information is removed
    :return: Smiles of a washed molecule object
    """
    mol = Chem.RemoveHs(mol)
    mol = saltDisconection(mol)
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)
    Chem.Cleanup(mol)
    mol, _ = NeutraliseCharges(mol)
    Chem.SanitizeMol(mol)
    if not remove_stereo:
        Chem.AssignStereochemistry(mol)
    Chem.SetAromaticity(mol)
    Chem.SetHybridization(mol)
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    smi = Chem.MolToSmiles(mol)
    return smi


def saltDisconection(mol):
    """
    Following instructions on MOE's (chemcomp) help webpage to create a similar
    dissociation between alkaline metals and organic atoms
    :param mol: RDKit molecule
    :return: Molecule with removed salts
    """
    mol = Chem.RWMol(mol)

    metals = [3, 11, 19, 37, 55]  # Alkaline: Li, Na, K, Rb, Cs
    organics = [6, 7, 8, 9, 15, 16, 17, 34, 35, 53]  # Organics: C, N, O, F, P, S, Cl, Se, Br, I
    bondsToDel = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.GetFormalCharge() != 0 or a2.GetFormalCharge() != 0:
            continue

        if a1.GetAtomicNum() in metals:
            if a2.GetAtomicNum() not in organics:
                continue
            if a1.GetDegree() != 1:
                continue
            bondsToDel.append(bond)

        elif a2.GetAtomicNum() in metals:
            if a1.GetAtomicNum() not in organics:
                continue
            if a2.GetDegree() != 1:
                continue
            bondsToDel.append(bond)

    for bond in bondsToDel:
        mol.RemoveBond(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())

    return Chem.Mol(mol)


def _InitialiseNeutralisationReactions():
    """ Taken from http://www.rdkit.org/docs/Cookbook.html
    """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


_reactions = None


def NeutraliseCharges(mol, reactions=None):
    """ Taken from http://www.rdkit.org/docs/Cookbook.html
    """
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions = _InitialiseNeutralisationReactions()
        reactions = _reactions
    #     mol = Chem.MolFromSmiles(smiles)
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (mol, True)
    else:
        return (mol, False)


def wash_molecules(mol_suppl):
    """
    Washes all molecules in mol_suppl.
    Returns a dictionary with unique smiles as key.
    If molecules have the '_Name' property, the values contain a list of these names.
    :param mol_suppl: RDKit mol supplier
    :return: dictionary of washed Smiles
    """
    good_count = 0
    count = 0
    washed = defaultdict(list)
    for mol in mol_suppl:
        count += 1
        if not mol:
            continue
        try:
            smi = wash(mol, True)
            if "." not in smi:
                if mol.HasProp("_Name"):
                    name = mol.GetProp("_Name")
                    washed[smi].append(name)
                elif smi not in washed:
                    washed[smi] = []
                good_count += 1
        except ValueError as ve:
            print(ve)
        if good_count % 10000 == 0:
            logger.info("Progress: {} mols washed".format(good_count))
    bad_count = count - good_count
    logging.info("Washing summary: good={} bad={}".format(good_count, bad_count))
    return washed


def export_washed(washed, smiles_file, duplicates_file):
    """
    Export a dictionary of molecules to a Smiles file
    :param washed: dictionary of smiles. values are lists of names
    :param smiles_file: output Smiles file
    :param duplicates_file:
    :return:
    """
    with open(smiles_file, "w") as smif:
        smif.write("Smiles Name\n")
        with open(duplicates_file, "w") as dupf:
            for smi, ids in washed.items():
                if ids:
                    smif.write("{} {}\n".format(smi, ids[0]))
                else:
                    smif.write("{}\n".format(smi))
                if len(ids) > 1:
                    dupf.write(" ".join(ids))
                    dupf.write("\n")


if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if len(sys.argv) != 4:
        print("Usage: python {} smiles-input-file washed-smiles-output-file duplicates-text-output-file".format(
            sys.argv[0]))
        sys.exit(1)
    input_file = sys.argv[1]
    washed_file = sys.argv[2]
    duplicates_file = sys.argv[3]
    if os.path.exists(washed_file) or os.path.exists(duplicates_file):
        print("Some output files already exist", file=sys.stderr)
        sys.exit(1)
    mol_suppl = Chem.SmilesMolSupplier(input_file)
    washed = wash_molecules(mol_suppl)
    export_washed(washed, washed_file, duplicates_file)
