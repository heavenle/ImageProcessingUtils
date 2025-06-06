import re
from numpy.random import shuffle
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType

"""
Taken and modified from Coley and co-workers:
https://github.com/connorcoley/rdchiral/blob/master/templates/template_extractor.py
"""

VERBOSE = False
USE_STEREOCHEMISTRY = True
MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS = 5
INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS = True


def mols_from_smiles_list(all_smiles):
    '''Given a list of smiles strings, this function creates rdkit
    molecules'''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols


def replace_deuterated(smi):
    return re.sub('\[2H\]', r'[H]', smi)


def clear_mapnum(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return mol


def get_tagged_atoms_from_mols(mols):
    '''Takes a list of RDKit molecules and returns total list of
    atoms and their tags'''
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms
        atom_tags += new_atom_tags
    #     print("*************")
    #     print(atoms, atom_tags)
    return atoms, atom_tags


def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags


# +
# smi="Br[CH:16]([c:17]1[cH:18][cH:19][cH:20][cH:21][cH:22]1)[C:23]([O:24][CH2:25][CH3:26])=[O:27].O=C([O-])[O-].[K+].[K+].[O:1]([CH2:2][CH2:3][O:4][c:5]1[n:6][c:7]2[c:8]([nH:9][cH:10][n:11]2)[c:12]([NH2:13])[n:14]1)[CH3:15].CN(C)C=O"
# mol_smi_1=Chem.MolFromSmiles(smi)
# print(get_tagged_atoms_from_mol(mol_smi_1)[1])
# mol_smi_1

# +
# from rdkit import Chem
# smi="[O:1]([CH2:2][CH2:3][O:4][c:5]1[n:6][c:7]2[c:8]([n:9][cH:10][n:11]2[CH:16]([c:17]2[cH:18][cH:19][cH:20][cH:21][cH:22]2)[C:23]([O:24][CH2:25][CH3:26])=[O:27])[c:12]([NH2:13])[n:14]1)[CH3:15]"
# mol_smi2=Chem.MolFromSmiles(smi)
# print(get_tagged_atoms_from_mol(mol_smi2)[1])
# mol_smi2

# +
# # 两个集合求差集
# set(get_tagged_atoms_from_mol(mol_smi_1)[1]) - set(get_tagged_atoms_from_mol(mol_smi2)[1])
# -

def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''
    # 基于基本属性比较两个RDKit原子

    if atom1.GetSmarts() != atom2.GetSmarts(): return True  # easiest check
    #     获取原子序号：GetAtomicNum()
    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True  # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
    #     获取原子形式电荷：GetFormalCharge()
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    #     获取原子连接数（受H是否隐藏影响）：GetDegree()
    if atom1.GetDegree() != atom2.GetDegree(): return True
    # if atom1.IsInRing() != atom2.IsInRing(): return True # do not want to check this!
    # e.g., in macrocycle formation, don't want the template to include the entire ring structure
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    #     是否为芳香键：GetIsAromatic()
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True
    # 对键进行遍历：m.GetBonds()
    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()])
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()])
    if bonds1 != bonds2: return True

    return False


def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')
            and a.GetProp('molAtomMapNumber') == str(mapnum)][0]


def get_tetrahedral_atoms(reactants, products):
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            atom_tag = ar.GetProp('molAtomMapNumber')
            ir = ar.GetIdx()
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    if ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED or \
                            ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms


def set_isotope_to_equal_mapnum(mol):
    for a in mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))


def get_frag_around_tetrahedral_center(mol, idx):
    '''Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes
    使用四面体原子的邻居构建MolFragment，其中分子已经更新为包含同位素
    '''

    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0 \
                   else '[#{}]'.format(a.GetAtomicNum()) for a in mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(mol, ids_to_include, isomericSmiles=True,
                                    atomSymbols=symbols, allBondsExplicit=True,
                                    allHsExplicit=True)


def check_tetrahedral_centers_equivalent(atom1, atom2):
    '''Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped
    检查四面体中心在手性上是否等效，忽略手性标签。拥有输入原子的分子必须具有同位素映射
    '''
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx())
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False


def clear_isotope(mol):
    [a.SetIsotope(0) for a in mol.GetAtoms()]


def get_changed_atoms(reactants, products):
    '''Looks at mapped atoms in a reaction and determines which ones changed'''
    # 观察反应中映射的原子，并确定哪些原子发生了变化

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)
    # verbose=True：表示设置运行的时候显示详细信息
    if VERBOSE: print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if VERBOSE: print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if VERBOSE: print('warning: different atom tags appear in reactants and products')
        # err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if VERBOSE: print('warning: total number of tagged atoms differ, stoichometry != 1?')
        # err = 1

    # Find differences 
    changed_atoms = []  # actual reactant atom species
    changed_atom_tags = []  # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):

        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atom_tags:  # don't bother comparing if we know this atom changes
                # If atom changed, add
                #                 print("&&&&&&&&&&&&&&&&&&&&&&&&&&")
                #                 print(prod_atoms[i])
                #                 print(reac_atoms[j])
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    # Atoms that change CHIRALITY (just tetrahedral for now...)
    tetra_atoms = get_tetrahedral_atoms(reactants, products)
    if VERBOSE:
        print('Found {} atom-mapped tetrahedral atoms that have chirality specified at least partially'.format(
            len(tetra_atoms)))
    [set_isotope_to_equal_mapnum(reactant) for reactant in reactants]
    [set_isotope_to_equal_mapnum(product) for product in products]
    for (atom_tag, ar, ap) in tetra_atoms:
        if VERBOSE:
            print('For atom tag {}'.format(atom_tag))
            print('    reactant: {}'.format(ar.GetChiralTag()))
            print('    product:  {}'.format(ap.GetChiralTag()))
        if atom_tag in changed_atom_tags:
            if VERBOSE:
                print('-> atoms have changed (by more than just chirality!)')
        else:
            unchanged = check_tetrahedral_centers_equivalent(ar, ap) and \
                        ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
            if unchanged:
                if VERBOSE:
                    print('-> atoms confirmed to have same chirality, no change')
            else:
                if VERBOSE:
                    print('-> atom changed chirality!!')
                #                     确保手性变化在反应中心旁边，而不是随机指定（必须连接到改变的原子）
                # Make sure chiral change is next to the reaction center and not
                # a random specifidation (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    if neighbor.HasProp('molAtomMapNumber'):
                        if neighbor.GetProp('molAtomMapNumber') in changed_atom_tags:
                            tetra_adj_to_rxn = True
                            break
                if tetra_adj_to_rxn:
                    if VERBOSE:
                        print('-> atom adj to reaction center, now included')
                    changed_atom_tags.append(atom_tag)
                    changed_atoms.append(ar)
                else:
                    if VERBOSE:
                        print('-> adj far from reaction center, not including')
    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]

    if VERBOSE:
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))

    return changed_atoms, changed_atom_tags, err


def get_special_groups(mol):
    '''Given an RDKit molecule, this function returns a list of tuples, where
    each tuple contains the AtomIdx's for a special group of atoms which should 
    be included in a fragment all together. This should only be done for the 
    reactants, otherwise the products might end up with mapping mismatches
    We draw a distinction between atoms in groups that trigger that whole
    group to be included, and "unimportant" atoms in the groups that will not
    be included if another atom matches.'''

    # Define templates
    group_templates = [
        (range(3), '[OH0,SH0]=C[O,Cl,I,Br,F]',),  # carboxylic acid / halogen
        (range(3), '[OH0,SH0]=CN',),  # amide/sulfamide
        (range(4), 'S(O)(O)[Cl]',),  # sulfonyl chloride
        (range(3), 'B(O)O',),  # boronic acid/ester
        ((0,), '[Si](C)(C)C'),  # trialkyl silane
        ((0,), '[Si](OC)(OC)(OC)'),  # trialkoxy silane, default to methyl
        (range(3), '[N;H0;$(N-[#6]);D2]-,=[N;D2]-,=[N;D1]',),  # azide
        (range(8), 'O=C1N([Br,I,F,Cl])C(=O)CC1',),  # NBS brominating agent
        (range(11), 'Cc1ccc(S(=O)(=O)O)cc1'),  # Tosyl
        ((7,), 'CC(C)(C)OC(=O)[N]'),  # N(boc)
        ((4,), '[CH3][CH0]([CH3])([CH3])O'),  # t-Butyl
        (range(2), '[C,N]=[C,N]',),  # alkene/imine
        (range(2), '[C,N]#[C,N]',),  # alkyne/nitrile
        ((2,), 'C=C-[*]',),  # adj to alkene
        ((2,), 'C#C-[*]',),  # adj to alkyne
        ((2,), 'O=C-[*]',),  # adj to carbonyl
        ((3,), 'O=C([CH3])-[*]'),  # adj to methyl ketone
        ((3,), 'O=C([O,N])-[*]',),  # adj to carboxylic acid/amide/ester
        (range(4), 'ClS(Cl)=O',),  # thionyl chloride
        (range(2), '[Mg,Li,Zn,Sn][Br,Cl,I,F]',),  # grinard/metal (non-disassociated)
        (range(3), 'S(O)(O)',),  # SO2 group
        (range(2), 'N~N',),  # diazo
        ((1,), '[!#6;R]@[#6;R]',),  # adjacency to heteroatom in ring
        ((2,), '[a!c]:a:a',),  # two-steps away from heteroatom in aromatic ring
        # ((1,), 'c(-,=[*]):c([Cl,I,Br,F])',), # ortho to halogen on ring - too specific?
        # ((1,), 'c(-,=[*]):c:c([Cl,I,Br,F])',), # meta to halogen on ring - too specific?
        ((0,), '[B,C](F)(F)F'),  # CF3, BF3 should have the F3 included
    ]

    # Stereo-specific ones (where we will need to include neighbors)
    # Tetrahedral centers should already be okay...
    group_templates += [
        ((1, 2,), '[*]/[CH]=[CH]/[*]'),  # trans with two hydrogens
        ((1, 2,), '[*]/[CH]=[CH]\[*]'),  # cis with two hydrogens
        ((1, 2,), '[*]/[CH]=[CH0]([*])\[*]'),  # trans with one hydrogens
        ((1, 2,), '[*]/[D3;H1]=[!D1]'),  # specified on one end, can be N or C
    ]

    """
    Additional group templates added 
    See publication:
    Thakkar A, Kogej T, Reymond JL, Engkvist O, Bjerrum EJ,  Datasets and Their Influence on the Development of Computer Assisted Synthesis Planning Tools in the Pharmaceutical Domain ChemRxiv
    """
    # Carried over from previous iteration (before merge with rdchiral)
    # There is some redundancy from merge with rdchiral and templates are not general
    group_templates += [
        (range(3), 'C(=O)Cl'),  # acid chloride
        (range(4), '[$(N-!@[#6])](=!@C=!@O)'),  # isocyanate
        (range(2), 'C=O'),  # carbonyl
        (range(4), 'ClS(Cl)=O'),  # thionyl chloride
        (range(5), '[#6]S(=O)(=O)[O]'),  # RSO3 leaving group
        (range(5), '[O]S(=O)(=O)[O]'),  # SO4 group
        (range(3), '[N-]=[N+]=[C]'),  # diazo-alkyl
        (range(15), '[#6][C]([#6])([#6])[#8]-[#6](=[O])-[#8]-[#6](=O)-[#8]C([#6])([#6])[#6]'),  # Boc Anhydride
        # Protecting Groups
        # Amino
        (range(18), '[#7]-[#6](=O)-[#8]-[#6]-[#6]-1-c2ccccc2-c2ccccc-12'),  # Fmoc
        (range(8), '[#6]C([#6])([#6])[#8]-[#6](-[#7])=O'),  # Boc
        (range(11), '[#7]-[#6](=O)-[#8]-[#6]-c1ccccc1'),  # Cbz
        (range(6), '[#7]-[#6](=O)-[#8]-[#6]=[#6]'),  # Voc
        (range(7), '[#7]-[#6](=O)-[#8]-[#6]-[#6]=[#6]'),  # Alloc
        (range(4), '[#6]-[#6](-[#7])=O'),  # Acetamide
        (range(7), '[#7]-[#6](=O)C(F)(F)F'),  # Trifluoroacetamide
        (range(7), '[#7]-[#6](=O)C(Cl)(Cl)Cl'),  # Trichloroacetamide
        (range(11), 'O=[#6]-1-[#7]-[#6](=O)-c2ccccc-12'),  # Phthalimide
        (range(8), '[#7]-[#6]-c1ccccc1'),  # Benzylamine
        (range(10), '[#6]-[#8]-c1ccc(-[#6]-[#7])cc1'),  # p-Methoxybenzyl (PMB)
        (range(9), '[#6]-[#8]-c1ccc(-[#7])cc1'),  # PMP
        (range(20), '[#7]C(c1ccccc1)(c1ccccc1)c1ccccc1'),  # Triphenylmethylamine
        (range(8), '[#7]=[#6]-c1ccccc1'),  # N-Benzylidene
        (range(11), '[#6]-c1ccc(cc1)S([#7])(=O)=O'),  # p-Toluenesulfonamide
        (range(13), '[#7]S(=O)(=O)c1ccc(cc1)-[#7+](-[#8-])=O'),  # p-Nosylsulfonamide
        (range(5), '[#6]S([#7])(=O)=O'),  # Mesylate
        (range(7), '[#7]-[#6]-1-[#6]-[#6]-[#6]-[#6]-[#8]-1'),  # N-Tetrahydropyranyl ether (N-THP)
        (range(14), '[#7]=[#6](-c1ccccc1)-c1ccccc1'),  # N-Benzhydrylidene
        (range(14), '[#7]-[#6](-c1ccccc1)-c1ccccc1'),  # N-Benzhydryl
        (range(5), '[#6]-[#7](-[#6])-[#6]=[#7]'),  # Dimethylaminomethylene
        (range(9), '[#7]-[#6](=O)-c1ccccc1'),  # N-Benzoate (N-Bz)
        (range(8), '[#7]=[#6]-c1ccccc1'),  # N-Benzylidene
        # Carbonyl
        (range(6), '[#6]-[#8]-[#6](-[#6])-[#8]-[#6]'),  # Dimethyl acetal
        (range(6), '[#6]-1-[#6]-[#8]-[#6]-[#8]-[#6]-1'),  # 1,3-Dioxane
        (range(5), '[#6]-1-[#6]-[#8]-[#6]-[#8]-1'),  # 1,3-Dioxolane
        (range(5), '[#6]-1-[#6]-[#16]-[#6]-[#16]-1'),  # 1,3-Dithiane
        (range(6), '[#6]-1-[#6]-[#16]-[#6]-[#16]-[#6]-1'),  # 1,3-Dithiolane
        (range(5), '[#6]-[#7](-[#6])-[#7]=[#6]'),  # N,N-Dimethylhydrazone
        (range(8), '[#6]C1([#6])[#6]-[#8]-[#6]-[#8]-[#6]1'),  # 1,3-Dioxolane (dimethyl)
        # Carboxyl
        (range(5), '[#6]-[#8]-[#6](-[#6])=O'),  # Methyl ester
        (range(6), '[#6]-[#6]-[#8]-[#6](-[#6])=O'),  # Ethyl ester
        (range(8), '[#6]-[#6](=O)-[#8]C([#6])([#6])[#6]'),  # t-Butyl ester
        (range(11), '[#6]-[#6](=O)-[#8]-[#6]-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1'),  # Benzyl ester
        (range(8), '[#6]-[#6](=O)-[#8][Si]([#6])([#6])[#6]'),  # Silyl Ester
        (range(8), '[#6]-[#6](=O)-[#16]C([#6])([#6])[#6]'),  # S-t-Butyl ester
        (range(6), '[#6]-[#6]-1=[#7]-[#6]-[#6]-[#8]-1'),  # 2-Alkyl-1,3-oxazoline
        # Hydroxyl
        (range(5), '[#6]-[#8]-[#6]-[#8]-[#6]'),  # Methoxymethyl ether
        (range(8), '[#6]-[#8]-[#6]-[#6]-[#8]-[#6]-[#8]-[#6]'),  # 2-Methoxyethoxymethyl ether
        (range(9), '[#8]-[#6](=O)-[#8]-[#6]C(Cl)(Cl)Cl'),  # 2,2,2-Trichloroethyl carbonate
        (range(7), '[#6]-[#8]C([#6])([#6])[#8]-[#6]'),  # Methoxypropyl acetal
        (range(7), '[#6]-[#6]-[#8]-[#6](-[#6])-[#8]-[#6]'),  # Ethoxyethyl acetal
        (range(11), '[#6]-[#8]-[#6]-[#8]-[#6]-[#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1'),  # Benzyloxymethyl acetal
        (range(8), '[#6]-[#8]-[#6]-1-[#6]-[#6]-[#6]-[#6]-[#8]-1'),  # Tetrahydropyranyl ether
        (range(10), '[#6]-[#8]-[#6](=O)-c1ccccc1'),  # Benzoate (O-Bz)
        (range(6), '[#6]-[#8]C([#6])([#6])[#6]'),  # t-Butyl ether
        (range(5), '[#6]-[#8]-[#6]-[#6]=[#6]'),  # Allyl ether
        (range(13), '[#6]-[#8]-[#6]-c1ccc2ccccc2c1'),  # Napthyl ether
        (range(11), '[#6]-[#8]-[#6]-c1ccc(-[#8]-[#6])cc1'),  # p-Methoxybenzyl ether
        (range(6), '[#6]-[#8][Si]([#6])([#6])[#6]'),  # Trimethylsilyl ether
        (range(8), '[#6]-[#6](-[#6])-[#6][Si]([#6])([#6])[#8]'),  # dimethylisopropylsilyl ether
        (range(9), '[#6]-[#6][Si]([#6]-[#6])([#6]-[#6])[#8]-[#6]'),  # Triethylsilyl ether
        (range(12), '[#6]-[#8][Si]([#6](-[#6])-[#6])([#6](-[#6])-[#6])[#6](-[#6])-[#6]'),  # Triisopropylsilyl ether
        (range(9), '[#6]-[#8][Si]([#6])([#6])C([#6])([#6])[#6]'),  # t-Butyldimethylsilyl ether
        (range(10), '[#6]-[#8]-[#6]-[#8]-[#6]-[#6][Si]([#6])([#6])[#6]'),  # [2-(trimethylsilyl)ethoxy]methyl
        (range(19),
         '[#6]-[#8][Si]([#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1)([#6]-1=[#6]-[#6]=[#6]-[#6]=[#6]-1)C([#6])([#6])[#6]'),
        # t-Butyldiphenylsilyl ether
        (range(9), '[#6]-[#8][Si]([#6])([#6])C([#6])([#6])[#6]'),  # t-Butyldimethylsilyl ether
        (range(20), '[#6]-[#6]-[#8][Si]([#6]-[#6]-[#6]-c1ccc(-[#6])cc1)([#6](-[#6])-[#6])[#6](-[#6])-[#6]'),
        # propyl-p-toluenediisopropylsilyl ether
        (range(5), '[#6]-[#8]-[#6](-[#6])=O'),  # Acetic acid ester (acetate)
        (range(8), '[#6]-[#8]-[#6](=O)C([#6])([#6])[#6]'),  # Pivalic acid ester
        (range(10), '[#6]-[#8]-[#6](=O)-c1ccccc1'),  # Benzoic acid ester
        (range(9), '[#6]-[#8]-[#6]-c1ccccc1'),  # Benzyl ether (O-Bn)
        # 1',2- 1',3-Diols
        (range(8), '[#6]C1([#6])[#8]-[#6]-[#6]-[#6]-[#8]1'),  # Acetonide
        (range(12), '[#6]-1-[#6]-[#8]-[#6](-[#8]-[#6]-1)-c1ccccc1'),  # Benzylidene acetal
        # Alkyne
        (range(6), '[#6][Si]([#6])([#6])C#C'),  # Trimethylsilyl ether (TMS-alkyne)
        (range(9), '[#6]-[#6][Si]([#6]-[#6])([#6]-[#6])C#C'),  # Triethylsilyl ether (TES-alkyne)
        # Thiol
        (range(5), '[#6]-[#16]-[#6](-[#6])=O'),  # S-Carbonyl deprotection (thioester)
        (range(15), '[#6]-[#6]-[#6](-[#6]-[#6])-[#6]C1([#6]-[#6]-[#6]-[#6]-[#6]1)[#6](-[#16])=O'),
        # S-Carbonyl deprotection (thioester)
        (range(9), '[#6]-[#6]-[#7](-[#6]-[#6])-[#6](=O)-[#16]-[#6]'),  # N,N-ethyl thiocarbamate
        (range(7), '[#6]-[#16]-[#6](=O)-[#7](-[#6])-[#6]'),  # N,N-methyl thiocarbamate
    ]

    # Build list
    groups = []
    for (add_if_match, template) in group_templates:
        #         matches是id的组合
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(template), useChirality=True)
        for match in matches:
            add_if = []
            for pattern_idx, atom_idx in enumerate(match):
                if pattern_idx in add_if_match:
                    add_if.append(atom_idx)
            groups.append((add_if, match))
    return groups


def expand_atoms_to_use(mol, atoms_to_use, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdX which should be included
    in the reaction, this function expands the list of AtomIdXs to include one 
    nearest neighbor with special consideration of (a) unimportant neighbors and
    (b) important functional groupings
    给定一个RDKit分子和一个应该包含在反应中的AtomIdX列表，此函数扩展了AtomIdX的列表，以包括一个最近邻，
    并特别考虑（a）不重要的邻居和（b） 重要的职能分组
    '''

    # Copy
    new_atoms_to_use = atoms_to_use[:]

    # Look for all atoms in the current list of atoms to use在当前要使用的原子列表中查找所有原子
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atoms_to_use: continue
        # Ensure membership of changed atom is checked against group 确保根据组检查更改的原子的成员资格
        for group in groups:
            if int(atom.GetIdx()) in group[0]:
                if VERBOSE:
                    #                     由于匹配而添加组
                    print('adding group due to match')
                    try:
                        print('Match from molAtomMapNum {}'.format(
                            atom.GetProp('molAtomMapNumber'),
                        ))
                    except KeyError:
                        pass
                for idx in group[1]:
                    if idx not in atoms_to_use:
                        new_atoms_to_use.append(idx)
                        symbol_replacements.append((idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(idx))))
        # Look for all nearest neighbors of the currently-included atoms
        for neighbor in atom.GetNeighbors():
            # Evaluate nearest neighbor atom to determine what should be included
            new_atoms_to_use, symbol_replacements = \
                expand_atoms_to_use_atom(mol, new_atoms_to_use, neighbor.GetIdx(),
                                         groups=groups, symbol_replacements=symbol_replacements)

    return new_atoms_to_use, symbol_replacements


def expand_atoms_to_use_atom(mol, atoms_to_use, atom_idx, groups=[], symbol_replacements=[]):
    '''Given an RDKit molecule and a list of AtomIdx which should be included
    in the reaction, this function extends the list of atoms_to_use by considering 
    a candidate atom extension, atom_idx
    给定一个RDKit分子和一个应该包含在反应中的AtomIdx列表，这个函数通过考虑候选原子扩展来扩展atoms_to_use列表，atom_idx
    '''

    # See if this atom belongs to any special groups (highest priority)
    found_in_group = False
    for group in groups:  # first index is atom IDs for match, second is what to include
        if int(atom_idx) in group[0]:  # int correction
            if VERBOSE:
                print('adding group due to match')
                try:
                    print('Match from molAtomMapNum {}'.format(
                        mol.GetAtomWithIdx(atom_idx).GetProp('molAtomMapNumber'),
                    ))
                except KeyError:
                    pass
            # Add the whole list, redundancies don't matter 
            # *but* still call convert_atom_to_wildcard!
            for idx in group[1]:
                if idx not in atoms_to_use:
                    atoms_to_use.append(idx)
                    symbol_replacements.append((idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(idx))))
            found_in_group = True
    if found_in_group:
        return atoms_to_use, symbol_replacements

    # How do we add an atom that wasn't in an identified important functional group?
    # Develop generalized SMARTS symbol

    # Skip current candidate atom if it is already included
    if atom_idx in atoms_to_use:
        return atoms_to_use, symbol_replacements

    # Include this atom
    atoms_to_use.append(atom_idx)

    # Look for suitable SMARTS replacement
    symbol_replacements.append((atom_idx, convert_atom_to_wildcard(mol.GetAtomWithIdx(atom_idx))))

    return atoms_to_use, symbol_replacements


def convert_atom_to_wildcard(atom):
    '''This function takes an RDKit atom and turns it into a wildcard 
    using heuristic generalization rules. This function should be used
    when candidate atoms are used to extend the reaction core for higher
    generalizability'''

    # Is this a terminal atom? We can tell if the degree is one
    if atom.GetDegree() == 1:
        symbol = '[' + atom.GetSymbol() + ';D1;H{}'.format(atom.GetTotalNumHs())
        if atom.GetFormalCharge() != 0:
            charges = re.search('([-+]+[1-9]?)', atom.GetSmarts())
            symbol = symbol.replace(';D1', ';{};D1'.format(charges.group()))

    else:
        # Initialize
        symbol = '['

        # Add atom primitive - atomic num and aromaticity (don't use COMPLETE wildcards)
        if atom.GetAtomicNum() != 6:
            symbol += '#{};'.format(atom.GetAtomicNum())
            if atom.GetIsAromatic():
                symbol += 'a;'
        elif atom.GetIsAromatic():
            symbol += 'c;'
        else:
            symbol += 'C;'

        # Charge is important
        if atom.GetFormalCharge() != 0:
            charges = re.search('([-+]+[1-9]?)', atom.GetSmarts())
            if charges: symbol += charges.group() + ';'

        # Strip extra semicolon
        if symbol[-1] == ';': symbol = symbol[:-1]

    # Close with label or with bracket
    label = re.search('\:[0-9]+\]', atom.GetSmarts())
    if label:
        symbol += label.group()
    else:
        symbol += ']'

    if VERBOSE:
        if symbol != atom.GetSmarts():
            print('Improved generality of atom SMARTS {} -> {}'.format(atom.GetSmarts(), symbol))

    return symbol


def reassign_atom_mapping(transform):
    '''This function takes an atom-mapped reaction SMILES and reassigns 
    the atom-mapping labels (numbers) from left to right, once 
    that transform has been canonicalized.'''

    all_labels = re.findall('\:([0-9]+)\]', transform)

    # Define list of replacements which matches all_labels *IN ORDER*
    replacements = []
    replacement_dict = {}
    counter = 1
    for label in all_labels:  # keep in order! this is important
        if label not in replacement_dict:
            replacement_dict[label] = str(counter)
            counter += 1
        replacements.append(replacement_dict[label])

    # Perform replacements in order
    transform_newmaps = re.sub('\:[0-9]+\]',
                               lambda match: (':' + replacements.pop(0) + ']'),
                               transform)
    return transform_newmaps


def get_strict_smarts_for_atom(atom):
    '''
    For an RDkit atom object, generate a SMARTS pattern that
    matches the atom as strictly as possible
    '''

    symbol = atom.GetSmarts()
    if atom.GetSymbol() == 'H':
        symbol = '[#1]'

    if '[' not in symbol:
        symbol = '[' + symbol + ']'

    # Explicit stereochemistry - *before* H
    if USE_STEREOCHEMISTRY:
        if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            if '@' not in symbol:
                # Be explicit when there is a tetrahedral chiral tag
                if atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    tag = '@'
                elif atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                    tag = '@@'
                if ':' in symbol:
                    symbol = symbol.replace(':', ';{}:'.format(tag))
                else:
                    symbol = symbol.replace(']', ';{}]'.format(tag))

    if 'H' not in symbol:
        H_symbol = 'H{}'.format(atom.GetTotalNumHs())
        # Explicit number of hydrogens: include "H0" when no hydrogens present
        if ':' in symbol:  # stick H0 before label
            symbol = symbol.replace(':', ';{}:'.format(H_symbol))
        else:
            symbol = symbol.replace(']', ';{}]'.format(H_symbol))

    # Explicit degree
    if ':' in symbol:
        symbol = symbol.replace(':', ';D{}:'.format(atom.GetDegree()))
    else:
        symbol = symbol.replace(']', ';D{}]'.format(atom.GetDegree()))

    # Explicit formal charge
    if '+' not in symbol and '-' not in symbol:
        charge = atom.GetFormalCharge()
        charge_symbol = '+' if (charge >= 0) else '-'
        charge_symbol += '{}'.format(abs(charge))
        if ':' in symbol:
            symbol = symbol.replace(':', ';{}:'.format(charge_symbol))
        else:
            symbol = symbol.replace(']', ';{}]'.format(charge_symbol))

    return symbol


def expand_changed_atom_tags(changed_atom_tags, reactant_fragments):
    '''Given a list of changed atom tags (numbers as strings) and a string consisting
    of the reactant_fragments to include in the reaction transform, this function 
    adds any tagged atoms found in the reactant side of the template to the 
    changed_atom_tags list so that those tagged atoms are included in the products'''

    expansion = []
    atom_tags_in_reactant_fragments = re.findall('\:([0-9]+)\]', reactant_fragments)
    for atom_tag in atom_tags_in_reactant_fragments:
        if atom_tag not in changed_atom_tags:
            expansion.append(atom_tag)
    if VERBOSE: print('after building reactant fragments, additional labels included: {}'.format(expansion))
    return expansion


def get_fragments_for_changed_atoms(mols, changed_atom_tags, radius=0,
                                    category='reactants', expansion=[]):
    '''Given a list of RDKit mols and a list of changed atom tags, this function
    computes the SMILES string of molecular fragments using MolFragmentToSmiles 
    for all changed fragments.
    expansion: atoms added during reactant expansion that should be included and
               generalized in product fragment
    '''
    fragments = ''
    mols_changed = []
    for mol in mols:
        # Initialize list of replacement symbols (updated during expansion)
        symbol_replacements = []

        # Are we looking for special reactive groups? (reactants only)
        if category == 'reactants':
            groups = get_special_groups(mol)
        else:
            groups = []

        # Build list of atoms to use
        atoms_to_use = []
        for atom in mol.GetAtoms():
            # Check self (only tagged atoms)仅标记的原子
            if ':' in atom.GetSmarts():
                if atom.GetSmarts().split(':')[1][:-1] in changed_atom_tags:
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    if symbol != atom.GetSmarts():
                        symbol_replacements.append((atom.GetIdx(), symbol))
                    continue

        # Fully define leaving groups and this molecule participates?
        if INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS and len(atoms_to_use) > 0:
            if category == 'reactants':
                for atom in mol.GetAtoms():
                    if not atom.HasProp('molAtomMapNumber'):
                        atoms_to_use.append(atom.GetIdx())

        # Check neighbors (any atom)
        for k in range(radius):
            atoms_to_use, symbol_replacements = expand_atoms_to_use(mol, atoms_to_use,
                                                                    groups=groups,
                                                                    symbol_replacements=symbol_replacements)

        if category == 'products':
            # Add extra labels to include (for products only)
            if expansion:
                for atom in mol.GetAtoms():
                    if ':' not in atom.GetSmarts(): continue
                    label = atom.GetSmarts().split(':')[1][:-1]
                    if label in expansion and label not in changed_atom_tags:
                        atoms_to_use.append(atom.GetIdx())
                        # Make the expansion a wildcard
                        symbol_replacements.append((atom.GetIdx(), convert_atom_to_wildcard(atom)))
                        if VERBOSE: print('expanded label {} to wildcard in products'.format(label))

            # Make sure unmapped atoms are included (from products)
            #             确保包含未映射的原子（来自产物）
            for atom in mol.GetAtoms():
                if not atom.HasProp('molAtomMapNumber'):
                    atoms_to_use.append(atom.GetIdx())
                    symbol = get_strict_smarts_for_atom(atom)
                    symbol_replacements.append((atom.GetIdx(), symbol))

        # Define new symbols based on symbol_replacements
        symbols = [atom.GetSmarts() for atom in mol.GetAtoms()]
        for (i, symbol) in symbol_replacements:
            symbols[i] = symbol

        if not atoms_to_use:
            continue

        # Keep flipping stereocenters翻转立体中心 until we are happy...
        # this is a sloppy fix during extraction to achieve consistency
        tetra_consistent = False
        num_tetra_flips = 0
        while not tetra_consistent and num_tetra_flips < 100:
            mol_copy = deepcopy(mol)
            [x.ClearProp('molAtomMapNumber') for x in mol_copy.GetAtoms()]
            this_fragment = AllChem.MolFragmentToSmiles(mol_copy, atoms_to_use,
                                                        atomSymbols=symbols, allHsExplicit=True,
                                                        isomericSmiles=USE_STEREOCHEMISTRY, allBondsExplicit=True)

            # Figure out what atom maps are tetrahedral centers弄清楚哪些原子图是四面体中心
            # Set isotopes to make sure we're getting the *exact* match we want
            #             设置同位素以确保我们得到我们想要的*精确*匹配
            this_fragment_mol = AllChem.MolFromSmarts(this_fragment)
            tetra_map_nums = []
            for atom in this_fragment_mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
                    if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                        tetra_map_nums.append(atom.GetProp('molAtomMapNumber'))
            map_to_id = {}
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    atom.SetIsotope(int(atom.GetProp('molAtomMapNumber')))
                    map_to_id[atom.GetProp('molAtomMapNumber')] = atom.GetIdx()

            # Look for matches
            tetra_consistent = True
            all_matched_ids = []

            # skip substructure matching if there are a lot of fragments
            #             如果有很多片段，则跳过子结构匹配
            # this can help prevent GetSubstructMatches from hanging 
            frag_smi = Chem.MolToSmiles(this_fragment_mol)
            if frag_smi.count('.') >= 1:
                break

            # for matched_ids in mol.GetSubstructMatches(this_fragment_mol, useChirality=True):
            #    all_matched_ids.extend(matched_ids)
            shuffle(tetra_map_nums)
            for tetra_map_num in tetra_map_nums:
                if VERBOSE: print('Checking consistency of tetrahedral {}'.format(tetra_map_num))
                # print('Using fragment {}'.format(Chem.MolToSmarts(this_fragment_mol, True)))
                if map_to_id[tetra_map_num] not in all_matched_ids:
                    tetra_consistent = False
                    if VERBOSE: print('@@@@@@@@@@@ FRAGMENT DOES NOT MATCH PARENT MOL @@@@@@@@@@@@@@')
                    if VERBOSE: print('@@@@@@@@@@@ FLIPPING CHIRALITY SYMBOL NOW      @@@@@@@@@@@@@@')
                    prevsymbol = symbols[map_to_id[tetra_map_num]]
                    if '@@' in prevsymbol:
                        symbol = prevsymbol.replace('@@', '@')
                    elif '@' in prevsymbol:
                        symbol = prevsymbol.replace('@', '@@')
                    else:
                        raise ValueError('Need to modify symbol of tetra atom without @ or @@??')
                    symbols[map_to_id[tetra_map_num]] = symbol
                    num_tetra_flips += 1
                    # IMPORTANT: only flip one at a time
                    break

                    # Clear isotopes
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

        if not tetra_consistent:
            # raise ValueError('Could not find consistent tetrahedral mapping, {} centers'.format(len(tetra_map_nums)))
            # Note: 为了不影响进度条展示，临时撰写。
            raise ValueError

        fragments += '(' + this_fragment + ').'
        mols_changed.append(Chem.MolToSmiles(clear_mapnum(Chem.MolFromSmiles(Chem.MolToSmiles(mol, True))), True))

    # auxiliary template information: is this an intramolecular reaction or dimerization?
    intra_only = (1 == len(mols_changed))
    dimer_only = (1 == len(set(mols_changed))) and (len(mols_changed) == 2)

    return fragments[:-1], intra_only, dimer_only


def canonicalize_transform(transform):
    '''This function takes an atom-mapped SMARTS transform and
    converts it to a canonical form by, if nececssary, rearranging
    the order of reactant and product templates and reassigning
    atom maps.'''

    transform_reordered = '>>'.join([canonicalize_template(x) for x in transform.split('>>')])
    return reassign_atom_mapping(transform_reordered)


def canonicalize_template(template):
    '''This function takes one-half of a template SMARTS string 
    (i.e., reactants or products) and re-orders them based on
    an equivalent string without atom mapping.'''

    # Strip labels to get sort orders
    template_nolabels = re.sub('\:[0-9]+\]', ']', template)

    # Split into separate molecules *WITHOUT wrapper parentheses*
    template_nolabels_mols = template_nolabels[1:-1].split(').(')
    template_mols = template[1:-1].split(').(')

    # Split into fragments within those molecules
    for i in range(len(template_mols)):
        nolabel_mol_frags = template_nolabels_mols[i].split('.')
        mol_frags = template_mols[i].split('.')

        # Get sort order within molecule, defined WITHOUT labels
        sortorder = [j[0] for j in sorted(enumerate(nolabel_mol_frags), key=lambda x: x[1])]

        # Apply sorting and merge list back into overall mol fragment
        template_nolabels_mols[i] = '.'.join([nolabel_mol_frags[j] for j in sortorder])
        template_mols[i] = '.'.join([mol_frags[j] for j in sortorder])

    # Get sort order between molecules, defined WITHOUT labels
    sortorder = [j[0] for j in sorted(enumerate(template_nolabels_mols), key=lambda x: x[1])]

    # Apply sorting and merge list back into overall transform
    template = '(' + ').('.join([template_mols[i] for i in sortorder]) + ')'

    return template


def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])


def extract_from_reaction(reaction):
    reactants = mols_from_smiles_list(replace_deuterated(reaction['reactants']).split('.'))
    products = mols_from_smiles_list(replace_deuterated(reaction['products']).split('.'))

    # if rdkit cant understand molecule, return
    if None in reactants: return {'reaction_id': reaction['_id']}
    if None in products: return {'reaction_id': reaction['_id']}

    # try to sanitize molecules
    try:
        for i in range(len(reactants)):
            reactants[i] = AllChem.RemoveHs(reactants[i])  # *might* not be safe
        for i in range(len(products)):
            products[i] = AllChem.RemoveHs(products[i])  # *might* not be safe
        [Chem.SanitizeMol(mol) for mol in reactants + products]  # redundant w/ RemoveHs
        [mol.UpdatePropertyCache() for mol in reactants + products]
    except Exception as e:
        # can't sanitize -> skip
        print(e)
        print('Could not load SMILES or sanitize')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    are_unmapped_product_atoms = False
    extra_reactant_fragment = ''
    for product in products:
        prod_atoms = product.GetAtoms()
        if sum([a.HasProp('molAtomMapNumber') for a in prod_atoms]) < len(prod_atoms):
            if VERBOSE: print('Not all product atoms have atom mapping')
            if VERBOSE: print('ID: {}'.format(reaction['_id']))
            are_unmapped_product_atoms = True

    if are_unmapped_product_atoms:  # add fragment to template
        for product in products:
            # Get unmapped atoms
            unmapped_ids = [
                a.GetIdx() for a in prod_atoms if not a.HasProp('molAtomMapNumber')
            ]
            if len(unmapped_ids) > MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS:
                # Skip this example - too many unmapped product atoms!
                return
            # Define new atom symbols for fragment with atom maps, generalizing fully
            atom_symbols = ['[{}]'.format(a.GetSymbol()) for a in prod_atoms]
            # And bond symbols...
            bond_symbols = ['~' for b in product.GetBonds()]
            if unmapped_ids:
                extra_reactant_fragment += AllChem.MolFragmentToSmiles(
                    product, unmapped_ids,
                    allHsExplicit=False, isomericSmiles=USE_STEREOCHEMISTRY,
                    atomSymbols=atom_symbols, bondSymbols=bond_symbols
                ) + '.'
        if extra_reactant_fragment:
            extra_reactant_fragment = extra_reactant_fragment[:-1]
            if VERBOSE: print('    extra reactant fragment: {}'.format(extra_reactant_fragment))

        # Consolidate repeated fragments (stoichometry)
        extra_reactant_fragment = '.'.join(sorted(list(set(extra_reactant_fragment.split('.')))))

    if None in reactants + products:
        print('Could not parse all molecules in reaction, skipping')
        print('ID: {}'.format(reaction['_id']))
        return {'reaction_id': reaction['_id']}

    # Calculate changed atoms
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    if err:
        if VERBOSE:
            print('Could not get changed atoms')
            print('ID: {}'.format(reaction['_id']))
        return
    if not changed_atom_tags:
        if VERBOSE:
            print('No atoms changed?')
            print('ID: {}'.format(reaction['_id']))
        # print('Reaction SMILES: {}'.format(example_doc['RXN_SMILES']))
        return {'reaction_id': reaction['_id']}

    try:
        # Get fragments for reactants
        reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(reactants, changed_atom_tags,
                                                                                     radius=1, expansion=[],
                                                                                     category='reactants')
        # Get fragments for products 
        # (WITHOUT matching groups but WITH the addition of reactant fragments)
        product_fragments, _, _ = get_fragments_for_changed_atoms(products, changed_atom_tags,
                                                                  radius=0,
                                                                  expansion=expand_changed_atom_tags(changed_atom_tags,
                                                                                                     reactant_fragments),
                                                                  category='products')
    except ValueError as e:
        if VERBOSE:
            print(e)
            print(reaction['_id'])
        return {'reaction_id': reaction['_id']}

    # Put together and canonicalize (as best as possible)
    rxn_string = '{}>>{}'.format(reactant_fragments, product_fragments)
    rxn_canonical = canonicalize_transform(rxn_string)
    # Change from inter-molecular to intra-molecular 
    rxn_canonical_split = rxn_canonical.split('>>')
    rxn_canonical = rxn_canonical_split[0][1:-1].replace(').(', '.') + \
                    '>>' + rxn_canonical_split[1][1:-1].replace(').(', '.')

    reactants_string = rxn_canonical.split('>>')[0]
    products_string = rxn_canonical.split('>>')[1]

    retro_canonical = products_string + '>>' + reactants_string

    # Load into RDKit
    rxn = AllChem.ReactionFromSmarts(retro_canonical)
    if rxn.Validate()[1] != 0:
        print('Could not validate reaction successfully')
        print('ID: {}'.format(reaction['_id']))
        print('retro_canonical: {}'.format(retro_canonical))
        if VERBOSE: raw_input('Pausing...')
        return {'reaction_id': reaction['_id']}

    template = {
        'products': products_string,
        'reactants': reactants_string,
        'reaction_smarts': retro_canonical,
        'intra_only': intra_only,
        'dimer_only': dimer_only,
        'reaction_id': reaction['_id'],
        'necessary_reagent': extra_reactant_fragment,
    }

    return template
