import sys
import os
sys.path.append(os.path.abspath("./"))
# print(sys.path)
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from rxnmapper import RXNMapper
from ThirdPkg.template_utils.amol_utils_rdchiral import Reaction

class ChemDataProcess:
    def __init__(self):
        pass

    @staticmethod
    def read_mol(mol_path):
        """
        读取mol文件 （静态函数）
        @param mol_path: mol文件路径
        @return: mol变量
        """
        mol = Chem.MolFromMolFile(mol_path)
        return mol
    
    @staticmethod
    def read_inchi(inchi_path):
        """
        读取inchi文件 （静态函数）
        @param mol_path: inchi文件路径
        @return: inchi变量
        """
        inchi = Chem.inchi.MolFromInchi(inchi_path)
        return inchi
    
    @staticmethod
    def standard_smiles(smiles):
        """
        smiles规范化 （静态函数）
        @param smiles: smiles
        @return: 标准化smiles
        """
        m = Chem.MolFromSmiles(smiles)
        newsmi = Chem.MolToSmiles(m)
        return newsmi
    
    @staticmethod
    def compound_smart_to_smiles(smiles):
        """
        去除SMILES字符串中的原子映射编号 （静态函数）
        @param smi: 线性smiles, 不能是化学反应，只能是单一化合物
        @return: smiles
        """
        try:
            m = Chem.MolFromSmiles(smiles)
            for a in m.GetAtoms():
                a.ClearProp('molAtomMapNumber')
            newsmi = Chem.MolToSmiles(m)
            return newsmi
        except Exception as e:
            return None
               
    def reaction_smart_to_smiles(self, smart):
        """
        去除化学反应中的原子映射编号。
        @param smart: 反应smart
        @return: 反应smiles
        """
        resc = smart.split('>')
        if len(resc) == 3:
            reactant = resc[0]
            reactant_list = [self.standard_smiles(self.compound_smart_to_smiles(ri)) for ri in reactant.split('.')]
            reactant_smiles = '.'.join(reactant_list)

            condition = resc[1]
            condition_list = [self.standard_smiles(self.compound_smart_to_smiles(ci)) for ci in condition.split('.')]
            condition_smiles = '.'.join(condition_list)

            product = resc[2]
            product_list = [self.standard_smiles(self.compound_smart_to_smiles(pi)) for pi in product.split('.')]
            product_smiles = '.'.join(product_list)

            reaction_smiles = f'{reactant_smiles}>{condition_smiles}>{product_smiles}'
        else:
            reaction_smiles = ''
        return reaction_smiles
    
    @staticmethod
    def reaction_smiles_to_smart(x):
        """
        生成原子映射（映射后的结果不包含反应条件）（静态函数）
        :param x: smiles
        :param num:指定位置
        :return:
        """
        try:
            reactants, agents, products = x.split(">")
            reaction = reactants + ">>" + products
            rxn_mapper = RXNMapper()
            results = rxn_mapper.get_attention_guided_atom_maps([reaction])
            c = results[0]['mapped_rxn']
            return c
        except:
            print(x)
            return np.nan   

    @staticmethod
    def reaction_smart_to_fingerprint(reaction, input_type, radius=2, n_bits=2048, f=True):
        """
        将反应 SMARTS 转化为分子指纹。（静态函数）
        @param reaction: 反应的 SMARTS 表达式
        @param radius: Morgan 指纹半径
        @param n_bits: 指纹长度
        @param f: 是否使用功能基
        @param input_type: 输入类型，可选值为'smiles','smarts'
        @return: fingerprint
        """
        if input_type == "smarts":
            reaction = rdChemReactions.ReactionFromSmarts(reaction)
            # 提取反应的反应物和产物
            reactants = [Chem.MolFromSmiles(Chem.MolToSmiles(reactant)) for reactant in reaction.GetReactants()]
            products = [Chem.MolFromSmiles(Chem.MolToSmiles(product)) for product in reaction.GetProducts()]
        elif input_type == "smiles":
            reactant_list = reaction.split('>')[0].split('.')
            product_list = reaction.split('>')[2].split('.')
            # 提取反应的反应物和产物
            reactants = [Chem.MolFromSmiles(reactant) for reactant in reactant_list]
            products = [Chem.MolFromSmiles(product) for product in product_list]
        else:
            print("Invalid input type. Please use 'smarts' or 'smiles'.")
            return None
        
        # 生成反应物和产物的指纹,
        # useFeatures=True,将同一类功能基作为一种特征结构
        reactant_fps = [AllChem.GetMorganFingerprintAsBitVect(reactant, radius, nBits=n_bits, useFeatures=f) for reactant
                        in reactants]
        product_fps = [AllChem.GetMorganFingerprintAsBitVect(product, radius, nBits=n_bits, useFeatures=f) for product in
                    products]
        
        # 使用逻辑或操作合并所有指纹
        combined_fp = reactant_fps[0]
        for fp in reactant_fps[1:] + product_fps:
            combined_fp |= fp
        return combined_fp  
    
    @staticmethod
    def reaction_Tanimoto_similarity(fp1, fp2):
        """
        计算两个反应的相似度。（静态函数）
        @param fp1: 反应指纹1
        @param fp2: 反应指纹2
        @return: 相似度
        """
        try:
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            sim = 0
        return sim
 

    @staticmethod
    def get_reaction_template(reaction_smiles, radius):
        """
        获取反应模板。（静态函数）
        @param reaction_smiles: 反应smiles
        @param radius: 模板半径
        @return: 反应模板->str
        """
        try:
            reaction = Reaction(reaction_smiles.split(" ")[0])
            return reaction.generate_reaction_template(radius=radius)[1]
        except:
            return None
        
    @staticmethod
    def get_radis_0_(reaction_smiles):
        """
        获取去除氢原子的反应模板。（静态函数）
        @param reaction_smiles: 反应smiles
        @return: 反应模板->str
        """
        try:
            reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles)
            #  去掉氢气，去掉数字之后的情况
            rdChemReactions.RemoveMappingNumbersFromReactions(reaction)

            reaction = rdChemReactions.ReactionToSmiles(reaction)

            return reaction
        except:
            return None  
            
if __name__ == "__main__":
    chem_data_process = ChemDataProcess()
    # Example usage
    
    smi = chem_data_process.reaction_smiles_to_smart("CC(C)Oc1ccc(N)cc1.Fc1cnc(Cl)nc1Nn1cccc1>>CC(C)Oc1ccc(Nc2ncc(F)c(Nn3cccc3)n2)cc1")
    smi = chem_data_process.get_reaction_template(smi, 0)
    print(smi)



