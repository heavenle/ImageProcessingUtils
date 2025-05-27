# 插入指令
sql_insert = "insert into thieme_main " \
              "(id, reaction_id, product, paragraph_text, paragraph_text_cn, reactant, catalyst, " \
              "solvent, reaction_smiles, file_name, yield, doi, content_type, create_time, standby_1, standby_2) " \
              "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

