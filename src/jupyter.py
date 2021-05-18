# import os
# import pandas as pd
# import nbformat as nbf
# import shutil

# def jupyter_Creator():
#     # Criação de lista para armazenar arquivos
#     files = []
#     os.chdir("static/uploads/dataset")

#     # Acessar arquivos da pasta Datasets")
#     folder = os.listdir()

#     # Selecionando elementos com a extensão .csv
#     for i in range(len(folder)):
#         if ".csv" in folder[i]:
#             files.append(folder[i])

#     # Criação do dicionário
#     dict_files = {}
#     for x in range(len(files)):
#         dict_files[files[x]] = pd.read_csv(files[x], encoding="utf-8")

#     def getList(dict):
#         list = []
#         for key in dict.keys():
#             list.append(key)

#         return list

#     # Armazenar o nome das chaves
#     keys_list = getList(dict_files)


#     paste_names = []
#     notebook_names = []
#     for names in keys_list:
#         split = names.split(".csv")
#         paste_names.append(split[0])
#         notebook_names.append(split[0] + ".ipynb")


#     os.chdir("../..")
#     os.chdir("uploads/notebook")
#     notebook_paste = os.getcwd()

#     for i in range(len(paste_names)):
#         if paste_names[i] in os.listdir():
#             shutil.rmtree(paste_names[i], ignore_errors=True)


#     for i in range(len(notebook_names)):
#         if not os.path.exists(paste_names[i]):
#             os.makedirs(paste_names[i])

#         os.chdir(paste_names[i])

#         # criando notebook
#         nb = nbf.v4.new_notebook()

#         # instanciando celulas
#         text1 = """\
# # AUTO-ML-TUTOR

#         """

#         text2 = """\
# --------------------------------------------------------------
#         """

#         text3 = """\
# ## Imports
#                 """

#         code1 = """\
# import os
# import sys
# import pandas as pd
# import numpy as np
# import pandas_profiling
# import easygui
# from sklearn.preprocessing import MaxAbsScaler
#     	\
#         """

#         text4 = """\
# --------------------------------------------------------------
#                 """

#         text5 = """\
# ## Carregar dataset
#                         """

#         code2 = """\
# def get_name():
#     dirs = os.getcwd()
#     dataset_name = dirs.split("/")[-1]+".csv"
#     return dataset_name

# dataset = get_name()
# print(get_name())\
#         """

#         code3 = """\
# os.chdir("../..")
# os.chdir("dataset")\
#         """

#         code4 = """\
# dataset = os.listdir()
# dataset[0]
# df = pd.read_csv(dataset[0])\
#         """

#         text6 = """\
# --------------------------------------------------------------
#                 """

#         text7 = """\
# ## Visualizar dataset e estatísticas
#                 """

#         text8 = """\
# ### ->Segue abaixo as 15 primeiras instâncias do dataset
#                 """

#         code5 = """\
# df.head(n=15)       
#                 """

#         text9 = """\
# ### ->Medidas estatísticas de cada coluna do dataset
#                         """

#         code6 = """\
# df.describe()       
#         """

#         text10 = """\
# ### ->Uso do Pandas profiling para mais estatísticas 
#                                 """

#         code7 = """\
# profile = pandas_profiling.ProfileReport(df, title= dataset[0]+' report', html={'style':{'full_width':True}})                
# """

#         code8 = """\
# profile.to_notebook_iframe()                
#         """
#         text11 = """\
# --------------------------------------------------------------
#                   """

#         text12 = """\
# ## Escolher rótulo
#                 """

#         code9 = """\
# df_columns = list(df.columns)
#         """

#         code10 = """\
# print("Colunas do dataset:")
# df_columns
#                 """

#         code11 = """\
# Y_choiced = easygui.buttonbox("Qual coluna será seu Y ? Dica: Em alguns dataset's a coluna rotulada é a última muitas vezes.",title = "Escolher-Y", choices = df_columns )
# easygui.msgbox ("Você escolheu coluna: " + Y_choiced)
#                 """

#         text13 = """\
# ### ->Divisão do X e Y
#                         """

#         code12 = """\
# array_X = df.drop(columns = [Y_choiced])
# array_Y = df[[Y_choiced]]
#         """

#         code13 = """\
# array_X 
#                 """

#         code14 = """\
# array_Y 
#                 """

#         nb['cells'] = [nbf.v4.new_markdown_cell(text1),
#                        nbf.v4.new_markdown_cell(text2),
#                        nbf.v4.new_markdown_cell(text3),
#                        nbf.v4.new_code_cell(code1),
#                        nbf.v4.new_markdown_cell(text4),
#                        nbf.v4.new_markdown_cell(text5),
#                        nbf.v4.new_code_cell(code2),
#                        nbf.v4.new_code_cell(code3),
#                        nbf.v4.new_code_cell(code4),
#                        nbf.v4.new_markdown_cell(text6),
#                        nbf.v4.new_markdown_cell(text7),
#                        nbf.v4.new_markdown_cell(text8),
#                        nbf.v4.new_code_cell(code5),
#                        nbf.v4.new_markdown_cell(text9),
#                        nbf.v4.new_code_cell(code6),
#                        nbf.v4.new_markdown_cell(text10),
#                        nbf.v4.new_code_cell(code7),
#                        nbf.v4.new_code_cell(code8),
#                        nbf.v4.new_markdown_cell(text11),
#                        nbf.v4.new_markdown_cell(text12),
#                        nbf.v4.new_code_cell(code9),
#                        nbf.v4.new_code_cell(code10),
#                        nbf.v4.new_code_cell(code11),
#                        nbf.v4.new_markdown_cell(text13),
#                        nbf.v4.new_code_cell(code12),
#                        nbf.v4.new_code_cell(code13),
#                        nbf.v4.new_code_cell(code14)]

#         # create jupyter notebook
#         nbf.write(nb, notebook_names[i])
#         os.chdir(notebook_paste)
