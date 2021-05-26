import os
import shutil

def delete_trash():
    #path da pasta raiz salvo
    start_point = os.getcwd()

    #deletar output.zip
    # if "output.zip" in os.listdir():
    #     os.remove("output.zip")

    #Remover arquivos csv's
    os.chdir(start_point+"/static/samples")
    folder1 = os.listdir()
    for i in range(len(folder1)):
        if ".csv" in folder1[i]:
            os.remove((folder1[i]))
    os.chdir(start_point)

    #Remove csv's da dataset
    os.chdir(start_point+"/static/uploads/dataset")
    folder2 = os.listdir()
    for i in range(len(folder2)):
        if ".csv" in folder2[i]:
            os.remove(folder2[i])
    os.chdir(start_point)

    # #Remove pastas de notebook
    # os.chdir("static/uploads/notebook")
    # folder3 = os.listdir()
    # for i in range(len(folder3)):
    #     shutil.rmtree(folder3[i], ignore_errors=True)
    #     print(folder3[i] + " removed !")
    # os.chdir(start_point)

