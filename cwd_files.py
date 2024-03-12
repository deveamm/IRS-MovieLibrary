import os

pathRoot = os.getcwd() # Ruta actual del programa

# Consigue las rutas de cada archivo de la carpeta de archivos de prueba
def getcwd(flag:int) -> list:
    nameFiles = []
    pwd = ""
    pathTest = ""

    if flag == 0:
        pathTest = "\\train\\n"
    elif flag == 1:
        pathTest = "\\train\\p"
    elif flag == 2:
        pathTest = "\\test\\n"    
    elif flag == 3:
        pathTest = "\\test\\p"
        
    # Se obtiene la ruta actual
    pwd = os.getcwd()
    # Se guarda la ruta actual
    pathRoot = pwd
    # Se cambia al directorio indicado por la flag
    os.chdir(pwd+pathTest)
    # Se obtiene la ruta del directorio indicado por la flag
    pwd = os.getcwd()
    # Se reestablece la ruta al directorio raiz
    os.chdir(pathRoot)

    # Se obtienen todas las rutas de cada archivo del corpus de prueba
    for file in os.listdir(pwd):
        nameFiles.append(pwd+"\\"+file)

    return nameFiles