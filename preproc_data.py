import numpy as wqd
import cwd_files, nltk_files

# FUNCION: Preprocesamiento de archivos del corpus de entrenamiento
def preproc_files(pathsFiles:list, reviewTextFile:str, vocabularyTextFile: str):
    textClean = [] # Guarda el texto preprocesado
    vocabulary = [] # Guarda el vocabulario del corpus

    for i in range(900):
        # Se abre el archivo actual para que su contenido sea preprocesado
        contentFile = open(str(pathsFiles[i]), "r").read().splitlines()
        # Se obtiene el texto preproesado
        textClean = nltk_files.cleantext(contentFile)
        # Se hace una copia de seguridad del archivo donde se guardara todo el texto preprocesado de cada archivo para que no se sobreescriba
        autoSave = open(reviewTextFile, "r").read()
        # Se escribe el texto preprocesado en el archivo general de reseñas
        newcontentFile = open(reviewTextFile, "w")
        newcontentFile.write(autoSave+" ".join(textClean)+"\n")
        newcontentFile.close()

        # Se obtiene el vocabulario de las reseñas indicadas en los parametros
        vocabulary = nltk_files.getvocabulary(reviewTextFile)
        vocabularyFile = open(vocabularyTextFile, "w")
        vocabularyFile.write("\n".join(vocabulary))
        vocabularyFile.close()

    print("[WRITTEN] File: "+reviewTextFile)
    print("[CREATED] File: "+vocabularyTextFile+" Length: "+str(len(vocabulary))+".")

    # Se crea la union del vocabulario de ambos corpus
    autoSave = open("Vocabulary.txt","r").read()
    vocabulary = set(autoSave.split()+vocabulary)
    vocabulary = sorted(list(vocabulary))
    testVocabulary = open("Vocabulary.txt","w")
    testVocabulary.write("\n".join(vocabulary))
    testVocabulary.close()

    print("[WRITTEN] File: "+"Vocabulary.txt"+" New length: "+str(len(vocabulary))+".")

# FUNCION: Preprocesamiento de archivos del corpus de prueba
def preproc_test_files(pathsFiles:list, reviewTextFile:str):
    textClean = [] # Guarda el texto preprocesado
    vocabulary = [] # Guarda el vocabulario del corpus
    for i in range(100):
        # Se abre el archivo actual para que su contenido sea preprocesado
        contentFile = open(str(pathsFiles[i]), "r").read().splitlines()
        # Se obtiene el texto preproesado
        textClean = nltk_files.cleantext(contentFile)
        # Se hace una copia de seguridad del archivo donde se guardara todo el texto preprocesado de cada archivo para que no se sobreescriba
        autoSave = open(reviewTextFile, "r").read()
        # Se escribe el texto preprocesado en el archivo general de reseñas
        newcontentFile = open(reviewTextFile, "w")
        newcontentFile.write(autoSave+" ".join(textClean)+"\n")
        newcontentFile.close()
    print("[WRITTEN] File: "+reviewTextFile)


    # Se crea la union del vocabulario de ambos corpus
    autoSave = open("Vocabulary.txt","r").read()
    vocabulary = set(autoSave.split()+vocabulary)
    vocabulary = sorted(list(vocabulary))
    testVocabulary = open("Vocabulary.txt","w")
    testVocabulary.write("\n".join(vocabulary))
    testVocabulary.close()

    print("[WRITTEN] File: "+"Vocabulary.txt"+" New length: "+str(len(vocabulary))+".")

#Obtiene la matriz de vectores de los reviews para el entrenamiento y un vocabulario más reducido
def obtenVectoresReviews(vocabulario:list, negReviews:list, posReviews:list):  
    reviews = []
    tablaTFReviews = []
    renglonTF = []
    vocabularioRed =[]

    reviews = negReviews.copy()   #Unificamos las 1800 reviews, negativas y positivas en una lista
    pReviews = posReviews.copy()  
    for r in pReviews:
        reviews.append(r)

    print("total reviews:", len(reviews))
    cc=0
    for w in vocabulario:
        aparicionesPalabra = 0
        for d in reviews:
            docList = d.split()
            wordFreq = docList.count(w)
            if wordFreq > 0:
                aparicionesPalabra += 1
            renglonTF.append(wordFreq)
        if aparicionesPalabra > 1:
            vocabularioRed.append(w)
            tablaTFReviews.append(renglonTF)
        renglonTF = []
        cc +=1
        print(cc)
    return vocabularioRed, tablaTFReviews

#Obtiene la matriz de vectores de los reviews para la prueba y un vocabulario más reducido
def obtenVectoresReviewsParaTest(vocabulario:wqd.ndarray, negReviews:list, posReviews:list):  
    reviews = []
    tablaTFReviews = []
    renglonTF = []
    vocabularioRed =[]

    reviews = negReviews.copy()   #Unificamos las 200 reviews (de prueba), negativas y positivas en una lista
    pReviews = posReviews.copy()  
    for r in pReviews:
        reviews.append(r)

    print("total reviews:", len(reviews))
    cc=0
    for w in vocabulario:
        for d in reviews:
            docList = d.split()
            wordFreq = docList.count(w)
            renglonTF.append(wordFreq)
        tablaTFReviews.append(renglonTF)
        renglonTF = []
        cc +=1
        print(cc)
    return tablaTFReviews

#FUNCION: Se generan las etiquetas para la reseñas 
def tagsreviews(numTags:int) -> list[int]:
    listTags = []

    for i in range(numTags):

        if i<numTags/2:
            listTags.append(0)
        else:
            listTags.append(1)
    return listTags

# FUNCION: Principal
def main():

    
    pathsFilesN = list # Rutas de los archivos con reseñas negativas del conjunto de entrenamiento
    pathsFilesP = list # Rutas de los archivos con reseñas positivas del conjunto de entrenamiento

    pathsFilesN = cwd_files.getcwd(0)
    pathsFilesP = cwd_files.getcwd(1)
    
    pathsFilesNT = cwd_files.getcwd(2)
    pathsFilesPT = cwd_files.getcwd(3)
    
    # Se crean los archivos donde se guardaran las reseñas y el vocabulario (datos de entrenamiento)
    createFile = open("NegativeReviewFiles.txt", "w")
    createFile.close()
    createFile = open("PositiveReviewFiles.txt", "w")
    createFile.close()
    createFile = open("Vocabulary.txt", "w")
    createFile.close()
    
    # Se crean los archivos donde se guardaran las reseñas y el vocabulario (datos de prueba)
    createFile = open("NegativeReviewFilesTest.txt", "w")
    createFile.close()
    createFile = open("PositiveReviewFilesTest.txt", "w")
    createFile.close()
    
    
    # Se preprocesan los archivos de ambos corpus (reseñas negativas y positivas) de entrenamiento
    preproc_files(pathsFilesN, "NegativeReviewFiles.txt", "VocabularyNTest.txt")
    preproc_files(pathsFilesP, "PositiveReviewFiles.txt", "VocabularyPTest.txt")

    
    # Se preprocesan los archivos de ambos corpus (reseñas negativas y positivas) de prueba
    preproc_test_files(pathsFilesNT, "NegativeReviewFilesTest.txt")
    preproc_test_files(pathsFilesPT, "PositiveReviewFilesTest.txt")
    
    

    ##############################################################################################################################################

    
    #Carga el archivo del vocabulario en una lista------------------------------Training data-------------------------
    vocabularyFile = open("Vocabulary.txt","r").read().split()    
    #Carga el archivo de los reviews negativos en una lista
    negativeReviews = open("NegativeReviewFiles.txt","r").read().splitlines()
    #Carga el archivo de los reviews positivos en una lista
    positiveReviews = open("PositiveReviewFiles.txt","r").read().splitlines()
    #Obtiene un vocabulario reducido y su matriz con los vectores de los reviews positivos y negativos
    vocabularioRed, tablaTFReviews = obtenVectoresReviews(vocabularyFile, negativeReviews, positiveReviews)
    print("La longitud del vocabulario reducido v1 es:",len(vocabularioRed))

    #Convierte la lista del vocabulario reducido a un array del vocabulario reducido
    arrayVocabulario = wqd.array(vocabularioRed)
    #Convierte la lista listas de la tablaTFReviews a una matriz
    matrizTFReviews = wqd.array(tablaTFReviews)
    #obtiene la matriz transpuesta y ahora cada renglon es un vector que representa una review
    matrizTFReviews = wqd.transpose(matrizTFReviews)

    wqd.savetxt("vocabularioReducido.txt",arrayVocabulario,fmt='%s')
    wqd.savetxt("cTrain.txt",matrizTFReviews,fmt='%d')
    
    #Carga el vocabulario reducido del archivo vocabularioReducido.txt------------------Test data-----------------------------
    vocabularioR = wqd.loadtxt("vocabularioReducido.txt",dtype=str)
    #Carga el archivo de los reviews negativos para el test en una lista
    negativeReviewsT = open("NegativeReviewFilesTest.txt","r").read().splitlines()
    #Carga el archivo de los reviews positivos para el test en una lista
    positiveReviewsT = open("PositiveReviewFilesTest.txt","r").read().splitlines()
    #Obtiene una matriz con los vectores de los reviews positivos y negativos (para el test o prueba)
    tablaTFReviewsT = obtenVectoresReviewsParaTest(vocabularioR, negativeReviewsT, positiveReviewsT)
    
    #Convierte la lista listas de la tablaTFReviewsT a una matriz
    matrizTFReviewsT = wqd.array(tablaTFReviewsT)
    #obtiene la matriz transpuesta y ahora cada renglón es un vector que representa una review
    matrizTFReviewsT = wqd.transpose(matrizTFReviewsT)

    wqd.savetxt("cTest.txt",matrizTFReviews,fmt='%d')
    
    # Se genera el vector de resultado o tags el cual nos indica que documentos son reseñas positivas y negativas para ambos corpus
    argAux = tagsreviews(1800)
    vectorResultados = wqd.array(argAux).transpose()
    wqd.savetxt("rTrain.txt", vectorResultados, fmt='%d')

    argAux = tagsreviews(200)
    vectorResultados = wqd.array(argAux).transpose()
    wqd.savetxt("rTest.txt", vectorResultados, fmt='%d')

    #Carga la matriz de entrenamiento del archivo cTrain
    cTrainEntradas = wqd.loadtxt("cTrain.txt",dtype=int)
    #Imprime al vector tf que representa a la primera reseña (columna 0)
    print(cTrainEntradas[:,0]) 
if __name__ == "__main__":
    main()