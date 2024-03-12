import nltk, string, re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Se define un conjunto predeterminado de palabras vacías en ingles 
stop_words = set(stopwords.words('english'))

# Quita palabras vacias, caracteres alfanumericos, signos de puntuacion, ... 
def cleantext(contentFile:list[str]) -> list:
    lineClean = []
    textClean = []
    listWords = []
    lemmaText = []
    re_punc = re.compile('[%s]'%re.escape(string.punctuation))

    for i in range(0,len(contentFile)):
        listWords = contentFile[i].split() 
        # Palabra por palabra filtramos las que contengan al menos un caracter alfanumérico
        for j in range(len(listWords)):
            # los conjuntos de caracteres que solo contengan signos de puntuación
            word = re_punc.sub("",listWords[j])
            # no son tomados en cuenta
            if len(word)>0 and word != "":
                # tampoco las palabras vacias   
                if word not in stop_words:
                    # las palabras con longitud menor a 3 y mayor a 10
                    if len(word) > 3 and len(word) < 10:
                        # palabras que contengan un numero
                        if re.search("\d\w+|\w+\d", word) == None:
                            # y que solo sean digitos o cadenas de digitos
                            if re.search("[^\d]", word) != None:
                                lineClean.append(word)
        # No se guardan las listas vacias
        if lineClean:
            # Se convierte la lista obetenida en string y se guarda
            textClean.extend(lineClean)
        listWords = []
        lineClean = []

    wnl = WordNetLemmatizer()
    # Se lemmatizan los sustantivos
    for lemmaWord in textClean:
        lemmaText.append(wnl.lemmatize(lemmaWord, "n"))
    # Se lemmatizan los verbos
    textClean = []
    for lemmaWord in lemmaText:
        textClean.append(wnl.lemmatize(word=lemmaWord, pos="v"))
    # Se lemmatizan los adjetivos
    lemmaText = []
    for lemmaWord in textClean:
        lemmaText.append(wnl.lemmatize(word=lemmaWord, pos="a"))
    # Se lemmatizan los adverbios
    textClean = []
    for lemmaWord in lemmaText:
        textClean.append(wnl.lemmatize(word=lemmaWord, pos="r"))
    lemmaText = []

    return textClean

# Obtiene el vocabulario 
def getvocabulary(path:str) -> list:
    contentFile = open(path,"r").read().split()
    vocabulary = []
    
    for word in contentFile:
        if word not in vocabulary:
            vocabulary.append(word)

    return vocabulary