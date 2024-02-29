from math import log, sqrt
import nltk
import re
import json

keywords = ["NOT", "AND", "OR"]

def TermeDoc(numDoc, fromFile = True):

    if fromFile:
        with open(f"D{numDoc}.txt", "r") as fichier:
            text = fichier.read()
    else :
        text = numDoc
    dico = dict()
    #Extraction des termes
    termes = list()
    ExpReg = nltk.RegexpTokenizer(r'(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*') # \d : équivalent à [0-9]
    termes = ExpReg.tokenize(text)

    #Suppression des mots vides
    MotsVides = nltk.corpus.stopwords.words('english')
    TermesSansMotsVides = [terme for terme in termes if terme.lower() not in MotsVides]
    stemmer = nltk.stem.PorterStemmer()
    TermesSansMotsVides = [stemmer.stem(i) for i in TermesSansMotsVides]
    #Comptage de la fréquence
    for t in TermesSansMotsVides:
        dico[t] = dico.get(t, 0) + 1
        
    if fromFile:
        fichier.close()
    return dico

def weight(freqs : dict, termsQ : list, stem=True, frequencyOnly = False) : 

    stemmer = nltk.stem.PorterStemmer()

    weights = {}
    freqTerms = {}

    documentCount = {}
    N = len(freqs)
    for term in termsQ:
        numOccurences = 0
        weight = {}
        freqTerm = {}
        for document in freqs.keys():
            if stem:
                term = stemmer.stem(term, to_lowercase=True)

            frequency = freqs[document].get(term, 0) 
            freqTerm[document] = frequency

            if not frequencyOnly:
                if frequency > 0:
                    numOccurences = numOccurences + 1
                    weight[document] = frequency / max(freqs[document].values())
                else:
                    weight[document] = 0
        
        if not frequencyOnly:
            for i in weight.keys():
                if weight[i] > 0:
                    weight[i] = weight[i] * log( 1 + N/numOccurences, 10)

            weights[term] = weight
        else:
            documentCount[term] = numOccurences

        freqTerms[term] = freqTerm


    if not frequencyOnly:
        return weights, freqTerms
    else:
        return documentCount, freqTerms


def removeNonPertinent(documents : dict):
    for i in list(documents.keys()):
        if documents[i] == 0 :
            del documents[i]


def scalarProduct(weights : dict) -> dict:

    scalarProducts = {}
    for term in weights.keys():
        for document in weights[term].keys():
            scalarProducts[document] = scalarProducts.get(document, 0) + weights[term][document]

    removeNonPertinent(scalarProducts)

    
    return scalarProducts



def _measure(freqs : dict, termsQ : list) -> tuple:

    """
    Helper function for cosineMeasure and jaccardMeasure
    """

    stemmer = nltk.stem.PorterStemmer()
    weightsQuery = {}
    weightsSquaredSum = {}

    termsQ = [stemmer.stem(term) for term in termsQ]
    try:
        with open("weightsSquaredSum.json", "r") as fp:
            weightsSquaredSum = json.load(fp)
            weights = weight(freqs, termsQ, stem=False)[0]
            scalar_product = scalarProduct(weights)

        pass

    except:
        
        for document in freqs.keys():
            weights = weight(freqs, freqs[document].keys(), stem=False)[0]
            for term in freqs[document].keys():
                weightsSquaredSum[document] = weightsSquaredSum.get(document, 0) + (weights[term][document])**2

            for term in termsQ:
                weightsQuery[term] = {**weightsQuery.get(term, {}) , **weights.get(term, {})}

        scalar_product = scalarProduct(weightsQuery)
        with open("weightsSquaredSum.json", "w") as fp:
            json.dump(weightsSquaredSum, fp, indent=4)
    
    return scalar_product, weightsSquaredSum


def cosineMeasure(freqs : dict, termsQ : list) -> dict:

    queryLen = len(termsQ)
    scalar_product, weightsSquaredSum = _measure(freqs, termsQ)
    cosine_measure = {}
    for document in scalar_product.keys():
        cosine_measure[document] = scalar_product[document] / sqrt(weightsSquaredSum[document] * queryLen)


    return cosine_measure


def jaccardMeasure(freqs : dict, termsQ : list) -> dict:

    queryLen = len(termsQ)
    scalar_product, weightsSquaredSum = _measure(freqs, termsQ)
    jaccard_measure = {}
    for document in scalar_product.keys():
        jaccard_measure[document] = scalar_product[document] / (weightsSquaredSum[document] + queryLen - scalar_product[document]) 


    return jaccard_measure

def BM25(freqs : dict , termsQ : list, b : float, k : float):
    

    bm25 = {}
    docCountQ, freqsQ = weight(freqs, termsQ, frequencyOnly=True)
    docLenghts = {}
    docLengthSum = 0
    for document in freqs.keys():
        docLenghts[document] = x = len(freqs[document])
        docLengthSum = docLengthSum + x

    avgdl = docLengthSum / len(freqs)

    N = len(freqs)
    for document in freqs.keys():
        dl = docLenghts[document]
        for term in freqsQ.keys():
            n = docCountQ[term]
            frequency = freqsQ[term][document]
            if frequency > 0:
                bm25[document] = bm25.get(document, 0) + (frequency * log((N - n + 0.5) / (n + 0.5), 10)) / (frequency + k*((1-b) + b*dl/avgdl)) 

    return bm25


def verifyRequest(request):

    tokens = request.split()

    if len(tokens) == 1 :
        if tokens[0] not in keywords:
            return True
        else:
            return False
    elif len(tokens) == 0:
        return False


    if tokens[0] == "AND" or tokens[0] == "OR":
        return False

    for i in range(len(tokens) - 1):
        if tokens[i] == "NOT":
            if tokens[i + 1] in keywords:
                return False
            
        elif tokens[i] == "OR" or tokens[i] == "AND":
            if tokens[i + 1] == "OR" or tokens[i + 1] == "AND":
                return False
            
        else: #operand
            if tokens[i+1] != "AND" and tokens[i+1] != "OR":
                return False
    
    if tokens[-1] in keywords:
        return False

    return True



def execRequest(request : str , freqs : dict) -> dict:

    if not verifyRequest(request):
        return None
    
    termes = []
    stemmer = nltk.stem.PorterStemmer()
    tokens = request.split()
    operators = []

    for i in tokens:
        if i not in keywords:
            termes.append(stemmer.stem(i))
        elif i != "NOT":
            operators.append(i)

    values = weight(freqs, termes, stem=False, frequencyOnly=True)[1]

    #### DEALING WITH NOT
    for i in range(len(tokens) - 1):
        if tokens[i] == "NOT":
            termFreq = values[stemmer.stem(tokens[i + 1])]
            for document in termFreq.keys():
                if termFreq[document] != 0:
                    termFreq[document] = 0
                else:
                    termFreq[document] = 1
            
    dict.pop
    ##### DEALING WITH AND / OR
    result = values.pop(termes[0])  ##Initial value
    for i in range(1, len(termes)):

        current_operand = values[termes[i]]
        current_operator = operators[i - 1]

        if current_operator == "AND":
            for document in current_operand.keys(): 
                result[document] = result[document] * current_operand[document]
        elif current_operator == "OR":
            for document in current_operand.keys(): 
                result[document] = result[document] + current_operand[document]

    return result 


if __name__ == "__main__":

    from preprocess import extractDocs  
    freqs = {}
    # for i in range(1,5):
    #     freqs["D" + str(i)] = TermeDoc(i)
    # weights = weight(freqs, ["results", "recommendation", "graph"])
    # print(weights[0])
    # # print(scalarProduct(weights[0]))
    #print(execRequest("results AND recommendation AND graph AND meme", freqs))

    docs = extractDocs()
    for i in range(len(docs)):
        docs[i] = re.sub(r"^.+\n", "", docs[i])
        docs[i] = re.sub(r"\.[T] *\n", "", docs[i])
        docs[i] = re.sub(r"\.A *\n(.+\n)+\.W *\n", "", docs[i])
        docs[i] = re.sub(r"\.B *\n\d+ *\n", "", docs[i])
        #print(docs[i])
        freqs["I" + str(i+1)] = TermeDoc(docs[i], fromFile=False)

    a = cosineMeasure(freqs, ["results", "recommendation", "graph"])
    # b = jaccardMeasure(freqs, ["results", "recommendation", "graph"])
    # c = BM25(freqs, ["results", "recommendation", "graph"], 0.75, 1.5 )
    print(a)
    # print(b)
    # print(c)


        



    

        

