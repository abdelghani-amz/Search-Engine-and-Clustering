import nltk 
import re 
import numpy as np 
from tp2 import TermeDoc, weight
import json

def extractDocs():
	with open("CISI.ALL", "r") as fichier:
		text = fichier.read()
		fichier.close()
	regexDoc = re.compile(r'\.I([\s\S]+?)\.X', re.MULTILINE)
	docs = regexDoc.findall(text)

	return docs


def createVocab():
	docs = extractDocs()
	MotsVides = nltk.corpus.stopwords.words('english')
	stemmer = nltk.stem.PorterStemmer()

	vocab = set()
	ExpReg = nltk.RegexpTokenizer(r'(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*') 

	for i in range(len(docs)):
		docs[i] = re.sub(r"^.+\n", "", docs[i])
		docs[i] = re.sub(r"\.[T] *\n", "", docs[i])
		docs[i] = re.sub(r"\.A *\n(.+\n)+\.W *\n", "", docs[i])
		docs[i] = re.sub(r"\.B *\n\d+ *\n", "", docs[i])

		tokens = ExpReg.tokenize(docs[i])
		vocab |= {stemmer.stem(terme) for terme in tokens if terme.lower() not in MotsVides}

	vocabStr = vocab.pop()
	for i in range(len(vocab)):
			vocabStr = vocabStr + "," + vocab.pop()

	file = open("vocab.txt", "w")
	file.write(vocabStr)
	file.close()


def extractFeatures():
	vocab = open("vocab.txt", "r").read().split(",")
	docs = extractDocs()
	freqs = dict()

	for i in range(len(docs)):
		docs[i] = re.sub(r"^.+\n", "", docs[i])
		docs[i] = re.sub(r"\.[T] *\n", "", docs[i])
		docs[i] = re.sub(r"\.A *\n(.+\n)+\.W *\n", "", docs[i])
		docs[i] = re.sub(r"\.B *\n\d+ *\n", "", docs[i])


		freqs["I" + str(i+1)] = TermeDoc(docs[i], fromFile=False)

	weights = weight(freqs, vocab, stem=False)[0]
	dataset = np.zeros((len(docs), len(vocab)), np.float32)
	
	for j in range(len(vocab)):
		termWeights = weights[vocab[j]]
		for i in range(len(docs)):
			dataset[i, j] = termWeights["I" + str(i+1)]
	
	np.savetxt("weights.csv",dataset, fmt='%.8f', delimiter=",")


def saveFreqsJson():
	docs = extractDocs()

	docsJson = {}
	for i in range(len(docs)):
		docs[i] = re.sub(r"^.+\n", "", docs[i])
		docs[i] = re.sub(r"\.[T] *\n", "", docs[i])
		docs[i] = re.sub(r"\.A *\n(.+\n)+\.W *\n", "", docs[i])
		docs[i] = re.sub(r"\.B *\n\d+ *\n", "", docs[i])

		docsJson["I" + str(i+1)] = TermeDoc(docs[i], fromFile=False)

	with open('frequencies.json', 'w') as fp:
		json.dump(docsJson, fp, indent=4)
		fp.close()



def extractQueries():
	with open("CISI.QRY", "r") as fichier:
		text = fichier.read()
		fichier.close()

	regexDoc = re.compile(r'\.W([\s\S]+?)\.I', re.MULTILINE)
	docs = regexDoc.findall(text)

	ExpReg = nltk.RegexpTokenizer(r'(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*') 
	stemmer = nltk.stem.PorterStemmer()
	MotsVides = nltk.corpus.stopwords.words('english')

	queries = {}
	for i in range(len(docs)):
		docs[i] = re.sub(r"\.B *\n.+\n", "", docs[i])
		docs[i] = re.sub(r"\.W?", "", docs[i])
		queries["Q" + str(i + 1)] = [stemmer.stem(terme) for terme in set(ExpReg.tokenize(docs[i])) if terme.lower() not in MotsVides]

	with open("queries.json", "w") as fp:
		json.dump(queries, fp, indent=4)
		fp.close()
		

def extractRelevantDocs():
	with open("CISI.REL", "r") as fichier:
		lines = fichier.readlines()
		fichier.close()


	relevantDocs = dict()
	for line in lines:
		tokens = line.split()
		query = tokens[0]
		document = tokens[1]
		relevantDocs["Q" + str(query)] = relevantDocs.get("Q" + str(query), []) + [document]

	with open("relevantDocs.json", "w") as fp:
		json.dump(relevantDocs, fp, indent=4)
		fp.close()



def extractQueryFeatures():
	
	with open("queries.json", "r") as fp:
		queries = json.load(fp)
		fp.close()

	with open("vocab.txt", "r") as fp:
		vocab = fp.read().split(",")
		fp.close()

	dataset = np.zeros((len(queries), len(vocab)), np.int8)

	for j in range(len(vocab)):
		for i in range(len(queries)):
			if vocab[j] in queries["Q" + str(i+1)]:
				dataset[i, j] = 1

	np.savetxt("queryFeatures.csv",dataset, fmt='%d', delimiter=",")


	



