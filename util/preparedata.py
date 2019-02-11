import pickle
import numpy
import re
from scipy import sparse
import argparse

def tokenize(sentences):
	words = []
	for sentence in sentences:
		w = word_extraction(sentence)
		words.extend(w)
		
	words = sorted(list(set(words)))
	return words

def word_extraction(sentence):
	ignore = ['a', "the", "is"]
	words = re.sub("[^\w]", " ",  sentence).split()
	cleaned_text = [w.lower() for w in words if w not in ignore]
	return cleaned_text 

def generate_bow(vocab, sentences):
	BagOfWords = numpy.zeros([len(sentences), len(vocab)])

	for i in range(len(sentences)):
		sentence = sentences[i]
		words = word_extraction(sentence)
		
		for w in words:
			index = vocab.index(w)
			BagOfWords[i][index] += 1

	return BagOfWords

parser = argparse.ArgumentParser()
parser.add_argument("--docFile", type = str, help = "Input document file", default = "../data/raw_data/doc.tsv")
parser.add_argument("--queryFile", type = str, help = "Input query file", default = "../data/raw_data/query.tsv")
parser.add_argument("--dataPath", type = str, help = "Input query file", default = "../data/")
args = parser.parse_args()

inputFile1 = args.docFile
inputFile2 = args.queryFile
dataPath = args.dataPath

inputfiles = [inputFile1, inputFile2]
allsentences = []

for inputFile in inputfiles:
	with open(inputFile, 'r') as file:
		for line in file:
			allsentences.append(line.strip())

vocab = tokenize(allsentences)

with open(dataPath + "vocab.txt", 'w') as file:
	for value in vocab:
		file.write(value.strip() + "\n")

# print(len(vocab))

for inputFile in inputfiles:
	sentences = []

	with open(inputFile, 'r') as file:
		for line in file:
			sentences.append(line.strip())

	BagOfWords = generate_bow(vocab, sentences)
	csrBagOfWords = sparse.csr_matrix(BagOfWords)

	with open(inputFile + ".pickle", 'wb') as handle:
		pickle.dump(csrBagOfWords, handle, protocol=pickle.HIGHEST_PROTOCOL)