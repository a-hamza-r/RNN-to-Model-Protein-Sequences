import numpy as np;

aminoAcids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "$"];
trainDataFile = "train.txt";
validationDataFile = "validation.txt";


def oneHotEncoding(sequence):
	oneHotEncoded = [[0 if num != count else 1 for count in range(len(aminoAcids)+1)] for num in sequence];
	return oneHotEncoded;


def loadData():
	with open(trainDataFile, "r") as train, open(validationDataFile, "r") as validation:
		trainData = [x.strip()+"$" for x in train.readlines()];
		validationData = [x.strip()+"$" for x in validation.readlines()];
	
	return (trainData, validationData);


def generateHotEncoding(inputs, labels):

	inputs = list(map(oneHotEncoding, inputs));
	labels = oneHotEncoding(labels);
	
	return (inputs, labels);


def hotEncodingToChar(encoding):
	charIndex = np.argmax(encoding);
	return aminoAcids[charIndex];


def indexToChar(index):
	return aminoAcids[index];

def charToIndex(char):
	return aminoAcids.index(char)+1;

def numberOfAminoAcids():
	return len(aminoAcids);