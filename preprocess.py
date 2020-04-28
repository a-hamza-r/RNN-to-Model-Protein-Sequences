import numpy as np;
import pandas as pd;


def readData(fileName, numSequences, lenSequence):
	data = [];
	with open(fileName, "r") as file:
		data = [x.strip()[:lenSequence] for x in file.readlines()][:numSequences];
	return data;


def generateSets(data):
	validationData = np.array(data[4::5]);
	trainData = np.delete(np.array(data), slice(4, None, 5));
	return trainData, validationData;


def save(dataType, data):
	with open(dataType+".txt", "w+") as f:
		for line in data:
			f.write(line+"\n");
			

if __name__ == '__main__':
	dataFile = "pdb_seqres_truncated.txt";
	numSequences = 1000;
	lenSequence = 100;
	data = readData(dataFile, numSequences, lenSequence);
	
	trainData, validationData = generateSets(data);
	
	save('train', trainData);
	save('validation', validationData);
