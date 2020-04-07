import numpy as np;
import pandas as pd;


def readData(fileName, numSequences, lenSequence):
	data = pd.read_csv(fileName, sep=" ", header=None);
	data = data.iloc[:numSequences];
	data = data.apply(lambda x: x[0][:lenSequence] if (len(x[0]) > lenSequence) else (x[0] + "0"*(lenSequence - len(x[0]))), axis=1);
	return data;


def generateSets(data):
	validationData = data[4::5];
	trainData = np.delete(data, slice(4, None, 5));
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
	
	trainData, validationData = generateSets(data.values);
	
	save('train', trainData);
	save('validation', validationData);
