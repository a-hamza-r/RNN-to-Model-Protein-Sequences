import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import matplotlib.pyplot as plt;
import numpy as np;
import pandas as pd;


def readData(fileName, numSequences, lenSequence):
	data = pd.read_csv(fileName, sep=" ", header=None);
	data = data.iloc[:numSequences];
	data = data.apply(lambda x: x[0][:lenSequence] if (len(x[0]) > lenSequence) else (x[0] + "0"*(lenSequence - len(x[0]))), axis=1);
	return data;


def generateSets(data):
	
	return data, data;


if __name__ == '__main__':
	dataFile = "pdb_seqres.txt";
	numSequences = 1000;
	lenSequence = 100;
	data = readData(dataFile, numSequences, lenSequence);
	
	trainData, validationData = generateSets(data);
	print(trainData);
	print(validationData);
