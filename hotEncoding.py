
aminoAcids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "$"];
trainDataFile = "train.txt";
validationDataFile = "validation.txt";


def oneHotEncoding(sequence):
	oneHotEncoded = [[0 if char != acid else 1 for acid in aminoAcids] for char in sequence];
	return oneHotEncoded;


def loadData(trainDataFile, validationDataFile):
	with open(trainDataFile, "r") as train, open(validationDataFile, "r") as validation:
		trainData = [x.strip()+"$" for x in train.readlines()];
		validationData = [x.strip()+"$" for x in validation.readlines()];
	
	return (trainData, validationData);


def generateHotEncoding():

	(trainData, validationData) = loadData(trainDataFile, validationDataFile);
	trainHotEncoded = list(map(oneHotEncoding, trainData));
	validationHotEncoded = list(map(oneHotEncoding, validationData));
	
	return (trainHotEncoded, validationHotEncoded);
		
