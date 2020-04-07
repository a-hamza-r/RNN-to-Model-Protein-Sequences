import torch;
import torch.nn as nn;
import numpy as np;
import matplotlib.pyplot as plt;
from hotEncoding import generateHotEncoding;


class LSTM(nn.Module):
	
	def __init__(self, nInp, nHid):
		super().__init__();
		self.input = nn.Linear(nInp + nHid, nHid);
		self.forgetGate = nn.Linear(nInp + nHid, nHid);
		self.inputGate = nn.Linear(nInp + nHid, nHid);
		self.outputGate = nn.Linear(nInp + nHid, nHid);
		self.sigmoid = nn.Sigmoid();
		self.tanh = nn.Tanh();
		self.nHid = nHid;

	def forward(self, input, hidden, cellState):
		combined = torch.cat((hidden, input), dim=1);
		inputCell = self.sigmoid(self.input(combined));  # g_i
		forgetGate = self.sigmoid(self.forgetGate(combined));   #f_i
		inputGate = self.sigmoid(self.inputGate(combined));   #i_i
		outputGate = self.sigmoid(self.outputGate(combined));   #o_i
		cellState1 = forgetGate*cellState + inputGate*inputCell;   #next cell state
		hidden1 = self.tanh(cellState1)*outputGate;

		return (hidden1, cellState1);

	def initHidden(self):
		return torch.zeros(1, self.nHid);


	def initCellState(self):
		return torch.zeros(1, self.nHid);


def train(lstm, data):
	hidden, cellState = lstm.initHidden(), lstm.initCellState();
	for x in data:
		for y in range(x.shape[0]):
			input = x[y].unsqueeze(0);
			hidden, cellState = lstm(input, hidden, cellState);
			print(hidden, cellState);


if __name__ == "__main__":
	
	(trainData, validationData) = generateHotEncoding();
	#print(trainData);

	trainData, validationData = torch.Tensor(trainData), torch.Tensor(validationData);
	
	print(trainData.shape);
	print(validationData.shape);

	nInp, nHid = trainData.shape[2], trainData.shape[2];

	lstm = LSTM(nInp, nHid);

	train(lstm, trainData);
