import torch;
import torch.nn as nn;
import numpy as np;
import matplotlib.pyplot as plt;
from hotEncoding import generateHotEncoding;


learningRate = 0.001;


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


def train(lstm, data, criterion):
	
	for sample in data:
		hidden, cellState = lstm.initHidden(), lstm.initCellState();
		for numChar in range(sample.shape[0]-1):
			lstm.zero_grad();
			inputChar, nextChar = sample[numChar], sample[numChar+1];
			hidden, cellState = lstm(inputChar, hidden, cellState);
			target = torch.LongTensor(np.argmax(nextChar).unsqueeze(0));
			loss = criterion(hidden, target);
			loss.backward();
			hidden = hidden.detach();
			cellState = cellState.detach();
			
			for p in lstm.parameters():
				p.data.add_(-learningRate, p.grad.data);
			print(loss);


def validate(lstm, data, criterion, hidden, cellState):
	
	with torch.no_grad():
		for sample in data:
			hidden, cellState = lstm.initHidden(), lstm.initCellState();
			for numChar in range(sample.shape[0]-1):
				inputChar, nextChar = sample[numChar], sample[numChar];
				hidden, cellState = lstm(inputChar, hidden, cellState);
				target = torch.LongTensor(np.argmax(nextChar).unsqueeze(0));
				loss = criterion(hidden, target);

				print(loss);


if __name__ == "__main__":
	
	(trainData, validationData) = generateHotEncoding();
	print(trainData[0]);

	trainData, validationData = torch.Tensor(trainData), torch.Tensor(validationData);
	
	nInp, nHid = trainData.shape[2], trainData.shape[2];

	lstm = LSTM(nInp, nHid);
	
	criterion = nn.CrossEntropyLoss();
	train(lstm, trainData.unsqueeze(2), criterion);
	#validate(lstm, validationData.unsqueeze(2), criterion);
