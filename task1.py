from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import optimizers
import keras.utils as ku 
import numpy as np 
from keras.utils import plot_model
import matplotlib.pyplot as plt
import math
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import sys
import random
from hotEncoding import generateHotEncoding, loadData, charToIndex, numberOfAminoAcids;


lenSequence = 100;
# load_previous = False

# truncate_to = 50
epochs = 25
batch_size = 32
learning_rate = 0.01

# num_train_sequences = 1000
# num_test_sequences = 50

# tokenizer = Tokenizer()
# max_seq_len = 0
# dict_size = 0

# def dataset_preparation(data, train, limit_num_sequences_to):

# 	global max_seq_len
# 	global dict_size

# 	# Split the new-line-sepeated data into individual sequences
# 	raw_sequences= data.lower().split("\n")

# 	if limit_num_sequences_to != -1:
# 		raw_sequences = raw_sequences[:limit_num_sequences_to]

# 	if train:
# 		# Tokenization
# 		tokenizer.fit_on_texts(raw_sequences)  # Recognize unique words in the data
# 		dict = tokenizer.word_index  # Dictionary mapping each unique word to a unique int
# 		dict_size = len(dict) + 1  # Dictionary size

# 	# Create input sequences using list of tokens
# 	input_sequences = []
# 	if train:
# 		max_seq_len = 0
# 	for sequence in raw_sequences:
# 		token_list = tokenizer.texts_to_sequences([sequence])[0]  # Map each word in sequence to its index in dictionary
# 		if truncate_to != -1:
# 			token_list = token_list[:truncate_to]  # Truncate each (tokenized) sequence to the required size
# 		for i in range(1, len(token_list)):
# 			# Get n-gram sequences
# 			n_gram_sequence = token_list[:i+1]
# 			input_sequences.append(n_gram_sequence)
# 		# Update max_sequence length
# 		if train:
# 			if len(input_sequences[-1]) > max_seq_len:
# 				max_seq_len = len(n_gram_sequence)

# 	# Pad input sequences
# 	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))

# 	# Create input sequences to feed to the model and their target labels
# 	inputs, labels = input_sequences[:,:-1], input_sequences[:,-1]

# 	# One-hot encode the labels
# 	labels = ku.to_categorical(labels, num_classes=dict_size)

# 	return inputs, labels, max_seq_len, dict_size


# train_data = open('train.txt').read()
# train_inputs, train_labels, max_seq_len, dict_size = dataset_preparation(train_data, train=True, limit_num_sequences_to=num_train_sequences)
# test_data = open('test.txt').read()
# test_inputs, test_labels, max_seq_len, dict_size = dataset_preparation(test_data, train=False, limit_num_sequences_to=num_test_sequences)
# # print(train_inputs.shape)
# # print(train_labels.shape)
# # print(test_inputs.shape)
# # print(test_labels.shape)

# learning rate schedule
def step_decay(epoch):
	global learning_rate
	# if epoch >= 13:
	# 	learning_rate = 0.00025
	# elif epoch >= 10:
	# 	learning_rate = 0.0005
	# elif epoch >= 6:
	# 	learning_rate = 0.0010
	# elif epoch >= 3:
	# 	learning_rate = 0.0015
	if epoch >= 19:
		learning_rate = 0.00001
	elif epoch >= 14:
		learning_rate = 0.000025
	elif epoch >= 9:
		learning_rate = 0.00005
	elif epoch >= 4:
		learning_rate = 0.0001
	# if epoch >= 44:
	# 	learning_rate = 0.00025
	# elif epoch >= 39:
	# 	learning_rate = 0.0005
	# elif epoch >= 29:
	# 	learning_rate = 0.0010
	# elif epoch >= 24:
	# 	learning_rate = 0.0015
	print("Learning Rate: %f" % learning_rate)
	return learning_rate
	# drop = 0.75
	# epochs_drop = 4
	# lrate = learning_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	# print("Learning Rate: %f" % lrate)
	# return lrate


# def run_model(inputs, labels, max_seq_len, dict_size):
	
# 	model = Sequential()
# 	model.add(Embedding(dict_size, 25, input_length=max_seq_len-1))  # Embedding layer
# 	model.add(LSTM(units=150))  # LSTM layer 1
# 	model.add(Dense(dict_size, activation='softmax'))  # Fully connected layer with softmax on top

# 	print("\n\n")
# 	print(model.summary())
# 	print("\n\n")

# 	optimizer = optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
# 	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# 	# earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='auto')  # Enabling early stopping with a patience of 5
# 	filepath="model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# 	lrate = LearningRateScheduler(step_decay)
# 	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# 	history = model.fit(inputs, labels, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(test_inputs, test_labels), callbacks=[checkpoint, lrate])	

# 	print(type(history.history['loss']))

# 	plt.plot(history.history['loss'])
# 	plt.plot(history.history['val_loss'])
# 	plt.title('Model Loss')
# 	plt.ylabel('Loss')
# 	plt.xlabel('Epoch')
# 	plt.legend(['Train', 'Test'], loc='upper left')

# 	plt.show()

# 	plt.plot(np.exp(history.history['loss']))
# 	plt.plot(np.exp(history.history['val_loss']))
# 	plt.title('Perplexity')
# 	plt.ylabel('Perplexity')
# 	plt.xlabel('Epoch')
# 	plt.legend(['Train', 'Test'], loc='upper left')

# 	plt.show()

# 	plt.plot(history.history['accuracy'])
# 	plt.plot(history.history['val_accuracy'])
# 	plt.title('Model Accuracy')
# 	plt.ylabel('Accuracy')
# 	plt.xlabel('Epoch')
# 	plt.legend(['Train', 'Test'], loc='upper left')

# 	plt.show()

# 	return model 


# def predict_seq(seed_text, num_words_to_predict, max_seq_len):
# 	for _ in range(num_words_to_predict):
# 		token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Map each word in sequence to its index in dictionary
# 		token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')  # Pre-pad the (tokenized) sequence
# 		predicted = model.predict_classes(token_list, verbose=0)  # Predict the next words
		
# 		output_word = ""
# 		for word, index in tokenizer.word_index.items():
# 			# Mapping tokenized word back to string
# 			if index == predicted:
# 				output_word = word
# 				break

# 		seed_text += " " + output_word.upper()  # Append the predicted word to the seed text

# 	return seed_text


# def mismatch_count(actual, predicted, bias):
# 	mismatches = 0
# 	for act_amino, pred_amino in zip(actual, predicted):
# 		if act_amino != pred_amino:
# 			mismatches += 1
# 	return mismatches-bias

# # if load_previous:
# # 	model = load_model('best_model.hdf5')
# # else:
# # 	model = run_model(train_inputs, train_labels, max_seq_len, dict_size)
# # 	model.save("model_last.h5")
# # 	print("Saved model to disk")

# # starting_characters = 1
# # chars_to_predict = 49

# def L2_dist_over_training_seqs(predicted):
# 	predicted = tokenizer.texts_to_sequences([predicted])[0]
# 	predicted = pad_sequences([predicted], maxlen=max_seq_len-1, padding='pre')
# 	count = 0
# 	l2_dist = 0
# 	for sequence in train_data.splitlines():
# 		count += 1
# 		sequence = tokenizer.texts_to_sequences([sequence])[0]  # Map each word in sequence to its index in dictionary
# 		sequence = pad_sequences([sequence], maxlen=max_seq_len-1, padding='pre')  # Pre-pad the (tokenized) sequence
# 		l2_dist += np.linalg.norm(predicted-sequence, ord=2)
# 		if count == 1000:
# 			break
# 	return l2_dist/count

# random_seq_l2 = 0
# num_rand_seq = 200
# for j in range(num_rand_seq):
# 	random_seq = ""
# 	for i in range(50):
# 		random_seq += random.choice("ACDEFGHIKLMNPQRSTVWY")+" "
# 	random_seq = random_seq[:-1]
# 	random_seq_l2 += L2_dist_over_training_seqs(random_seq)
# random_seq_l2 = random_seq_l2/num_rand_seq
# print("Average L2 Distance of random sequences from training sequences: %.2f" % random_seq_l2)
# print()
# print()

# l2_dist_model = 0
# for k in range(num_rand_seq):
# 	# print("==============================================================================================================")
# 	predicted = predict_seq(random.choice("ACDEFGHIKLMNPQRSTVWY"), chars_to_predict, max_seq_len)
# 	# print(predicted)
# 	l2_dist_model += L2_dist_over_training_seqs(predicted)
# 	# print("==============================================================================================================")
# 	# print()

# 	# input()

# l2_dist_model = l2_dist_model/num_rand_seq
# print("Average L2 Distance of sequences generated by model from training sequences: %.2f" %l2_dist_model)

def makeNGram(data):
	sequences = [];
	for x in data:
		for i in range(len(x)-1):
			sequences.append(x[:i+2])

	return sequences;


def prepareData():
	(trainData, validationData) = loadData();
	trainData = [list(map(charToIndex, x)) for x in trainData];
	validationData = [list(map(charToIndex, x)) for x in validationData];

	trainData = makeNGram(trainData);
	validationData = makeNGram(validationData);

	trainData = np.array(pad_sequences(trainData, maxlen=lenSequence+1, padding='pre'));
	validationData = np.array(pad_sequences(validationData, maxlen=lenSequence+1, padding='pre'));

	trainInputs, trainLabels = trainData[:,:-1], trainData[:,-1];
	validationInputs, validationLabels = validationData[:,:-1], validationData[:,-1];

	trainLabels = ku.to_categorical(trainLabels, num_classes=numberOfAminoAcids()+1);
	trainInputs = ku.to_categorical(trainInputs, num_classes=numberOfAminoAcids()+1);
	validationLabels = ku.to_categorical(validationLabels, num_classes=numberOfAminoAcids()+1);
	validationInputs = ku.to_categorical(validationInputs, num_classes=numberOfAminoAcids()+1);

	return (trainLabels, trainInputs, validationLabels, validationInputs);


def train(trainInputs, trainLabels, validationInputs, validationLabels):
	model = Sequential();
	model.add(LSTM(units=150));
	model.add(Dense(numberOfAminoAcids()+1, activation='softmax'));

	# print(model.summary());

	optimizer = optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# filepath="model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
	lrate = LearningRateScheduler(step_decay)
	# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	history = model.fit(trainInputs, trainLabels, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(validationInputs, validationLabels))	

	print(type(history.history['loss']))

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')

	plt.show()

	plt.plot(np.exp(history.history['loss']))
	plt.plot(np.exp(history.history['val_loss']))
	plt.title('Perplexity')
	plt.ylabel('Perplexity')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')

	plt.show()

	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')

	plt.show()

	return model 

if __name__ == '__main__':

	data = prepareData();
	(trainLabels, trainInputs, validationLabels, validationInputs) = data;
	print(trainInputs.shape)
	print(trainLabels.shape);
	print(validationInputs.shape);
	print(validationLabels.shape);

	train(trainInputs, trainLabels, validationInputs, validationLabels);