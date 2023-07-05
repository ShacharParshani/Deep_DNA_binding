import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from pre_process_and_dataset import *
from cnn_Model import *
from train_and_test import *


# read the txt files
txt_input = 'RBNS_training/RBP1_input.seq'
txt_c5 = 'RBNS_training/RBP1_5nM.seq'
txt_c20 = 'RBNS_training/RBP1_20nM.seq'
txt_c80 = 'RBNS_training/RBP1_80nM.seq'
txt_c320 = 'RBNS_training/RBP1_320nM.seq'
txt_c1300 = 'RBNS_training/RBP1_1300nM.seq'

# TODO remove comment
# create the dataset
# data_pre_process(txt_input, txt_c5, txt_c20, txt_c80, txt_c320, txt_c1300)

# TODO save the number of protein in file name
# create a dataset instance
dna_dataset = DNA_Dataset("dataset.csv")


# TODO transform?

batch_size = 64
# split the dataset into train and test
train_dataset, test_dataset = torch.utils.data.random_split(dna_dataset, [int(len(dna_dataset) * 0.8),
                                                                          len(dna_dataset) - int(
                                                                                len(dna_dataset) * 0.8)])


# create a dataloader instance
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dataiter = iter(train_loader)
batch_images, batch_labels = next(dataiter)

# create a model instance
model = Our_Model(batch_size)

# choose a loss function
criterion = nn.CrossEntropyLoss()
# choose an optimizer and learning rate for the training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0003)

# choose the number of epochs
number_of_epochs = 5

# Train the model
model = train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size)
# Evaluate the trained model
test(model, test_loader, criterion, batch_size)
