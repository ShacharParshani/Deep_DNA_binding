import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset


def data_pre_process(txt_input, txt_c5, txt_c20, txt_c80, txt_c320, txt_c1300):
    # create a empty new csv file with 2 columns (sequence, label)
    dataset = pd.DataFrame(columns=['sequence', 'label'])
    # read the input txt file and append the first column to the new csv file where the label is 0
    with open(txt_input, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 0}, ignore_index=True)
    # read the c1300 txt file and append the first column to the new csv file where the label is 1
    with open(txt_c1300, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 1}, ignore_index=True)
    # read the c320 txt file and append the first column to the new csv file where the label is 2
    with open(txt_c320, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 2}, ignore_index=True)
    # read the c80 txt file and append the first column to the new csv file where the label is 3
    with open(txt_c80, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 3}, ignore_index=True)
    # read the c20 txt file and append the first column to the new csv file where the label is 4
    with open(txt_c20, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 4}, ignore_index=True)
    # read the c5 txt file and append the first column to the new csv file where the label is 5
    with open(txt_c5, 'r') as f:
        for line in f:
            dataset = dataset.append({'sequence': line.strip().split('\t')[0], 'label': 5}, ignore_index=True)
    return dataset





class dna_dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        sample = {'sequence': sequence, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        # convert each label to one-hot encoding
        label = torch.zeros(6)
        label[sample['label']] = 1
        return sample, label

