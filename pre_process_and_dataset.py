import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import pandas as pd
from tqdm import tqdm


def data_pre_process(txt_input, txt_c5, txt_c20, txt_c80, txt_c320, txt_c1300):
    # create an empty new csv file with 2 columns (sequence, label)
    dataset = pd.DataFrame(columns=['sequence', 'label'])
    # read the input txt file and append the first column to the new csv file where the label is 0
    with open(txt_input, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 0} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    print("done reading input file")
    # read the c1300 txt file and append the first column to the new csv file where the label is 1
    with open(txt_c1300, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 1} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    # read the c320 txt file and append the first column to the new csv file where the label is 2
    with open(txt_c320, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 2} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    # read the c80 txt file and append the first column to the new csv file where the label is 3
    with open(txt_c80, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 3} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    # read the c20 txt file and append the first column to the new csv file where the label is 4
    with open(txt_c20, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 4} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    # read the c5 txt file and append the first column to the new csv file where the label is 5
    with open(txt_c5, 'r') as f:
        lines = f.readlines()
        data = [{'sequence': line.strip().split('\t')[0], 'label': 5} for line in lines]
        dataset = pd.concat([dataset, pd.DataFrame(data)])
    # save dataset to csv file
    dataset.to_csv('dataset.csv', index=False)


class DNA_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Read sequence string
        sequence_string = self.data.iloc[idx, 0]

        # convert into One-hot encoding matrix (4, 41)
        sequence_length = len(sequence_string)
        sequence = torch.zeros(4, sequence_length)

        for i, char in enumerate(sequence_string):
            if i >= 41:
                break

            if char == 'A':
                sequence[0, i] = 1
            elif char == 'C':
                sequence[1, i] = 1
            elif char == 'G':
                sequence[2, i] = 1
            elif char == 'T':
                sequence[3, i] = 1

        # Padding sequences with uniform distribution
        max_sequence_length = 41
        if sequence_length < max_sequence_length:
            padding_length = max_sequence_length - sequence_length
            padding = torch.empty(4, padding_length).uniform_()
            sequence = torch.cat([sequence, padding], dim=1)

        # Read label
        label = self.data.iloc[idx, 1]
        sample = {'sequence': sequence, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        # Convert each label to one-hot encoding
        label = torch.zeros(6)
        label[sample['label']] = 1
        return sequence, label
