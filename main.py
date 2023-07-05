import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
import pandas as pd
from dataset import *


# read the txt files
txt_input = 'input.txt'
txt_c5 = 'c5.txt'
txt_c20 = 'c20.txt'
txt_c80 = 'c80.txt'
txt_c320 = 'c320.txt'
txt_c1300 = 'c1300.txt'


dataset = data_pre_process(txt_input, txt_c5, txt_c20, txt_c80, txt_c320, txt_c1300)

# create the dataset
dna_dataset = dna_dataset(dataset)

#