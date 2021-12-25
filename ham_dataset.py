''' ------------------- HAM DATALOADER ------------------------ '''
import os
import pandas as pd
import numpy as np
from glob import glob

# %matplotlib inline
# python libraties
import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchsummary import summary
# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
# from mighty_losses import CurricularFace
# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# data_dir = "/home/sazzad/Documents/ISBI/HAM10000/"
data_dir = "./data/HAM10000/"
all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))

df = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata'))

df['path'] = ''

print('lession unique -', len(df['lesion_id'].unique()))
print('image unique -', len(df['image_id'].unique()))

df = df.drop_duplicates('lesion_id')

# all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

df['path'] = df['image_id'].map(imageid_path_dict.get)
df['cell_type'] = df['dx'].map(lesion_type_dict.get)
df['cell_type_idx'] = pd.Categorical(df['dx']).codes
df.tail()

# print(df['cell_type_idx'].value_counts())

# split dataset into train, test and validation
df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=101, stratify=df['cell_type_idx'])
df_train, df_val = train_test_split(df_trainval, test_size=0.1, random_state=101, stratify=df_trainval['cell_type_idx'])

# print stats of train, test and validation set

# keyerror index missing
df_train = df_train.reset_index()
df_val = df_val.reset_index()
df_test = df_test.reset_index()


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

input_size = 32
norm_mean = (0.49139968, 0.48215827, 0.44653124)
norm_std = (0.24703233, 0.24348505, 0.26158768)
# define the transformation of the train images.
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])


trainset = HAM10000(df_train, transform=train_transform)
valset = HAM10000(df_val, transform=val_transform)
testset = HAM10000(df_test, transform=val_transform)

def get_ham_dataset():
    return trainset, valset, testset


cls_num_list = []
for i in range(7):
    cls_num_list.append(0)

def get_cls_num_list():
    for i in range((len(trainset))):
        target = trainset[i][1]
        cls_num_list[target]+=1
    print(cls_num_list)
    return cls_num_list
    
# print(cls_num_list)
    