import pandas as pd
from pathlib import PosixPath

def get_label(_path: PosixPath):

    if _path.match('*/train_curated/*'):
        labelfn = 'train_curated.csv'
    elif _path.match('*/train_noisy/*'):
        labelfn = 'train_noisy.csv'
    
    fname = _path.name

    labeldf = pd.read_csv(_path.parent.parent/labelfn)

    return labeldf.loc[labeldf['fname']==fname, 'labels'].values[0]

def get_label_from_index(index: int, type='curated'):

    if type == 'curated':
        labelfn = 'train_curated.csv'
    elif type == 'train_noisy':
        labelfn = 'train_noisy.csv'

    labeldf = pd.read_csv('data/'+labelfn)

    return labeldf.iloc[index, 'labels'].values[0]