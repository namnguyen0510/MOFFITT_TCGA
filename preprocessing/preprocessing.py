import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from utils import *


dircs = ['raw_data' + '/' + x for x in sorted(os.listdir('raw_data'))]
out_dirc = 'processed_data'

for dir in sorted(os.listdir('raw_data')):
    try:
        os.mkdir(os.path.join(out_dirc, dir))
    except:
        pass

for dir in dircs:
    label = dir.split('/')[1]
    print(label)
    _out = os.path.join(out_dirc,label)
    data = FeatureExtractor(dir)
    data._save(data.get_raw_data,_out)
    print(data._Summary(data.get_raw_data))
