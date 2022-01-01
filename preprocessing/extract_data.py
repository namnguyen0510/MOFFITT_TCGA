import torch
import pandas as pd
import numpy as np
import shutil
import tqdm
from dataloader import *
from utils import *
from model import *

# CODE for dimension reduction
dirc = 'omic_data'
data = ReduceDim(dirc)
data._select_k_best()
