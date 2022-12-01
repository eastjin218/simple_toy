import os, glob
import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("dataload.log")
logger.info('Process start!!')

data_path ='/home/simple_toy/dataset/005930_2020.csv'

df = pd.read_csv(data_path)

class WindowGenerator():
    def __init(self, input_width, label_width, shift, 
                train_df, val_df, test_df,
                label_columns=None):
        self.traind_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
