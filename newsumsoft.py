# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 01:54:26 2023

@author: DELL
"""
import os
import pandas as pd
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


folder_path = './stanlem/stablelm-base-alpha-7b'

new_folder_path = './stanlem/stablelm-base-alpha-7b-softmax'


os.makedirs(new_folder_path, exist_ok=True)


for filename in os.listdir(folder_path):

    if filename.endswith('.xlsx') or filename.endswith('.xls'):

        file_path = os.path.join(folder_path, filename)
 
        df = pd.read_excel(file_path)

        sum_values = df.sum()
        df_sum = pd.DataFrame(softmax(sum_values)).transpose()
        df_sum.index = ['Total']

        new_filename = os.path.splitext(filename)[0] + '_sum_softmax.xlsx'
        new_file_path = os.path.join(new_folder_path, new_filename)
 
        df_sum.to_excel(new_file_path, index=False)