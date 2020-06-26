import pandas as pd
import os
import re
import numpy as np
from tqdm import tqdm
from cache_em_all import Cachable

# Data preparation code from https://c4m-uoft.github.io/projects/physionet/physionet2019_handout.pdf

DATA_DIR = "data/training_setA/"  # Path to the data

# Names of all columns in the data that contain physiological data
physiological_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets']

# Names of all columns in the data that contain demographic data
demographic_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

# The combination of physiological and demographic data is what we will use as features in our model
feature_cols = physiological_cols + demographic_cols

# The name of the column that contains the value we are trying to predic
label_col = "SepsisLabel"

# Pre-calculated means and standard deviation of all physiological and demographic columns. We will use this to normalize
# data using their z-score. This isn't as important for simpler models such as random forests and decision trees,
# but can result in significant improvements when using neural networks
physiological_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777])
physiological_std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])

demographic_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
demographic_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

def load_single_file(file_path):
    '''
    Create pandas df from each file, and add hour and patient column.
    '''
    df = pd.read_csv(file_path, sep='|')
    df['hours'] = df.index
    df['patient'] = re.search('p(.*?).psv', file_path).group(1)
    return df

def clean_data(data):
    '''
    Normalize data and fill in missing values
    with 0.
    '''
    data.reset_index(inplace=True, drop=True)

    # Normalizes physiological and demographic data using z-score.
    data[physiological_cols] = (data[physiological_cols] - physiological_mean) / physiological_std
    data[demographic_cols] = (data[demographic_cols] - demographic_mean) / demographic_std

    # Maps invalid numbers (NaN, inf, -inf) to numbers (0, really large number, really small number)
    data[feature_cols] = np.nan_to_num(data[feature_cols])

    return data

    
def get_data_files():
    '''
    Helper function to read in data from the given directory.
    '''
    return [os.path.join(DATA_DIR, x) for x in sorted(os.listdir(DATA_DIR)) if int(x[1:-4]) % 5 > 0]

@Cachable('training_setA.csv')
def load_data():
    '''
    Combine each PSV into one contiuous pandas df, and cache as training_setA.csv.
    '''
    data = get_data_files()
    data_frames = [clean_data(load_single_file(d)) for d in tqdm(data)]
    merged = pd.concat(data_frames)
    return merged

if __name__ == "__main__":
    load_data()
