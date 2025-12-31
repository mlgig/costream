import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

@pytest.fixture
def dummy_data_dir(tmp_path):
    """Creates a temporary directory with synthetic subject CSVs."""
    d = tmp_path / "data"
    d.mkdir()
    
    # Subject 1: 2 files
    df1 = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100), 'label': np.zeros(100)})
    df1.to_csv(d / "subj1_trial1.csv", index=False)
    
    df2 = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100), 'label': np.zeros(100)})
    df2.to_csv(d / "subj1_trial2.csv", index=False)
    
    # Subject 2: 1 file
    df3 = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100), 'label': np.zeros(100)})
    df3.to_csv(d / "subj2_trial1.csv", index=False)
    
    return d

@pytest.fixture
def synthetic_dfs():
    """
    Returns a list of DataFrames mimicking a real dataset.
    - DF 0: ADL only (all zeros)
    - DF 1: One Event (at index 500)
    - DF 2: Two Events (at index 200 and 800)
    """
    freq = 100
    length = 10 * freq # 10 seconds
    
    # DF 0: Pure ADL
    df0 = pd.DataFrame(np.random.normal(0, 0.1, (length, 1)), columns=['mag'])
    df0['label'] = 0
    
    # DF 1: Single Event
    df1 = df0.copy()
    df1.loc[500:550, 'label'] = 1 # Event block
    
    # DF 2: Multi Event
    df2 = df0.copy()
    df2.loc[200:210, 'label'] = 1
    df2.loc[800:850, 'label'] = 1
    
    return [df0, df1, df2]