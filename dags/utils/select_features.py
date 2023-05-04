import pandas as pd

from utils.files_util import load_files, save_files

def select_features():
    x_val, sel_feat_df = load_files(['x_val', 'selected_features'])
    sel_feat = sel_feat_df['selected_features'].to_list()
    selected_x_val=x_val[sel_feat]
    selected_x_val.name='selected_x_val'
    save_files([selected_x_val])

