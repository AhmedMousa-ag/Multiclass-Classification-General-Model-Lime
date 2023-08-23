#from Utils.preprocess.preprocess import prep_TEXT
import config
import os
from autogluon.tabular import  TabularPredictor, TabularDataset


MODEL_SAVE_PATH = config.MODEL_SAVE_PATH



def load_model(save_path=MODEL_SAVE_PATH):
    model = TabularPredictor.load(save_path)
    print(f"Loaded model from: {save_path} successfully")
    return model
