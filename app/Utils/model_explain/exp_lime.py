"""This module for model explainable using lime"""

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import config
from Utils.preprocess.schema_handler import produce_schema_param
from Utils.preprocess.preprocess import PreprocessData
import os
import glob
import datetime
import pandas as pd


class Explainer:
    def __init__(self, model_predictor):
        """Use this class for explaining predictions
        Args:
        model_predictor: is the model predictor class, the class must have to function,
            1- get_class_names()--> return list of class names, and
            2- predict_explain()--> takes string data to process, make prediction and return probabilities for each class
        """
        self.model_predictor = model_predictor
        self.class_names = self.model_predictor.get_class_names()
        train_data = pd.read_csv(config.TRAIN_DATA_PATH)
        self.preprocessor= PreprocessData(
                train_data, train=False, shuffle_data=False)
        self.preprocessor.drop_ids()
        train_data = self.preprocessor.get_data().to_numpy()
        self.explainer = LimeTabularExplainer(class_names=self.class_names,training_data=train_data)
        self.MAX_LOCAL_EXPLANATIONS = 3

    def explain_data(self, data, top_labels=None):
        """Make lime computations and produce explain object that has results will be accessed later"""
        num_feature = 20  # Number of tokens to explain
        self.exp = self.explainer.explain_instance(
            data_row=data,
            predict_fn=self.model_predictor.predict_explain,
            labels=range(len(self.class_names)),
            num_features=num_feature,
            top_labels=top_labels,
        )
        return self.exp

    def get_label_intercept(self):
        self.indx_pred = np.argmax(self.exp.predict_proba)
        return self.exp.intercept[self.indx_pred]

    def get_prediction(self):
        """Returns final prediction class"""
        self.indx_pred = np.argmax(self.exp.predict_proba)
        prediction = self.class_names[self.indx_pred]
        print("prediction", prediction)
        return prediction

    def get_label_probabilities(self):
        """Returns each label with their predicted probability"""
        label_probs = {}
        predic_proba = self.exp.predict_proba
        for indx, label in enumerate(self.class_names):
            label_probs[label] = str(np.round(predic_proba[indx], 5))
        print("label_probs", label_probs)
        return label_probs

    def get_explanations(self):
        explanations = {}
       # explanations["intercept"] = np.round(self.get_label_intercept(), 5)
        explanations["featureScores"] = self.get_feature_score()
        print("explaination: ",explanations)
        return explanations

    def get_feature_score(self):
        """Returns a dictionary containing each word with their position and score"""
        #features_list = self.exp.as_list(self.indx_pred)
       # print("idx pred: ",self.indx_pred)
       # print("words_list: ",self.exp.as_list())
        features_map = self.exp.as_map()
        features_names = self.model_predictor.get_columns_names()
        features_with_score = {}
        for label in features_map.keys():
            for feautre in features_map[label]:
                col_idx = feautre[0]
                feature_name = features_names[col_idx]
                feature_score = np.round(feautre[1], 5)
                features_with_score[feature_name] =  str(feature_score)

        return features_with_score

    def produce_explainations(self, data):
        """Takes data to explain and return a dictionary with predictions, labels and words with their position and score"""
        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time.
            Given {data.shape[0]} samples.
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        data = data.head(self.MAX_LOCAL_EXPLANATIONS)

        output = {}
        output["status"] = "success"
        output["message"]=""
        output["timestamp"] =datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        #id_col_name = preprocessor.get_id_col_name()
        ids = preprocessor.get_ids()
        preprocessor.drop_ids()
        data = self.preprocessor.get_data()

        pred_list = []
        data = data.to_numpy()
        for id, row in zip(ids, data):
            result = {}
            self.explain_data(data=row)
            result["sampleId"] = id
            result["predictedClass"] = self.get_prediction()
            result["predictedProbabilities"] = self.get_label_probabilities()
            result["explanations"] = self.get_explanations()
            pred_list.append(result)
        output["predictions"] = pred_list
        output["explanationMethod"]="Lime"
        print("output: ",output)
        return output


def read_data_config_schema():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    try:
        schema = produce_schema_param(config.DATA_SCHEMA)
        return schema
    except:
        raise Exception(f"Error reading json file at: {config.DATA_SCHEMA}")


def get_id_text_targ_col():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    schema = read_data_config_schema()
    id_col = schema["id"]
    #text_col = schema #TODO
    targ_col = schema["target_col"]
    return id_col, targ_col
