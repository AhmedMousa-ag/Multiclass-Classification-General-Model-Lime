"""This module for model explainable using lime"""

from lime.lime_text import LimeTextExplainer
import numpy as np
import config
from Utils.preprocess.schema_handler import produce_schema_param
import os
import glob

# TODO document module file


class explainer:
    def __init__(self, model_predictor):
        """Use this class for explaining predictions
        Args:
        model_predictor: is the model predictor class, the class must have to function,
            1- get_class_names()--> return list of class names, and
            2- predict_explain()--> takes string data to process, make prediction and return probabilities for each class
        """
        self.model_predictor = model_predictor
        self.class_names = self.model_predictor.get_class_names()
        print("class names are: ", self.class_names)
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        self.MAX_LOCAL_EXPLANATIONS = 3

    def explain_texts(self, text: str, top_labels=None):
        """Make lime computations and produce explain object that has results will be accessed later"""
        num_feature = 20  # Number of tokens to explain
        self.exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.model_predictor.predict_explain,
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
            label_probs[label] = np.round(predic_proba[indx], 5)
        print("label_probs", label_probs)
        return label_probs

    def get_explanations(self):
        explanations = {}
        explanations["intercept"] = np.round(self.get_label_intercept(), 5)
        explanations["token_scores"] = self.get_word_pos_score()
        return explanations

    def get_word_pos_score(self):
        """Returns a dictionary containing each word with their position and score"""
        words_list = self.exp.as_list(self.indx_pred)
        words_map = self.exp.as_map()[self.indx_pred]
        words_with_score = {}
        for i in range(len(words_list)):
            word_pos = words_map[i][0]
            word_name = str(words_list[i][0])
            word_score = np.round(words_map[i][1], 5)
            words_with_score[word_name] = {"position": word_pos, "score": word_score}

        return words_with_score

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
        id_col, text_col, targ_col = get_id_text_targ_col()
        ids = data[id_col]
        texts = data[text_col]
        pred_list = []
        for id, txt in zip(ids, texts):
            result = {}
            print(f"raw text: {txt}")
            self.explain_texts(text=txt)
            result[id_col] = id
            result["label"] = self.get_prediction()
            result["probabilities"] = self.get_label_probabilities()
            result["explanations"] = self.get_explanations()
            pred_list.append(result)

        output["predictions"] = pred_list
        return output


def read_data_config_schema():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    path = glob.glob(
        os.path.join(os.pardir, "ml_vol", "inputs", "data_config", "*.json")
    )[0]
    try:
        schema = produce_schema_param(config.DATA_SCHEMA)
        return schema
    except:
        raise Exception(f"Error reading json file at: {path}")


def get_id_text_targ_col():
    """The only reason we are producing schema here and not using Utils or preprocessor is that
    we would like to generalize this exp_lime to almost all text classification algo at Ready Tensor."""
    schema = read_data_config_schema()
    id_col = schema["id"]
    text_col = schema #TODO
    targ_col = schema["target_col"]
    return id_col, text_col, targ_col
