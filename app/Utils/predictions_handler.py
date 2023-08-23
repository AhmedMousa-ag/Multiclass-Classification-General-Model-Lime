import warnings
import numpy as np
import os
import config
import pandas as pd
from Utils.model_builder import load_model
from Utils.preprocess.preprocess import PreprocessData, prep_NUMERIC, prep_TEXT
import logging
import os
from abc import ABC, abstractmethod



SAVED_TEST_PRED_PATH = config.SAVED_TEST_PRED_PATH
seed = config.RAND_SEED

class __predictor_base_explain():
    """This is a base class for predictor to insure that the most important functions will be available,
    please modify next function as how it suits your algo, just be careful to the output"""

    def predict_explain(self,text: str):
        """This function takes text as string input.
        P.S: text processing happens here
        return: np array of prediction probabilities for each label"""
        if not isinstance(text,list):
            text = list(text)
        if len(text)<1:
            text = np.expand_dims(text,axis=0)

        preds = self.model.predict(text)
        if preds.shape[1] >1:
            return preds
        else:
            return np.array([[float(1-x),float(x)] for x in preds])


    def get_class_names(self):
        encoder = prep_NUMERIC.get_Label_Encoder()
        return list(encoder.classes_)


class Predictor(__predictor_base_explain):
    def __init__(self, data=None, model=None):

        if model is None:
            self.model = load_model()
        else:  # Model should be reloaded before getting the request,
               # that's the reason to pass the model to the predictor.
            self.model = model

        if not data is None:
            self.preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        self.text_vectorizer = None

    def predict_test(self, data=None):  # called for test prediction
        if not data is None:
            self.preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        id_col_name = self.preprocessor.get_id_col_name()
        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data)


        num_uniq_preds = [2 if np.array(preds).shape[1]==1 else np.array(preds).shape[1]][0]
        uniqe_preds_names = np.squeeze(self.preprocessor.invers_labels(sorted(range(num_uniq_preds))))
        results_pd = pd.DataFrame([])
        results_pd[id_col_name] = ids


        if num_uniq_preds > 2:
            for i in range(len(preds[0,:])): # Iterate over number of columns of model prediction
                col_name = self.preprocessor.invers_labels([i])[0]
                results_pd[col_name] = preds[:,i]
        else:
            #This means it's either 0 or 1
                pred = np.squeeze(preds)
                results_pd[uniqe_preds_names[0]] = 1-pred
                results_pd[uniqe_preds_names[1]] = pred

        # will convert get final prediction # uncomment if want to get final prediction column
        '''preds = self.conv_labels_no_probability(preds)
        preds = self.preprocessor.invers_labels(preds)
        results_pd["prediction"] = preds '''
        results_pd = results_pd.sort_values(by=[id_col_name])
        return results_pd



    def predict_get_results(self, data=None):
        if not data is None:
            self.preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        id_col_name = self.preprocessor.get_id_col_name()
        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()
        preds = self.model.predict(processed_data)
        preds = self.conv_labels_no_probability(preds)
        preds = self.preprocessor.invers_labels(preds)
        results_pd = pd.DataFrame([])
        results_pd[id_col_name] = ids
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=[id_col_name])
        return results_pd

    def conv_labels_no_probability(self, preds):
        #preds = np.array(np.squeeze(preds))
        if preds.shape[1] < 2:
            preds = np.array(np.squeeze(preds))
            if preds.size < 2:  # If passed one prediction it cause and error if not expanded dimension
                prediction = np.array(np.expand_dims(
                    np.round(preds), axis=0), dtype=int)
            else:
                prediction = np.array(np.round(preds), dtype=int)

            return prediction
        else:
            if preds.size < 2:  # If passed one prediction it cause an error if not expanded dimenstion
                prediction = np.array(np.expand_dims(
                    np.argmax(preds,axis=1), axis=0), dtype=int)

            else:

                prediction = np.array(np.argmax(preds,axis=1), dtype=int)
            return prediction

    def save_predictions(self, save_path=SAVED_TEST_PRED_PATH):
        path = os.path.join(save_path, "test_predictions.csv")
        test_result = self.predict_test()
        test_result.to_csv(path,index=False)
        print(f"saved results to: {path}")
