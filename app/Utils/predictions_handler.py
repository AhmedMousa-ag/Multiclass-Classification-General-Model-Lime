import numpy as np
import os
import config
import pandas as pd
from Utils.model_builder import load_model
from Utils.preprocess.preprocess import PreprocessData, prep_NUMERIC
import os
from Utils.preprocess.schema_handler import produce_schema_param


SAVED_TEST_PRED_PATH = config.SAVED_TEST_PRED_PATH
seed = config.RAND_SEED

class __predictor_base_explain():
    """This is a base class for predictor to insure that the most important functions will be available,
    please modify next function as how it suits your algo, just be careful to the output"""

    def predict_explain(self,data):
        """This function takes text as string input.
        P.S: text processing happens here
        return: np array of prediction probabilities for each label"""
        if not isinstance(data,pd.DataFrame):
            data = pd.DataFrame(data,columns=self.get_columns_names())
        preds = self.model.predict_proba(data).to_numpy()
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

        self.schema = produce_schema_param(config.DATA_SCHEMA)

    def get_columns_names(self):
        col_names = []
        for feature in self.schema["features"]:
            col_names.append(feature["name"])
        return col_names

    def predict_test(self, data=None):  # called for test prediction
        if not data is None:
            self.preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        id_col_name = self.preprocessor.get_id_col_name()
        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()

        processed_data = self.preprocessor.get_data()

        preds = self.model.predict(processed_data).to_numpy()
        pred_probab = self.model.predict_proba(processed_data).to_numpy()
        print("prediction probability: ",pred_probab)
        num_uniq_preds = len(self.schema["target_classes"])
        uniqe_preds_names = self.preprocessor.invers_labels(preds)
        results_pd = pd.DataFrame([])
        results_pd[id_col_name] = ids


        if num_uniq_preds > 2:
            for i in range(len(pred_probab[0,:])): # Iterate over number of columns of model prediction
                col_name = self.preprocessor.invers_labels([i])[0]
                results_pd[col_name] = pred_probab[:,i]
        else:
            #This means it's either 0 or 1
                pred_probab = np.squeeze(pred_probab)
                results_pd[uniqe_preds_names[0]] = 1-pred_probab
                results_pd[uniqe_preds_names[1]] = pred_probab

        # will convert get final prediction # uncomment if want to get final prediction column
        '''preds = self.conv_labels_no_probability(preds)
        preds = self.preprocessor.invers_labels(preds)
        results_pd["prediction"] = preds '''
        results_pd = results_pd.sort_values(by=[id_col_name])
        return results_pd



    def predict_get_results(self, data=None):
        print("passed data: ",data)
        if not data is None:
            self.preprocessor = PreprocessData(
                data, train=False, shuffle_data=False)

        id_col_name = self.preprocessor.get_id_col_name()
        ids = self.preprocessor.get_ids()
        self.preprocessor.drop_ids()
        print("after cleaning data: ",data)
        processed_data = self.preprocessor.get_data()
        print("processed_data: ",processed_data)
        preds = self.model.predict(processed_data)
        preds = self.preprocessor.invers_labels(preds)
        results_pd = pd.DataFrame([])
        results_pd[id_col_name] = ids
        results_pd["prediction"] = preds
        results_pd = results_pd.sort_values(by=[id_col_name])
        return results_pd

    def conv_labels_no_probability(self, preds):
        #preds = np.array(np.squeeze(preds))
        if len(self.schema["target_classes"]) < 2:
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
