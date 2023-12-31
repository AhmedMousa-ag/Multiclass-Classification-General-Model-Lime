# major part of code sourced from aws sagemaker example:
# https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/decision_trees/predictor.py

import io
import pandas as pd
import flask
from flask import request, jsonify
import traceback
import sys
import os
import warnings
from Utils.predictions_handler import Predictor
from Utils.model_builder import load_model
import config
from Utils.model_explain.exp_lime import Explainer
import json


MODEL_NAME = config.MODEL_NAME
failure_path = config.FAILURE_PATH


model = load_model()

predictor =  Predictor(model=model)

explainer =  Explainer(predictor)
# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. """
    status = 200
    response = f"Hello - I am {MODEL_NAME} model and I am at your service!"
    print(response)
    return flask.Response(response=response, status=status, mimetype="application/json")


@app.route("/infer", methods=["POST"])
def infer():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        raw_data = []
        for row in s["instances"]:
            raw_data.append(row)
        data = pd.DataFrame(raw_data)
    elif flask.request.content_type == 'application/json': # checks for json data
        data = flask.request.get_json()
        raw_data = []
        for row in data["instances"]:
            raw_data.append(row)
        data = pd.DataFrame(raw_data)
    else:
        return flask.Response(
            response="This predictor only supports CSV, and json data",
            status=415, mimetype="text/plain"
        )

    # Do the prediction
    try:
        predector = Predictor(model=model)

        predictions = predector.predict_get_results(data=data)
        # Convert from dataframe to CSV
        result = json.dumps(predictions)

        return flask.Response(response=result, status=200, mimetype="text/csv")

    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        path = os.path.join(failure_path,"serve_error.txt")
        with open(path, 'w') as s:
            s.write('Exception during inference: ' + str(err) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during inference: ' +
              str(err) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.

        return flask.Response(
            response="Error generating predictions. Check failure file.",
            status=400, mimetype="text/plain"
        )


@app.route("/explain", methods=["POST"])
def explain_xai():
    """Get local explanations on a few samples. In this  server, we take data as CSV, convert
    it to a pandas data frame for internal use.
    Explanations come back using the ids passed in the input data.
    """
    data = None
    print("started explaining")
    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        raw_data = []
        for row in s["instances"]:
            raw_data.append(row)
        data = pd.DataFrame(raw_data)
    elif flask.request.content_type == 'application/json': # checks for json data
        data = flask.request.get_json()
        raw_data = []
        for row in data["instances"]:
            raw_data.append(row)
        data = pd.DataFrame(raw_data)
    else:
        return flask.Response(
            response="This predictor only supports CSV, and json data",
            status=415, mimetype="text/plain"
        )

    # Do the prediction
    try:

        result =  explainer.produce_explainations(data)
        result = json.dumps(result)
        return flask.Response(response=result, status=200, mimetype='application/json')
    except Exception as err:
        # Write out an error file. This will be returned as the failureReason to the client.
        trc = traceback.format_exc()
        path = os.path.join(failure_path,"serve_error.txt")
        with open(path, 'w') as s:
            s.write('Exception during explanation: ' + str(err) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during explanation: ' +
              str(err) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.

        return flask.Response(
            response="Error generating explanation. Check failure file.",
            status=400, mimetype="text/plain"
        )
