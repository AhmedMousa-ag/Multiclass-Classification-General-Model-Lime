# Multiclass-Classification-General-Model-Lime

This model uses Autogluon which do automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.
This code repo uses Lime for model explanations as well (XAI).

## Navigatie Code

### Dockerfile

Has the environment requirements, don't change it unless needed to.

## Navigatie Code

### app:

Is where all of code used to run train, test or serve modules.

config.py: is a file to determine main configurations such as files paths used during preprocessing, training, serving.

inference_app.py: is a file that declare infer/ping and predictions to requested data happens.

#### Utils:

Utils consist of:

1- model_explain: Where text explainability using Lime happens

2- preprocess folder: has the preprocess classes to process data according to schema.

3- model_builder.py: where the Machine Learning model defined, built, and loaded.

4- predictions_handler.py: called when needed a prediction for inference or testing/predic.

5- utils.py: general functions to help such as load json files.

#### train

python file, called to start training on the required dataset and saves trained model to be called later during inference or during testing.

#### predict

python file, called to generate test.csv file to test model preformance after training.

#### serve

python file, called to generate inferences in production on your server, listens to port 8080.
