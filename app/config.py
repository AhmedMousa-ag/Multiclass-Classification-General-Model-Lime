import os
from Utils.utlis import read_json_file
import glob
from dotenv import load_dotenv

load_dotenv()


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


prefix = os.path.join(os.pardir,"model_inputs_outputs")
#prefix = os.path.join("model_inputs_outputs")

#app_prefix = os.path.join("app") #os.pardir,


RAND_SEED = 42

OUTPUTS_PATH = os.path.join(prefix, "outputs")

check_dir(os.path.join(OUTPUTS_PATH, "inputs", "schema"))
#print("dirctory: ",os.path.join(prefix, "inputs", "schema"))
DATA_SCHEMA_PATH = glob.glob(os.path.join(prefix, "inputs", "schema", "*.json"))[
    0
]  # Gets the first file of json type


DATA_SCHEMA = read_json_file(DATA_SCHEMA_PATH)



FAILURE_PATH =  os.path.join(OUTPUTS_PATH, "errors")
#FAILURE_PATH = os.path.join(ERRORS_PATH,"train_error.txt")
check_dir(FAILURE_PATH)


#HYPER_PARAM_PATH = glob.glob(os.path.join(prefix, "model", "model_config", "*.json"))[0]
#check_dir(os.path.join(prefix, "model", "model_config"))


DATA_PATH = os.path.join(prefix, "inputs", "data")
check_dir(DATA_PATH)
check_dir(os.path.join(DATA_PATH, "training"))
check_dir(os.path.join(DATA_PATH, "testing"))


TRAIN_DATA_PATH = glob.glob(
    os.path.join(DATA_PATH, "training", "*.csv")
)[0]


TEST_DATA_PATH = glob.glob(
    os.path.join(DATA_PATH, "testing", "*.csv")
)[0]


MODEL_NAME = "tf_trained_glove_embed_bidirectional_model" #TODO

MODEL_SAVE_PATH = os.path.join(prefix, "model", "artifacts")
check_dir(MODEL_SAVE_PATH)


# os.path.join("Utils","preprocess","artifacts")
PREPROCESS_ARTIFACT_PATH = MODEL_SAVE_PATH
check_dir(PREPROCESS_ARTIFACT_PATH)


SAVED_TEST_PRED_PATH = os.path.join(OUTPUTS_PATH, "predictions")
check_dir(SAVED_TEST_PRED_PATH)


HYPER_PARAM_PATH = os.path.join(OUTPUTS_PATH, "hpt_outputs")
check_dir(HYPER_PARAM_PATH)

#Autogluon Time Limit
TIMELIMIT= 30 #mins

#Hyperparameter Configurations
NUM_EPOCHS_LOWER = os.getenv("NUM_EPOCHS_LOWER")
NUM_EPOCHS_UPPER = os.getenv("NUM_EPOCHS_UPPER")
LEARNING_RATE_LOWER = os.getenv("LEARNING_RATE_LOWER")
LEARNING_RATE_UPPER = os.getenv("LEARNING_RATE_UPPER")
ACTIVATIONS =  os.getenv("ACTIVATIONS")
DROPOUT_PROB_LOWER = os.getenv("DROPOUT_PROB_LOWER")
DROPOUT_PROB_UPPER = os.getenv("DROPOUT_PROB_UPPER")


# gbm_options
NUM_BOOST_ROUND_LOWER = os.getenv("NUM_BOOST_ROUND_LOWER")
NUM_BOOST_ROUND_UPPER = os.getenv("NUM_BOOST_ROUND_UPPER")
NUM_LEAVES_LOWER=os.getenv("NUM_LEAVES_LOWER")
NUM_LEAVES_UPPER=os.getenv("NUM_LEAVES_UPPER")

