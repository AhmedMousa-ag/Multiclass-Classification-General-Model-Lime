from autogluon.common import space
import config
NUM_EPOCHS_LOWER = config.NUM_EPOCHS_LOWER
NUM_EPOCHS_UPPER = config.NUM_EPOCHS_UPPER
LEARNING_RATE_LOWER = config.LEARNING_RATE_LOWER
LEARNING_RATE_UPPER = config.LEARNING_RATE_UPPER
ACTIVATIONS =  config.ACTIVATIONS
DROPOUT_PROB_LOWER = config.DROPOUT_PROB_LOWER
DROPOUT_PROB_UPPER = config.DROPOUT_PROB_UPPER


# gbm_options
NUM_BOOST_ROUND_LOWER = config.NUM_BOOST_ROUND_LOWER
NUM_BOOST_ROUND_UPPER = config.NUM_BOOST_ROUND_UPPER
NUM_LEAVES_LOWER = config.NUM_LEAVES_LOWER
NUM_LEAVES_UPPER = config.NUM_LEAVES_UPPER
nn_options = {  # specifies non-default hyperparameter values for neural network models
    'num_epochs': space.Int(lower=NUM_EPOCHS_LOWER,upper=NUM_EPOCHS_UPPER,default=50),  # number of training epochs (controls training time of NN models)
    'learning_rate': space.Real(LEARNING_RATE_LOWER, LEARNING_RATE_UPPER, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'activation': space.Categorical(ACTIVATIONS),  # activation function used in NN (categorical hyperparameter, default = first entry)
    'dropout_prob': space.Real(DROPOUT_PROB_LOWER, DROPOUT_PROB_UPPER, default=0.1),  # dropout probability (real-valued hyperparameter)
}


gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    'num_boost_round': space.Int(lower=NUM_BOOST_ROUND_LOWER, upper=NUM_BOOST_ROUND_UPPER, default=100),  # number of boosting rounds (controls training time of GBM models)
    'num_leaves': space.Int(lower=NUM_LEAVES_LOWER, upper=NUM_LEAVES_UPPER, default=36),  # number of leaves in trees (integer hyperparameter)
}


HYPERPARAMETERS = {  # hyperparameters of each model type
                   'GBM': gbm_options,
                   'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
                  }  # When these keys are missing from hyperparameters dict, no models of that type are trained


num_trials = 20  # try at most 20 different hyperparameter configurations for each type of model
search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

TIMELIMIT = 2*60

HYPERPARAMETER_TUNE_KWARGS = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler' : 'local',
    'searcher': search_strategy,
}  # Refer to TabularPredictor.fit docstring for all valid values



