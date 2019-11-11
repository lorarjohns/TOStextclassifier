'''

Multi-class classifier for the Keras Reuters dataset
using GridSearchCV via the sklearn wrapper

'''

import numpy as np
import keras

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K, metrics
from sklearn.model_selection import GridSearchCV

from keras.preprocessing.text import Tokenizer

vocab_size = 3000
batch_size = 32
epochs = 5

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=vocab_size, test_split=0.2)
print(f"x_train shape: {x_train.shape}\nx_test shape: {x_test.shape}")

classes = np.max(y_train) + 1

print("Vectorizing data . . . ")
tokenizer = Tokenizer(num_words=vocab_size)
x_train = tokenizer.sequences_to_matrix(x_train, mode="binary")
x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")

y_train = keras.utils.to_categorical(y_train, num_classes=classes)
y_test = keras.utils.to_categorical(y_test, num_classes=classes)
print(f"y_train shape: {y_train.shape}\n y_test shape: {y_test.shape}")


def make_model(activator="relu", alpha=0.3, optimizer="sgd", 
        dense_layer_size=32, num_layers=3, 
        dropout_rate=0.1, loss="categorical_crossentropy", 
        metrics=["accuracy", "mse", "categorical_crossentropy"]):

    '''
    
    Usage: create a model with variables for 
    hyperparameters that can be replaced in GridSearch.
    Define anything you want, as long as it would be a 
    valid way to make a single model, then use those
    kwargs as inputs to GridSearch param grid.

    Tip: Set a default value for each kwarg.

    '''

    print("Building model . . . ")
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation(activator))
    
    # Add hidden layers
    for i in range(num_layers):
        model.add(Dense(dense_layer_size))
        model.add(keras.layers.LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model
   

''' code to fit and score a single model

fit_model = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=.2)
score = model.evaluate(x_test, y_test)
print(f"Test score: {score[0]}\nTest accuracy: {score[1]}")

'''

###                  ###
### For GridSearchCV ###
###                  ###

'''Define the parameters that may
be passed to GridSearch. 

param_grid can take any parameter defined as a
kwarg of the make_model function in which
you wrap your neural-network creation.

'''

activator_candidates = ["relu"]
alpha_candidates = [0.1, 0.3]
num_layer_candidates = [1, 3]
dense_size_candidates = [32, 64, 128]
dropout_candidates = [0.0, 0.1, 0.3]

loss_func_candidates = ["sparse_categorical_crossentropy", 
"cosh", "categorical_hinge"]
optimizer_candidates = ["sgd", "adam", "rmsprop"]

classifier = KerasClassifier(make_model, batch_size=32)

# If scoring param is not given to GridSearchCV,
# it will use the same scoring metrics as the estimator.
validator = GridSearchCV(classifier, param_grid={
    'dense_layer_size': dense_size_candidates,
    'alpha': alpha_candidates, 
    'num_layers': num_layer_candidates,
#    'dropout_rate': dropout_candidates,
    'optimizer': optimizer_candidates
    }, n_jobs=-1)

validator.fit(x_train, y_train)

print(f"The parameters of the best model are: {validator.best_params_}")

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model

best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)

for metric, value in zip(metric_names, metric_values): 
    print(f"{metric} :  {value}")

__name__ == "__main__"
