from numpy import mean
from numpy import std
from numpy import dstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import *
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report,accuracy_score,recall_score,precision_score,auc,roc_curve
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

s25 = np.load('s25.npy')
h25 = np.load('h25.npy')

s25_labels = np.array([0 for _ in range(0,len(s25))])
h25_labels = np.array([1 for _ in range(0,len(h25))])

X = np.append(s25,h25,axis=0)
y = np.append(s25_labels,h25_labels,axis=0)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0,regularizer=False):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    if regularizer:
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
        )(x, x)
    else:
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    if regularizer:
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu",
                         kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1,
                         kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    else:
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

n_classes = 2

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    regularizer=False
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout,regularizer)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    if regularizer:
        outputs = layers.Dense(n_classes, activation="softmax",
                              kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def evaluate_model(trainX, trainy, testX, testy,regularizer=False):
    verbose, epochs, batch_size = 0, 50, 2
    input_shape = trainX.shape[1:]
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=2,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,regularizer=regularizer
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    # fit network
    history=model.fit(trainX, trainy, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    y_pred = model.predict(testX, batch_size=batch_size, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)

    fpr, tpr, thresholds = roc_curve(testy, y_pred[:,1])
    
    
    #_, accuracy,precision,recall,auc = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy_score(testy,y_pred_bool),precision_score(testy,y_pred_bool),recall_score(testy,y_pred_bool),auc(fpr,tpr),history

def summarize_results(accuracies,precisions,recalls,aucs):
    m, s = mean(accuracies), std(accuracies)
    print( ' Accuracy: %.3f%% (+/-%.3f) ' % (m, s))
    m, s = mean(precisions), std(precisions)
    print( ' Precision: %.3f%% (+/-%.3f) ' % (m, s))
    m, s = mean(recalls), std(recalls)
    print( ' Recall: %.3f%% (+/-%.3f) ' % (m, s))
    m, s = mean(aucs), std(aucs)
    print( ' AUC: %.3f%% (+/-%.3f) ' % (m, s))
    
def run_experiment(X,y,repeats=3):
    # load data
    trainX, testX,trainy, testy = train_test_split(X, y, test_size=0.20, random_state=42)
    # repeat experiment
    accuracies = list()
    precisions = list()
    recalls = list()
    aucs = list()
    histories = list()
    for r in range(repeats):
        accuracy,precision,recall,auc,history = evaluate_model(trainX, trainy, testX, testy)
        histories.append(history)
        accuracy = accuracy * 100.0
        precision = precision * 100.0
        recall = recall * 100.0
        auc = auc * 100.0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(auc)
    # summarize results
    summarize_results(accuracies,precisions,recalls,aucs)
    return histories

histories=run_experiment(X,y)

mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))
plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation loss and accuracy')
plt.ylabel('loss / accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc','train_loss','valid_loss'], loc='center right')
plt.savefig('images/combined-transformer2.png')
plt.show()
plt.clf()

mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc'], loc='lower right')
plt.savefig('images/learning-curve-transformer2.png')
plt.show()
plt.clf()


plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))
plt.title('Mean (5-fold) training/validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_loss','valid_loss'], loc='center right')
plt.savefig('images/loss-transformer2.png')
plt.show()
plt.clf()


def run_experiment_2(X,y,repeats=3):
    # load data
    trainX, testX,trainy, testy = train_test_split(X, y, test_size=0.20, random_state=42)
    mean = trainX.mean(axis=0)
    trainX -= mean
    std = trainX.std(axis=0)
    trainX /= std
    
    testX -= mean
    testX /= std
    # repeat experiment
    accuracies = list()
    precisions = list()
    recalls = list()
    aucs = list()
    histories = list()
    for r in range(repeats):
        accuracy,precision,recall,auc,history = evaluate_model(trainX, trainy, testX, testy)
        histories.append(history)
        accuracy = accuracy * 100.0
        precision = precision * 100.0
        recall = recall * 100.0
        auc = auc * 100.0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(auc)
    # summarize results
    summarize_results(accuracies,precisions,recalls,aucs)
    return histories

histories=run_experiment_2(X,y)

mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))
plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation loss and accuracy')
plt.ylabel('loss / accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc','train_loss','valid_loss'], loc='center right')
plt.savefig('images/combined-transformer2-normalized.png')
plt.show()
plt.clf()


mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc'], loc='lower right')
plt.savefig('images/learning-curve-transformer2-normalized.png')
plt.show()
plt.clf()

plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))
plt.title('Mean (5-fold) training/validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_loss','valid_loss'], loc='center right')
plt.savefig('images/loss-transformer2-normalized.png')
plt.show()
plt.clf()


def run_experiment_3(X,y,repeats=3):
    # load data
    trainX, testX,trainy, testy = train_test_split(X, y, test_size=0.20, random_state=42)
    mean = trainX.mean(axis=0)
    trainX -= mean
    std = trainX.std(axis=0)
    trainX /= std
    
    testX -= mean
    testX /= std
    # repeat experiment
    accuracies = list()
    precisions = list()
    recalls = list()
    aucs = list()
    histories = list()
    for r in range(repeats):
        accuracy,precision,recall,auc,history = evaluate_model(trainX, trainy, testX, testy,regularizer=True)
        histories.append(history)
        accuracy = accuracy * 100.0
        precision = precision * 100.0
        recall = recall * 100.0
        auc = auc * 100.0
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(auc)
    # summarize results
    summarize_results(accuracies,precisions,recalls,aucs)
    return histories

histories=run_experiment_3(X,y)

mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))
plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation loss and accuracy')
plt.ylabel('loss / accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc','train_loss','valid_loss'], loc='center right')
plt.savefig('images/combined-transformer2-normalized-regularized.png')
plt.show()
plt.clf()


mean_acc = np.mean([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
mean_std = np.std([h.history['sparse_categorical_accuracy'] for h in histories],axis=0)
plt.plot(mean_acc)
plt.plot(np.mean([h.history['val_sparse_categorical_accuracy'] for h in histories],axis=0))

plt.title('Mean (5-fold) training/validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_acc', 'valid_acc'], loc='lower right')
plt.savefig('images/learning-curve-transformer2-normalized-regularized.png')
plt.show()
plt.clf()

plt.plot(np.mean([h.history['loss'] for h in histories],axis=0))
plt.plot(np.mean([h.history['val_loss'] for h in histories],axis=0))
plt.title('Mean (5-fold) training/validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
# summarize history for loss
plt.legend(['train_loss','valid_loss'], loc='center right')
plt.savefig('images/loss-transformer2-normalized-regularized.png')
plt.show()
plt.clf()
