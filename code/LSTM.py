# This model structure is based on 
# Next Basket Recommendation with Neural Networks 
# http://ceur-ws.org/Vol-1441/recsys2015_poster15.pdf

######################################
# import packages
######################################
print("Load package...")
import os
import sys
import numpy as np
import pandas as pd
import sys

from string import punctuation
from collections import defaultdict
import ast

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Activation, merge, LSTM
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras import metrics
from keras import regularizers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

from F_score import f_score


######################################
# Denfine parameters and constants
######################################
# define constants and parameters
print("Define parameters...")
INPUT_DIR = '../input/'
TRANSFORM_DIR = "../transform/"
OUTOUT_DIR = "../output/"
PRIOR_DATA_FILE = TRANSFORM_DIR + 'orders_prior_merge.csv'
TRAIN_DATA_FILE = TRANSFORM_DIR + 'orders_train_merge.csv'
TEST_DATA_FILE = TRANSFORM_DIR + 'orders_test.csv'

VALIDATION_SPLIT_RATIO = 0.1

num_dense = np.random.randint(100, 120)
rate_drop_dense = 0.15 + np.random.rand() * 0.25
STAMP = 'nn_%d_%.2f'%(num_dense, rate_drop_dense)


######################################
# load data
######################################
print("Load data...")
orders_train_merge = pd.read_csv(TRAIN_DATA_FILE)
orders_prior_merge = pd.read_csv(PRIOR_DATA_FILE)
orders_test = pd.read_csv(TEST_DATA_FILE)
products = pd.read_csv(INPUT_DIR+"products.csv")

# Calculate constants from data
MAX_ITEM = max(max(orders_train_merge["product_id"].apply(lambda x:len(ast.literal_eval(x)))), 
               max(orders_prior_merge["product_id"].apply(lambda x:len(ast.literal_eval(x)))))
PRE_BASKET = min(min(orders_train_merge.order_number), min(orders_test.order_number))-1


######################################
# Train and test split
######################################
print("Train and test split...")
# get useful columns
prior = orders_prior_merge[["order_id","user_id","product_id"]]
train = orders_train_merge[["order_id","user_id","product_id"]]
test = orders_test[["order_id","user_id"]]

# create user_id dict to accomodate users in train and test
train_usr_dict = dict.fromkeys(train.user_id)
test_usr_dict = dict.fromkeys(test.user_id)

# split train and test prior data
prior_train = prior.ix[[x in train_usr_dict for x in list(prior.user_id)],:]
prior_test = prior.ix[[x in test_usr_dict for x in list(prior.user_id)],:]


######################################
# Create train and val set based on train above
######################################
print("Train and val split...")
# Train Validation split
def split_train_val(data, val_ratio):
    shuffled_indices = np.random.permutation(len(data))
    val_set_size = int(len(data)*val_ratio)
    val_indices = shuffled_indices[:val_set_size]
    train_indices = shuffled_indices[val_set_size:]
    return data.iloc[train_indices], data.iloc[val_indices]

np.random.seed(1992)
train_train, train_val = split_train_val(train, VALIDATION_SPLIT_RATIO)

# create user_id dict for train_train and train_val
train_train_usr_dict = dict.fromkeys(train_train.user_id)
train_val_usr_dict = dict.fromkeys(train_val.user_id)

# split train_train and train_val prior data
prior_train_train = prior_train.ix[[x in train_train_usr_dict for x in list(prior_train.user_id)],:]
prior_train_val = prior_train.ix[[x in train_val_usr_dict for x in list(prior_train.user_id)],:]



######################################
# Preprocessing data for modeling
######################################
print("Preprocessing data...")
# prior
def get_pre_baskets(prior):
    return prior.groupby("user_id").tail(3)

train_train_baskets = get_pre_baskets(prior_train_train)
train_val_baskets = get_pre_baskets(prior_train_val)
test_baskets = get_pre_baskets(prior_test)

assert min(train_val_baskets.groupby("user_id").apply(len)) == PRE_BASKET
assert min(train_train_baskets.groupby("user_id").apply(len)) == PRE_BASKET

# extract sequence of items from pandas frame
sequence_train_train = list(map(ast.literal_eval,train_train_baskets["product_id"].tolist()))
sequence_train_val = list(map(ast.literal_eval,train_val_baskets["product_id"].tolist()))
sequence_test = list(map(ast.literal_eval,test_baskets["product_id"].tolist()))

# pad all train and val with max items
data_train_train = pad_sequences(sequence_train_train, maxlen=MAX_ITEM, padding='post', truncating='post')
data_train_val = pad_sequences(sequence_train_val, maxlen=MAX_ITEM, padding='post', truncating='post')
data_test = pad_sequences(sequence_test, maxlen=MAX_ITEM, padding='post', truncating='post')

# extract sequence of items from pandas frame
sequence_train_train = list(map(ast.literal_eval,train_train_baskets["product_id"].tolist()))
sequence_train_val = list(map(ast.literal_eval,train_val_baskets["product_id"].tolist()))
sequence_test = list(map(ast.literal_eval,test_baskets["product_id"].tolist()))

# extract data for three previous purchase of each user for train, val and test
data_train_train_bs1 = data_train_train[0::3]
data_train_train_bs2 = data_train_train[1::3]
data_train_train_bs3 = data_train_train[2::3]

data_train_val_bs1 = data_train_val[0::3]
data_train_val_bs2 = data_train_val[1::3]
data_train_val_bs3 = data_train_val[2::3]

data_test_bs1 = data_test[0::3]
data_test_bs2 = data_test[1::3]
data_test_bs3 = data_test[2::3]

# extract label sequence
label_sequence_train_train = list(map(ast.literal_eval,train_train["product_id"].tolist()))
label_sequence_train_val = list(map(ast.literal_eval,train_val["product_id"].tolist()))

# concatenate the three buskets to one purchase sequence for each user
data_train_train_one_user = np.concatenate([data_train_train_bs1,data_train_train_bs2,data_train_train_bs3], axis = 1)
mlb = MultiLabelBinarizer(classes=products["product_id"].tolist(), sparse_output=True)

# encode the label sequence
train_train_label = mlb.fit_transform(label_sequence_train_train)
train_val_label = mlb.fit_transform(label_sequence_train_val)

weight_val = np.ones(MAX_ITEM)


######################################
# Create mask for cross entropy 
######################################
print("Creating mask for cross entropy...")
# create user buying history
prior_sequence = list(map(ast.literal_eval,prior["product_id"].tolist()))
prior_usr = prior["user_id"]
usr_history = {k: [] for k in set(prior_usr)}  

for i in range(len(prior_usr)):
    usr_history[prior_usr[i]].extend(prior_sequence[i])
usr_history_encode = [usr_history[x] for x in range(1, len(usr_history)+1)]

# create user list for train, val and test
mlb = MultiLabelBinarizer(classes=range(1,49689), sparse_output=True)
usr_history = mlb.fit_transform(usr_history_encode)

# create user purchase history and separate for train, val and test
train_train_usr = train_train_baskets.ix[0::3, "user_id"].values
train_val_usr = train_val_baskets.ix[0::3, "user_id"].values
test_usr = test_baskets.ix[0::3, "user_id"].values

del prior
del train
del test
del train_usr_dict
del test_usr_dict
del prior_train
del prior_test

######################################
# Define the model graph
######################################
print("Define graph...")
# parameters
EMBEDDING_DIM = 100
num_products = len(products)+1

# graph
product_embedding_layer = Embedding(input_dim=num_products,
        output_dim=EMBEDDING_DIM,
        embeddings_initializer='normal',
        mask_zero=True,
        input_length=3*MAX_ITEM,
        trainable=True)

lstm_layer = LSTM(100)

bs_input = Input(shape=(3*MAX_ITEM,), dtype='int32')
b_hist = Input(shape=(49688,), dtype='float32')

embedded_bs_input = product_embedding_layer(bs_input)
lstm_out = lstm_layer(embedded_bs_input)

merged = BatchNormalization()(lstm_out)
merged = Dropout(rate_drop_dense)(merged)
merged = Dense(num_dense, kernel_initializer='normal', activation="relu")(merged)

merged = BatchNormalization()(lstm_out)
merged = Dropout(rate_drop_dense)(merged)
preds = Dense(num_products-1, kernel_initializer='normal', activation='sigmoid')(merged)
mask_preds = merge([preds, b_hist], mode='mul')

######################################
# Compile model 
######################################
model = Model(inputs=[bs_input,b_hist], outputs=mask_preds)
model.compile(loss='binary_crossentropy', \
        optimizer='adam')
print(STAMP)
print("The model structure is:")
model.summary()

######################################
# Train model 
######################################
print("Start training...")
# set early stopping
early_stopping =EarlyStopping(monitor='val_loss', patience=3)  
bst_model_path = OUTOUT_DIR + STAMP + '.h5' 
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

# Batch generator
def batch_generator(data_train_train_one_user, user_dict, buy_history, label_mat, batch_size):
    N = np.shape(label_mat)[0]
    number_of_batches = N/batch_size
    counter=0
    shuffle_index = np.arange(N)
    np.random.shuffle(shuffle_index)
    while True:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        data_input = data_train_train_one_user[index_batch, :]
        u_id = user_dict[index_batch]
        b_hist_input = buy_history[u_id,:].todense()
        label_input = np.asarray(label_mat[index_batch].todense())
        counter += 1
        yield([data_input, b_hist_input],label_input)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

# Train the model
hist = model.fit_generator(generator=batch_generator(
                    data_train_train_one_user[0:1000], train_train_usr,\
                    usr_history, train_train_label[0:1000], 20), \
                    validation_data=([data_train_train_one_user[0:1000], usr_history[train_train_usr[0:1000],:].todense()],
                                      train_train_label[0:1000].todense()), \
                    nb_epoch=20, steps_per_epoch=np.shape(train_train_label[0:1000])[0]/20, \
                    callbacks=[early_stopping, model_checkpoint]
                   )


# model.load_weights(bst_model_path) # sotre model parameters in .h5 file
# bst_val_score = min(hist.history['val_loss'])
# usr_history[train_train_usr[0:1000],:].todense()

# make the prediction
print('Making prediction')
preds = model.predict([data_train_train_one_user[0:1000], usr_history[train_train_usr[0:1000],:].todense()],
                        batch_size=128, verbose=1)

# print("preds")
# print(preds)

# print("arg_sorted_preds")
# print(np.argsort(preds))

# print("Max prob in each row")
# trueth = np.array(train_train_label[0:1000].todense())
# print((trueth * preds).max(axis = 1))

