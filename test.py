from time import time
import keras.backend as K
import tensorflow as tf
import datetime
import random
import rng
import os
from shutil import copyfile


from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.utils import plot_model
from sklearn.metrics import silhouette_samples, silhouette_score
class ClusteringLayer(Layer):
    # '''
    # Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    # sample belonging to each cluster. The probability is calculated with student's t-distribution.
    # '''

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.keras.backend.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        # ''' 
        # student t-distribution, as used in t-SNE algorithm.
        # It measures the similarity between embedded point z_i and centroid µ_j.
        #          q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
        #          q_ij can be interpreted as the probability of assigning sample i to cluster j.
        #          (i.e., a soft assignment)
       
        # inputs: the variable containing data, shape=(n_samples, n_features)
        
        # Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        #'''
        q = 1.0 / (1.0 + (tf.keras.backend.sum(tf.keras.backend.square(tf.keras.backend.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.keras.backend.transpose(tf.keras.backend.transpose(q) / tf.keras.backend.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def autoencoderFun(dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric auto-encoder model.
  
    dims: list of the sizes of layers of encoder like [500, 500, 2000, 10]. 
          dims[0] is input dim, dims[-1] is size of the latent hidden layer.

    act: activation function
    
    return:
        (autoencoder_model, encoder_model): Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    
    input_data = Input(shape=(dims[0],), name='input')
    x = input_data
    
    # internal layers of encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)

    x = encoded
    # internal layers of decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    
    decoded = x
    
    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model     = Model(inputs=input_data, outputs=encoded, name='encoder')
    
    return autoencoder_model, encoder_model


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

importedData = pd.read_csv('C:/Users/Raymond Fey/Desktop/New Folder/datasets/cancer.csv')

numeric_columns = importedData.columns.values.tolist()
scaler = MinMaxScaler()
importedData[numeric_columns] = scaler.fit_transform(importedData[numeric_columns])

x = importedData.values
x.shape

n_clusters = 5
n_epochs   = 1000
batch_size = 250
dims = [6, 500, 500, 2000, 10] 

loss1=['mean_squared_error','mean_absolute_error','mean_squared_logarithmic_error','log_cosh']
iterations=0
while(1):
    rngIndex2=random.randint(0,len(loss1)-1)
    rngIndex3=random.randint(0,len(loss1)-1)
    init = VarianceScaling(scale=1. / 3., mode='fan_out', distribution='untruncated_normal')

    kmeans = KMeans(n_clusters=n_clusters)
    y_pred_kmeans = kmeans.fit_predict(x)
    autoencoder, encoder = autoencoderFun(dims, init=init)
    pretrain_optimizer = SGD(lr=0.01, momentum=0.9)
    pretrain_epochs = n_epochs
    batch_size = batch_size
    save_dir = 'C:/Users/Raymond Fey/Desktop/New Folder/results'

    print(loss1[rngIndex2])
    print(loss1[rngIndex3])
    e = datetime.datetime.now()

    autoencoder.compile(optimizer=pretrain_optimizer, loss=loss1[rngIndex2])
    autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder.save_weights(save_dir + '/ae_weights_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.h5')

    autoencoder.load_weights(save_dir + '/ae_weights_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.h5')

    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)

    model.compile(optimizer=SGD(0.01, 0.9), loss=loss1[rngIndex3])

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(x))

    y_pred_last = np.copy(y_pred)

    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    loss = 0
    index = 0
    maxiter = 1000 # 8000
    update_interval = 100 # 140
    index_array = np.arange(x.shape[0])

    tol = 0.001

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x, verbose=1)
            p = target_distribution(q)  # update the auxiliary target distribution p

        idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
        loss = model.train_on_batch(x=x[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    model.save_weights(save_dir + '/DEC_model_final_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.h5')

    model.load_weights(save_dir + '/DEC_model_final_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.h5')

    # Eval.
    q = model.predict(x, verbose=1)
    p = target_distribution(q)  # update the auxiliary target distribution p

    # evaluate the clustering performance
    y_pred = q.argmax(1)

    data_all = importedData.copy()

    data_all['cluster'] = y_pred

    x_embedded = TSNE(n_components=2).fit_transform(x)

    lel = data_all['cluster'].value_counts()
    print(lel, file=open(save_dir + '/Report_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.txt', "a"))

    x_embedded.shape
    vis_x = x_embedded[:, 0]
    vis_y = x_embedded[:, 1]

    if(iterations%25==0 and iterations!=0 ):
        rng.multiModal2()
        print('New Data Set Generated')

    if((data_all['cluster'].value_counts().min() > 600) and (data_all['cluster'].value_counts().max() < 1500)):
        print('Autoencoder Loss Function used: ' +loss1[rngIndex2])
        print('Compile Loss Function used: ' + loss1[rngIndex3])
        print('Autoencoder Loss Function used: ' +loss1[rngIndex2] + "\n" + 'Compile Loss Function used: ' + loss1[rngIndex3], file=open(save_dir + '/Success_' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.txt', "a"))
        copyfile('datasets/cancer.csv' , 'datasets/cancer' + str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second) + '.csv')
    iterations +=1

    


# plt.figure()
# plt.title(str(e.month) + '_' + str(e.day) + '_' + str(e.year) + '_' + str(e.hour) + '_' + str(e.minute) + '_' + str(e.second))
# plt.scatter(vis_x, vis_y, c=y_pred, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(256))
# plt.clim(-0.5, 9.5)

# plt.show()