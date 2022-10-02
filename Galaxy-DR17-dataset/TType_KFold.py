
def build_model(num_components, model_name, lr):

  # xsize=212 #69
  # ysize=212 #69

  xsize=69
  ysize=69
  
    
  cnn = Sequential([
  Conv2D(32, (6,6), input_shape=(xsize,ysize,3), padding="same", activation='relu'),
                  #kernel_initializer='random_normal', bias_initializer='zeros'), 
  #tfkl.BatchNormalization(),  
  Dropout(0.5),

  Conv2D(64, (5,5), padding="same", activation='relu'),
  MaxPool2D(pool_size=(2, 2)),
  #tfkl.BatchNormalization(),  
  Dropout(0.25),
  
  Conv2D(128, (2,2), padding="same",activation='relu'),
  MaxPool2D(pool_size=(2, 2)),
  #tfkl.BatchNormalization(),
  Dropout(0.25),

  Conv2D(128, (3,3), padding="same", activation='relu'),
  #tfkl.BatchNormalization(), 
  Dropout(0.25),


   ## Dense layers
  Flatten(),
 
  Dense(128),
  Dropout(0.5),

  Dense(64),
  Dropout(0.5),

  Dense(1)


  #This is old
  #tfkl.Dense(64, activation='relu'),
  #tfkl.BatchNormalization(), 
  #tfkl.Dropout(0.5),
  
                                                           
    ])     

  print(cnn.summary())

  #pdb.set_trace()
  
  opt = Adam(lr=lr)
  cnn.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    
  #cnn.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])


  return cnn

####################################

def train_model(x_train, t_train, model, model_name, nb_epochs, batch_size):


    # Define validation
    ntrain=int(x_train.shape[0]*0.9) 
    nval=int(x_train.shape[0]*0.1)
    ind=random.sample(range(0, ntrain+nval-1), ntrain+nval-1)

    D_train = x_train[ind[0:ntrain],:,:,:]   
    D_val = x_train[ind[ntrain:ntrain+nval],:,:,:]
    Z_train = t_train[ind[0:ntrain]]
    Z_val = t_train[ind[ntrain:ntrain+nval]]

    print('Train and val shape', D_train.shape, Z_train.shape, D_val.shape, Z_val.shape)

    modelcheckpoint = ModelCheckpoint(pathout+ "/" +model_name+"_best.h5",
                                      monitor='val_loss',verbose=0,save_best_only=True)

    # Create an experiment with your api key
    # experiment = Experiment(
    #     api_key="jI7nb18fRtELRYBNZRyWIC7b3",
    #     project_name="ttype-kfold",
    #     workspace="helenadominguez",
    # )

    # experiment = Experiment(
    #     api_key="jI7nb18fRtELRYBNZRyWIC7b3",
    #     project_name="ttype-original",
    #     workspace="helenadominguez",
    # )

    experiment = Experiment(
        api_key="jI7nb18fRtELRYBNZRyWIC7b3",
        project_name="ttype-100-epochs",
        workspace="helenadominguez",
    )


    # this will do preprocessing and realtime data augmentation
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,
        zca_whitening=False, 
        rotation_range=45,
        width_shift_range=0.05,  
        height_shift_range=0.05, 
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.75,1.3],
        )

        
    datagen.fit(D_train)


    print("Training...")    
    history = model.fit_generator(
                datagen.flow(D_train, Z_train, batch_size=batch_size),           
                epochs=nb_epochs,            
                validation_data=(D_val, Z_val),
                #samples_per_epoch=D_train.shape[0],
                #callbacks=[CometLogger(experiment)]
                callbacks=[modelcheckpoint]
            )


    model.summary(print_fn=summary_to_file(model_name))
    plot_model(model, to_file='./{model_name}_model.png')
    
    # save weights
    print("Saving model...")
    model.save(pathout+ "/" +model_name+".h5",overwrite=True)

    experiment.end()

    return history


###############################################
  
def summary_to_file(model_id):

    def print_fn(s):
        with open(f"{model_id}_summary.txt", "a") as f:
            f.write(s + "\n")

    return print_fn


###############################################


import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import random

from comet_ml import Experiment
from comet_callback import CometLogger

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import KFold

from astropy.io import fits
from astropy.visualization.stretch import SqrtStretch
from astropy.visualization import ImageNormalize, MinMaxInterval

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam, SGD


from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential



python='3'


#Load data
#===============
pathinData="/lhome/ext/ice043/ice0431/helena/MPL-9"

X = np.load(pathinData+'/image_vector_13454.npy') # (13454, 69, 69, 3)
Y = np.load(pathinData+'/target_vector_13454.npy') #(13408,)
ID = np.load(pathinData+'/ID_vector_13454.npy')


# X = np.load(pathinData+'/image_vector_13454_212x212.npy') # (13454, 69, 69, 3)
# Y = np.load(pathinData+'/target_vector_13454_212x212.npy') #(13408,)
# ID = np.load(pathinData+'/ID_vector_13454_212x212.npy')


#remove missing images
X=X[ID!=-999, :, :, :] #(13454, 69, 69, 3) (bad crop eran 13273)
Y=Y[ID!=-999]
ID=ID[ID!=-999]


#fill empty bins
Y_o=np.copy(Y)  #Very important to avoid overwritting!!

Y[Y_o==-5]=-3
Y[Y_o==-3]=-2
Y[Y_o==-2]=-1

#pdb.set_trace()

X, Y, ID = shuffle(X, Y, ID, random_state=0)

# Y rescaling
from sklearn.preprocessing import StandardScaler,  MinMaxScaler
scalerY =  MinMaxScaler().fit(Y.reshape((-1,1)))
Ys = scalerY.transform(Y.reshape((-1,1)))


## images normalization
print('Min/Max before normalization:', np.max(X), np.min(X))

mu = np.amax(X,axis=(1,2))
print(mu.shape)
for i in range(0,mu.shape[0]):
    #print i
    X[i,:,:, 0] = X[i,:,:, 0]/mu[i,0]
    X[i,:,:, 1] = X[i,:,:, 1]/mu[i,1]
    X[i,:,:, 2] = X[i,:,:, 2]/mu[i,2]

print('Min/Max after normalization:', np.max(X), np.min(X))


# Spliting in Training and Test datasets
X_train = X[0:len(X)//10*9, : , :, :]  #12060
X_test = X[len(X)//10*9:,:, :, :]      #1348

T_train = Ys[0:len(Ys)//10*9]
T_test = Ys[len(Ys)//10*9:]

Y_train = Y[0:len(Y)//10*9]
Y_test = Y[len(Y)//10*9:]

Yo_train = Y_o[0:len(Y)//5*4]
Yo_test = Y_o[len(Y)//5*4:]


ID_train = ID[0:len(ID)//10*9]
ID_test = ID[len(ID)//10*9:]


######## K-folding ############

num_run=0

kf = KFold(n_splits=15)
#kf = KFold(n_splits=10)

kf.get_n_splits(X_train)
print(kf)

for train_index, test_index in kf.split(X_train):

    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    t_train, t_test = T_train[train_index], T_train[test_index]
    id_train, id_test = ID_train[train_index], ID_train[test_index]


    print('Model number  ', num_run)
    print('=======================')

    print ('X_train.shape= ', x_train.shape)  # 3582
    print ('Y_train.shape= ', y_train.shape)
    print ('T_train.shape= ', t_train.shape)
    print ('X_test.shape= ', x_test.shape)
    print ('Y_test.shape= ', y_test.shape)

    num_run=num_run+1

    #pdb.set_trace()


    ###### MODEL SET UP ########

    #Define output path
    pathout='/lhome/ext/ice043/ice0431/helena/MPL-9/TType_KFold/100_epochs_batch_32'

    num_components=1
    model_name='model_128_64_lr-5_100_epochs_batch_32_run_'+str(num_run)+''
    lr=0.00002
    #lr=0.0002
    nb_epochs=100
    batch_size = 300 #changed with 100 epochs, before 32


    hyperparams = dict(
        img_width = 69,
        img_height = 69,
        img_channels = 3,
        train_epochs = nb_epochs,
        train_size = len(Y_train),
        batch_size = batch_size,
        pos_batch_ratio = 0.50,

        # cnn params
        n_classes = 1,
        conv_filters = [32, 64, 128, 128],
        fully_connected_neurons = [128, 64],
        activation = "relu",
        use_batch_norm = False,
        dropout_rate = 0.25,

        # augmentation params
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 359
    )


    cnn = build_model(num_components, model_name, lr)
    history = train_model(x_train, t_train, cnn, model_name, nb_epochs, batch_size )


    # Plot distribution
    # ===============
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pathout+'/'+'input_TType_'+model_name+'.pdf') as pdf:

        plt.figure()

        plt.hist(y_train, color='b', label='Y train', bins=np.arange(-5., 13., 1))
        plt.hist(y_test, color='r', label='Y test', bins=np.arange(-5., 13., 1))
        plt.xlabel(r'$TType$')
        plt.xlim(-6, 13.)
        plt.legend()

        pdf.savefig()
        plt.close()

    print ('Y_train.shape= ', t_train.shape)
    print ('X_train.shape= ', x_train.shape)

    print(np.max(t_train), np.min(t_train))

    #Plot loss
    #===============
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pathout+"/"+'trainning_loss_'+model_name+'.pdf') as pdf:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        pdf.savefig()
        plt.close()


    #===== Predictions =====

    if python=='2':
      outputs = cnn(X_test)
      Y_pred = scalerY.inverse_transform(outputs.numpy().reshape(-1,1))

      cnn.load_weights(pathout + "/" + model_name + "_best.h5")
      outputs = cnn.predict_proba(X_test)
      Y_pred_best = scalerY.inverse_transform(outputs.numpy().reshape(-1, 1))


    if python=='3':

      # Last model
      outputs = cnn.predict_proba(x_test)
      Y_pred = scalerY.inverse_transform(outputs.reshape(-1,1))

      outputs = cnn.predict_proba(X_test)
      Y_pred_reserved = scalerY.inverse_transform(outputs.reshape(-1, 1))

      #Best model
      cnn.load_weights(pathout + "/" + model_name + "_best.h5")
      outputs = cnn.predict_proba(x_test)
      Y_pred_best = scalerY.inverse_transform(outputs.reshape(-1,1))

      outputs = cnn.predict_proba(X_test)
      Y_pred_best_reserved = scalerY.inverse_transform(outputs.reshape(-1, 1))

      #pdb.set_trace()

    #=========== save file with input/output ===============

    Y_val = np.array(y_test)
    Y_val_reserved = np.array(Y_test)


    print("Saving Y_pred.fit")
    print("====================")
    col1 = fits.Column(name='ID', format='K', array=id_test)
    col2 = fits.Column(name='Y_pred', format='F', array=Y_pred)
    col3 = fits.Column(name='Y_pred_best', format='F', array=Y_pred_best)
    col4 = fits.Column(name='Y_in', format='F', array=Y_val)

    cols = fits.ColDefs([col1, col2, col3, col4])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(pathout+'/'+model_name+".fit",overwrite='True')


    print("Saving Y_pred_reserved.fit")
    print("====================")
    col1 = fits.Column(name='ID', format='K', array=ID_test)
    col2 = fits.Column(name='Y_pred', format='F', array=Y_pred_reserved)
    col3 = fits.Column(name='Y_pred_best', format='F', array=Y_pred_best_reserved)
    col4 = fits.Column(name='Y_in', format='F', array=Y_val_reserved)

    cols = fits.ColDefs([col1, col2, col3, col4])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(pathout + '/' + model_name + "_reserved.fit", overwrite='True')


#pdb.set_trace()

 
##########################################
    

