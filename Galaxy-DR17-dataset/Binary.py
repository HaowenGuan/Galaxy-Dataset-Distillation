
def build_model_DS18(num_components, model_name, lr):

    
    img_channels=3
    img_rows=69
    img_cols=69
    
    #========= Model definition (as in GZOO convnet)=======
    
    model = Sequential()
    model.add(Conv2D(32, (6,6), padding="same",
                        input_shape=(img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(Activation('relu'))
    
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (2, 2), padding="same"))
    model.add(Activation('relu'))


    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))

    #Neuronal Networks starts here
    #-------------------------------#
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5)) 
    model.add(Dense(1,  activation='sigmoid')) #init='uniform',
   
    print("Compilation...")
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    print("Model Summary")
    print("===================")
    model.summary()

    return model


#######################################################

def build_model(num_components, model_name, lr):

  #pdb.set_trace()
    
  cnn = Sequential([
  Conv2D(32, (6,6), input_shape=(69,69,3), border_mode="same", activation='relu'),
                  #kernel_initializer='random_normal', bias_initializer='zeros'), 
  #tfkl.BatchNormalization(),  
  Dropout(0.5),

  Conv2D(64, (5,5), padding="same", activation='relu'), #padding=
  MaxPool2D(pool_size=(2, 2)),
  #tfkl.BatchNormalization(),  
  Dropout(0.25),
  #Dropout(0.5),
  
  Conv2D(128, (2,2), padding="same",activation='relu'),
  MaxPool2D(pool_size=(2, 2)),
  #tfkl.BatchNormalization(),
  Dropout(0.25),
  #Dropout(0.5),

  Conv2D(128, (3,3), padding="same", activation='relu'),
  #tfkl.BatchNormalization(), 
  Dropout(0.25),
  #Dropout(0.5),


  ## Dense layers
  Flatten(),
 
##  Dense(128),
##  Dropout(0.5),

  Dense(64, activation='relu'),
  Dropout(0.5),

  Dense(1, init='uniform', activation='sigmoid')
                                                           
    ])     

  print(cnn.summary())

  cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


  return cnn

####################################

def train_model(x_train, y_train, model, model_name, nb_epochs):


    batch_size=30

    patience_par = 10
    earlystopping = EarlyStopping(monitor='val_loss', patience=patience_par, verbose=0, mode='auto')
    modelcheckpoint = ModelCheckpoint('Binary/'+model_name+"_best.h5",
                                      monitor='val_loss', verbose=0, save_best_only=True)


    # Define validation
    ntrain=int(x_train.shape[0]*0.9) 
    nval=int(x_train.shape[0]*0.1)
    ind=random.sample(range(0, ntrain+nval-1), ntrain+nval-1)

    D_train = x_train[ind[0:ntrain], :, :, :]
    D_val = x_train[ind[ntrain:ntrain+nval], :, :, :]
    Z_train = y_train[ind[0:ntrain]]
    Z_val = y_train[ind[ntrain:ntrain+nval]]

    print('Train and val shape', D_train.shape, Z_train.shape, D_val.shape, Z_val.shape)

    # Create an experiment with your api key:
    experiment = Experiment(
        api_key="jI7nb18fRtELRYBNZRyWIC7b3",
        project_name="binary",
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
    history = model.fit(
                datagen.flow(D_train, Z_train,
                batch_size=batch_size),
                epochs=nb_epochs,
                validation_data=(D_val, Z_val),
                callbacks=[modelcheckpoint]
            )

    
    # save weights
    print("Saving model...")
    #model.save_weights(model_name+".hd5",overwrite=True)
    model.save('Binary/'+model_name+".h5", overwrite=True)

    experiment.end()

    return history



###############################################

def balanced_class_weights(y):
   counter = Counter(y)
   majority = max(counter.values())
   return  {cls: float(majority/count) for cls, count in counter.items()}

##################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle
import random
from collections import Counter

from comet_ml import Experiment
from comet_callback import CometLogger

import tensorflow as tf


from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import preprocessing

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


#remove missing images
X=X[ID!=-999, :, :, :] #(13408, 69, 69, 3)
Y=Y[ID!=-999]
ID=ID[ID!=-999]


#separete E/S0
Y_o=np.copy(Y)    #Very important to avoid overwritting!!
#Y[Y_o <= 0]=0    #2536
Y[Y_o <= 0]=0    #4433
Y[Y_o > 0 ]=1    #7505

class_weight = balanced_class_weights(Y)



X, Y, Y_o, ID = shuffle(X, Y, Y_o, ID) #, random_state=0)

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


# Reserve a sample that never passes through the machine
frac=8.5

X_train = X[0: int(len(X)//10*frac), :, :, :]  #24640
X_test = X[int(len(X)//10*frac):, :, :, :]      #

Y_train = Y[0: int(len(Y)//10*frac)]
Y_test = Y[int(len(Y)//10*frac): ]

Yo_train = Y_o[0: int(len(Y)//10*frac)]
Yo_test = Y_o[int(len(Y)//10*frac): ]

ID_train = ID[0: int(len(ID)//10*frac)]
ID_test = ID[int(len(ID)//10*frac) :]


class_weight = balanced_class_weights(Y_train)

print ('Reserved sample shape ', X_test.shape)
print ('KFold sample shape ', X_train.shape)
print('Class weights', class_weight)

# Reserved sample shape  (2018, 69, 69, 3)
# KFold sample shape  (11390, 69, 69, 3)


print(np.max(Y_train),np.min(Y_train))



# K-folding

num_run=0

kf = KFold(n_splits=10)
kf.get_n_splits(X)
print(kf)


# remove S0 ONLY from trainning, not from reserved sample
X_train = X_train[Yo_train != 0, :, :, :]  # (11938, 69, 69, 3)
ID_train = ID_train[Yo_train != 0]
Y_train = Y_train[Yo_train != 0]


for train_index, test_index in kf.split(X_train):


    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]

    yo_train, yo_test = Yo_train[train_index], Yo_train[test_index]
    id_train, id_test = ID_train[train_index], ID_train[test_index]



    print ('X_train.shape= ', x_train.shape)  # 3582
    print ('Y_train.shape= ', y_train.shape)
    print ('Yo_train.shape= ', yo_train.shape)
    print ('X_test.shape= ', x_test.shape)
    print ('Y_test.shape= ', y_test.shape)
    print ('Yo_test.shape= ', yo_test.shape)

    # X_train.shape = (10251, 69, 69, 3)
    # Y_train.shape = (10251,)
    # Yo_train.shape = (10251,)
    # X_test.shape = (1139, 69, 69, 3)
    # Y_test.shape = (1139,)
    # Yo_test.shape = (1139,)

    #pdb.set_trace()

    num_run=num_run+1

    #Define output path
    pathout='/lhome/ext/ice043/ice0431/helena/MPL-9/Binary/'
    model_name = 'model_Binary_no_S0_run_' + str(num_run) + ''

    #Plot distribution
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pathout+'input_Binary_'+model_name+'.pdf') as pdf:

      plt.figure()
      bins = np.arange(-5, 1, 1)
      plt.hist(yo_train, color='b', label='Y train', bins=bins )
      plt.hist(yo_test, color='r', label='Y test', bins=bins)
      plt.xlabel(r'$Class$')
      plt.xlim(-6, 2)
      plt.legend()

      pdf.savefig()
      plt.close()



    ###### MODEL SET UP ########


    num_components=1
    #lr=0.00002
    lr=0.001
    nb_epochs=100



    #cnn = build_model(num_components, model_name, lr)
    cnn = build_model_DS18(num_components, model_name, lr)
    history = train_model(x_train, y_train, cnn, model_name,
                          nb_epochs) #, class_weight)


    #Plot loss
    #===============
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pathout+'trainning_loss_'+model_name+'.pdf') as pdf:
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
      Y_pred = cnn(x_test)



    if python=='3':
      Y_pred = cnn.predict_proba(x_test)
      Y_pred_reserved = cnn.predict_proba(X_test)

      cnn.load_weights('Binary/'+model_name+"_best.h5")
      Y_best = cnn.predict_proba(x_test)
      Y_best_reserved = cnn.predict_proba(X_test)


    #=========== save file with input/output ===============

    Y_val = np.array(y_test)
    Yo_val = np.array(yo_test)

    print("Saving Y_pred.fit")
    print("====================")
    col1 = fits.Column(name='ID', format='K', array=id_test)
    col2 = fits.Column(name='Y_pred', format='F', array=Y_pred)
    col3 = fits.Column(name='Y_in', format='F', array=Y_val)
    col4 = fits.Column(name='Yo_in', format='F', array=Yo_val)
    col5 = fits.Column(name='Y_best', format='F', array=Y_best)

    cols = fits.ColDefs([col1, col2, col3, col4, col5])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(pathout + '/' + model_name + ".fit", overwrite='True')

    Y_val_reserved = np.array(Y_test)
    Yo_val_reserved = np.array(Yo_test)

    print("Saving Y_pred.fit")
    print("====================")
    col1 = fits.Column(name='ID', format='K', array=ID_test)
    col2 = fits.Column(name='Y_pred', format='F', array=Y_pred_reserved)
    col3 = fits.Column(name='Y_in', format='F', array=Y_val_reserved)
    col4 = fits.Column(name='Yo_in', format='F', array=Yo_val_reserved)
    col5 = fits.Column(name='Y_best', format='F', array=Y_best_reserved)

    cols = fits.ColDefs([col1, col2,col3, col4, col5])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(pathout+'/'+model_name+"_rserved.fit",overwrite='True')

    TP=np.where((Y_val >= 0.5) & (Y_pred >= 0.5))
    TN=np.where((Y_val < 0.5) & (Y_pred < 0.5))
    FP=np.where((Y_val < 0.5) & (Y_pred >= 0.5))
    FN=np.where((Y_val >= 0.5) & (Y_pred < 0.5))

    print("TP, TN, FP, FN")
    print(np.shape(TP), np.shape(TN), np.shape(FP), np.shape(FN))




 
##########################################
    


