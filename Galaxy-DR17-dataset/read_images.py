#############################################################

def extract_thumb(im,x,y,size):
    if size %2==1:
        size = size-1
    #print(size)
    #print(x)
    #print(im.shape)
    up_x=int(x-size/2)
    dow_x=int(x+size/2)
    up_y=int(y-size/2)
    dow_y=int(y+size/2)
    res=im[up_x:dow_x,up_y:dow_y]        
    return res



#############################################################

def read_data(pathin,maxim):
    size_im=69 #64
    size_crop=207 #192

    # Read catalog
    data=fits.getdata(pathin+'NairAbraham_SDSS_casJobs_all.fit',1)
    idcat=data['objid']
    Ttype=data['TT']
    Tq=data['TTq']
    
    # Quiality samples
    ok=np.where(Tq == 0) 
    id_ok=idcat[ok]
    tt=Ttype[ok]
    tq_ok=Tq[ok]
    

    nparams=1 
    D=np.zeros([maxim,size_im,size_im, 3]) #channels last
    Y=np.zeros([maxim,nparams])
    idvec=np.zeros([maxim], dtype=np.long)

    pdb.set_trace() 
    
    iteri=0
    
    for numgal in id_ok[0:maxim]:

        try:          
           
                path=pathin+'jpgs_original/' 
                namegal=namegal=str(numgal)+".jpg"                    
                scidata = cv.imread(path+namegal) #424,424
                
                print('Reading: ', namegal)
                lx,ly, nbands=scidata.shape
                print("LX", lx)
                print("LY", ly)

                if lx != ly:
                    print("Not squared image")
                    pdb.set_trace()
            
                scidata = extract_thumb(scidata,int(lx/2.0),int(ly/2.0),size_crop)
                scidata=zoom(scidata, [1/3.,1./3,1], order=3)

                print( "Final image size ",scidata.shape)
                #scidata = np.transpose(scidata) we want channels last    
   
                    
                tmp=(id_ok == numgal)                
                print("Iter ID TType: ", iteri, id_ok[tmp][0], tt[tmp][0])
                
                D[iteri, :,:, :]=scidata
                idvec[iteri]=str(numgal)                 
                Y[iteri]=tt[tmp][0]
                Y = Y.squeeze()

                #pdb.set_trace()

                #==== save example =======
                #if iteri%10==0:
                    #print("Saving example")
                    #misc.imsave(pathin+"examples_jpg_69x69/"+namegal,np.transpose(scidata))   

                #only change index if saved value                    
                iteri=iteri+1  

        except:
            
            print('Bad galaxy ', numgal)
            

    pdb.set_trace() 

    print("Saving image and target vector")
    path_input='/home/ubuntu/data/helena/MPL-9/'
    np.save(path_input+"image_vector_"+str(maxim)+".npy",D)
    np.save(path_input+"target_vector_"+str(maxim)+".npy",Y)
    np.save(path_input+"ID_vector_"+str(maxim)+".npy",idvec) 

    
#############################################################
        #CODE STARTS HERE #
#############################################################

#source activate tensorflow_p27

print("Importing packages")

from astropy.io import fits
import os
import numpy as np
from scipy.ndimage import zoom
import pdb


import cv2 as cv
from matplotlib.pyplot import imread

#misc deprecated, use above
#from scipy.misc import imresize
#import scipy.misc


#==============================

READ_IMAGES=True


pathin="/home/ubuntu/data/helena/MPL-9/"
maxim=13454      #number of images read in D, Y
#maxim=100
nparams=1


## Read image vector
if READ_IMAGES:
    print("Reading images")
    read_data(pathin,maxim)  #read images

pdb.set_trace() 

## Loading image vector
print("Loading D, Y")
path_input='/home/ubuntu/data/MPL-9/'
D=np.load(path_input+"image_vector_"+str(maxim)+".npy")
Y=np.load(path_input+"target_vector_"+str(maxim)+".npy")
id_train=np.load(path_input+"ID_vector_"+str(maxim)+".npy")

            

