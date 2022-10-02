
import os
#import urllib2 #this need of python 2, otherwise
from urllib.request import urlopen
import pylab as pl
import numpy as np
import pdb
from matplotlib import image
from astropy.io import fits
from astropy.table import Table

####################################################

def _fetch(outfile, RA, DEC, scale, width=424, height=424):

    """Fetch the image at the given RA, DEC from the SDSS server"""
    url = ("http://casjobs.sdss.org/ImgCutoutDR7/""getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.2f&width=%i&height=%i"% (RA, DEC, scale, width, height))

    #pdb.set_trace()
    
    print("downloading%s" % url)
    print(" ->%s" % outfile)
    print("scale", scale)

    pdb.set_trace()

    fhandle = urlopen(url)
    open(outfile, 'wb').write(fhandle.read())

    
    #fhandle = urllib2.urlopen(url)
    #open(outfile, 'w').write(fhandle.read())

    #pdb.set_trace()



####################################################

data="Nair"

if data=='GZOO':
    
    dir='/lhome/ext/ice043/ice0431/helena/MPL-9/GZOO/'

    data=fits.getdata(dir+'zoo2MainSpecz_sizes.fit',1)


    
    ID=data['dr7objid']
    ra=data['ra']
    dec=data['dec']
    R_90=data['petroR90_r']
    R=R_90


    print("Number of images to be read", ID.shape[0])
    #pdb.set_trace()
    
    for j in range(20236,ID.shape[0]):
    #for j in range(0,138966):



        
        print("Number galaxy", j, ID[j])
        print("Radii",  R[j], R_90[j])
        file_name=dir+'/jpgs/'+str(ID[j])+".jpg"

        if not os.path.exists(file_name):

            _fetch(file_name,ra[j],dec[j],scale=R_90[j]*0.024)


#================================================
        
if data=='MaNGA':
    
    dir='/lhome/ext/ice043/ice0431/helena/MPL-9'

    cat=fits.getdata(dir+'/MaNGA_targets_extNSA_tiled_ancillary.fits')
    ID_NSA=cat['MANGAID']
    R_NSA=cat['NSA_SERSIC_TH50']

    data=fits.getdata(dir+'/pymorph_DR16.fits',1)
    ID_Manga=data['MANGA_ID']
    ID=data['INTID']
    ra=data['RA']
    dec=data['DEC']
    R_S=data['A_HL_S']


    print("Number of images to be read", ID.shape[0])

    pdb.set_trace()
     

    for j in range(0,ID.shape[0]):
    #for j in range(3,6):
        
        k=np.where(ID_NSA == ID_Manga[j])  #find galaxy in NSA

        if np.shape(k) == (1,1): #galaxy exists in NSA
        
            print("Number galaxy", j, ID[j], ID_Manga[j], ID_NSA[k])
            print("Radii",  R_S[j], R_NSA[k])

            #_fetch(dir+'/jpgs_original/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R_NSA[k]*0.024)
            _fetch('/lhome/ext/ice043/ice0431/helena/MPL-11/jpg_tests/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R_NSA[k]*0.024)

#================================================
            
if data=='Nair':
    
    dir='/lhome/ext/ice043/ice0431/helena/MPL-9/'

    data=fits.getdata(dir+'/Download_sdss_fits/NairAbraham_SDSS_casJobs_all.fit', 1)


    ID_Manga=data['galcount']
    ID=data['objid']
    ra=data['_RA']
    dec=data['dec']
    R_90=data['petror90_r']
    R=data['petrorad_r']



    print("Number of images to be read", ID.shape[0])
    pdb.set_trace()
     
    #for j in range(0,ID.shape[0]):
    for j in range(12117, 12118):
        

        
        print("Number galaxy", j, ID[j], ID_Manga[j])
        print("Radii",  R[j], R_90[j])

        #_fetch(dir+'/jpgs_original/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R_90[j]*0.024)
        _fetch('/lhome/ext/ice043/ice0431/helena/MPL-11/jpg_tests/' + str(ID[j]) + ".jpg", ra[j], dec[j],
               scale=R_90[j] * 0.024)



#galcount, ra , dec, redshift, R90=np.loadtxt('/home/helena/DR7/Petro_R90_all.txt', unpack=True)
#dir='/Users/helena/Desktop/MANGA/MPL-9'
    
#_fetch('/home/helena/DR7/cutouts/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R90[j]*0.024)
    
    

