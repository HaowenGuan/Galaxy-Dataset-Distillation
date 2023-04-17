import os
# import urllib2 #this need of python 2, otherwise
from urllib.request import urlopen

import numpy as np
# import pylab as pl
# import numpy as numpy
# import pdb
# from matplotlib import image
from astropy.io import fits
# from astropy.table import Table
from multiprocessing import Pool, cpu_count, get_context
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def _fetch(outfile, RA, DEC, scale, width=424, height=424):
    """Fetch the image at the given RA, DEC from the SDSS server"""
    url = ("http://casjobs.sdss.org/ImgCutoutDR7/""getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.8f&width=%i&height=%i" % (
        RA, DEC, scale, width, height))

    print("downloading%s" % url)
    print(" ->%s" % outfile)
    print("scale", scale)

    fhandle = urlopen(url)
    open(outfile, 'wb').write(fhandle.read())

    # fhandle = urllib2.urlopen(url)
    # open(outfile, 'w').write(fhandle.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--download', action='store_true', default=False, help='download-dataset')
    parser.add_argument('--dataset', type=str, default='gzoo2', help='dataset')

    args = parser.parse_args()

    if args.dataset == "gzoo2":
        # gzoo2 here -----------------------------------------
        # if not os.path.exists('gzoo2/image'):
        #     os.makedirs('gzoo2/image')

        gzoo = fits.open(os.path.join('gzoo2', 'zoo2MainSpecz_sizes.fit'))[1].data
        # gzoo = fits.open(os.path.join('gzoo2', 'zoo2MainSpecz.fits'))[1].data
        ID = gzoo['dr7objid']
        ra = gzoo['ra']
        dec = gzoo['dec']
        R_90 = gzoo['petroR90_r']
        gzoo_dict = dict()
        for i, id in enumerate(ID):
            gzoo_dict[id] = [ra[i], dec[i], R_90[i]]

        class_9_path = "/data/sbcaesar/classes/6000/9/"

        #
        # print(len(images))

        for image in os.listdir(class_9_path):
            if ".jpg" not in image:
                continue
            id = int(image[:-4])
            image_dir = os.path.join(class_9_path, image)
            print(image_dir)
            im = Image.open(image_dir)
            new_im = np.transpose(im)
        print("Number of images to be read", ID.shape[0])
        j = gzoo_dict[588848899389522032]
        print(j)
        _fetch(class_9_path + "588848899389522032.jpg", j[0], j[1], j[2] * 0.15)
        image_dir = class_9_path + "588848899389522032.jpg"
        im = Image.open(image_dir)
        plt.imshow(im)
        plt.show()
        # if args.download:
        #     with Pool(max(cpu_count(), 8)) as pool:
        #         args = []
        #         for j in range(50000, 100000):#ID.shape[0]
        #             args.append(['gzoo2/image/' + str(ID[j]) + ".jpg", ra[j], dec[j], R_90[j] * 0.024])
        #         results = pool.starmap(_fetch, args)


    elif args.dataset == "dl-DR17":
        data = []
        ttype = dict()
        with open(os.path.join('gzoo2', 'table2.dat')) as file:
            for line in file:
                cur = line.rstrip().split()
                if len(cur) == 60:
                    cur.pop(-2)
                data.append(cur)
                ttype[cur[0]] = cur[-11]
        
        if not os.path.exists('MaNGA/image'):
            os.makedirs('MaNGA/image')

        cat = fits.getdata(os.path.join('MaNGA', 'MaNGA_targets_extNSA_tiled_ancillary.fits'))
        ID_NSA = cat['MANGAID']
        R_NSA = cat['NSA_SERSIC_TH50']

        data = fits.getdata(os.path.join('MaNGA', 'manga-pymorph-DR17.fits'))
        dl17 = fits.open(os.path.join('MaNGA', 'manga-morphology-dl-DR17.fits'))[1].data
        # Reference: https://academic.oup.com/mnras/article/483/2/2057/5188692
        ID_Manga = data['MANGA_ID']  # MaNGA identification
        ID = data['INTID']  # Internal identification number
        ra = data['RA']  # Object right ascension (deg)
        dec = data['DEC']  # Object declination (deg)
        R_S = data['A_HL_S']  # Half-light semimajor axis (arcsec) from Sérsic fit
        tt = (dl17["T-Type"] + 0.5).astype(int)

        print("Number of images to be read", ID.shape[0])

        if args.download:
            print("Multi-thread processing with", cpu_count(), "core(s).")
            if not os.path.exists('MaNGA/image/excluded_from_NSA/'):
                os.makedirs('MaNGA/image/excluded_from_NSA/')

            with Pool(max(cpu_count(), 8)) as pool:
                args = []

                for j in range(0, ID.shape[0]):
                    k = np.where(ID_NSA == ID_Manga[j])  # find galaxy in NSA

                    if np.shape(k) == (1, 1):  # galaxy exists in NSA
                        args.append(['MaNGA/image/' + str(ID[j]) + ".jpg", ra[j], dec[j], R_NSA[k] * 0.024])
                    else:
                        args.append(['MaNGA/image/excluded_from_NSA/' + str(ID[j]) + ".jpg", ra[j], dec[j], R_S[j] * 0.024])

                results = pool.starmap(_fetch, args)





    exit(0)

    # ----------------------------------------------
    data = "Nair"

    if data == 'GZOO':

        dir = '/lhome/ext/ice043/ice0431/helena/MPL-9/GZOO/'

        data = fits.getdata(dir + 'zoo2MainSpecz_sizes.fit', 1)

        ID = data['dr7objid']
        ra = data['ra']
        dec = data['dec']
        R_90 = data['petroR90_r']
        R = R_90

        print("Number of images to be read", ID.shape[0])
        # pdb.set_trace()

        for j in range(20236, ID.shape[0]):
            # for j in range(0,138966):

            print("Number galaxy", j, ID[j])
            print("Radii", R[j], R_90[j])
            file_name = dir + '/jpgs/' + str(ID[j]) + ".jpg"

            if not os.path.exists(file_name):
                _fetch(file_name, ra[j], dec[j], scale=R_90[j] * 0.024)

    # ================================================

    if data == 'MaNGA':

        dir = '/lhome/ext/ice043/ice0431/helena/MPL-9'

        cat = fits.getdata(dir + '/MaNGA_targets_extNSA_tiled_ancillary.fits')
        ID_NSA = cat['MANGAID']
        R_NSA = cat['NSA_SERSIC_TH50']

        data = fits.getdata(dir + '/pymorph_DR16.fits', 1)
        ID_Manga = data['MANGA_ID']
        ID = data['INTID']
        ra = data['RA']
        dec = data['DEC']
        R_S = data['A_hl_S']

        print("Number of images to be read", ID.shape[0])

        pdb.set_trace()

        for j in range(0, ID.shape[0]):
            # for j in range(3,6):

            k = np.where(ID_NSA == ID_Manga[j])  # find galaxy in NSA

            if np.shape(k) == (1, 1):  # galaxy exists in NSA

                print("Number galaxy", j, ID[j], ID_Manga[j], ID_NSA[k])
                print("Radii", R_S[j], R_NSA[k])

                # _fetch(dir+'/jpgs_original/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R_NSA[k]*0.024)
                _fetch('/lhome/ext/ice043/ice0431/helena/MPL-11/jpg_tests/' + str(ID[j]) + ".jpg", ra[j], dec[j],
                       scale=R_NSA[k] * 0.024)

    # ================================================

    if data == 'Nair':

        dir = '/lhome/ext/ice043/ice0431/helena/MPL-9/'

        data = fits.getdata(dir + '/Download_sdss_fits/NairAbraham_SDSS_casJobs_all.fit', 1)

        ID_Manga = data['galcount']
        ID = data['objid']
        ra = data['_RA']
        dec = data['dec']
        R_90 = data['petror90_r']
        R = data['petrorad_r']

        print("Number of images to be read", ID.shape[0])
        pdb.set_trace()

        # for j in range(0,ID.shape[0]):
        for j in range(12117, 12118):
            print("Number galaxy", j, ID[j], ID_Manga[j])
            print("Radii", R[j], R_90[j])

            # _fetch(dir+'/jpgs_original/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R_90[j]*0.024)
            _fetch('/lhome/ext/ice043/ice0431/helena/MPL-11/jpg_tests/' + str(ID[j]) + ".jpg", ra[j], dec[j],
                   scale=R_90[j] * 0.024)

    # galcount, ra , dec, redshift, R90=np.loadtxt('/home/helena/DR7/Petro_R90_all.txt', unpack=True)
    # dir='/Users/helena/Desktop/MANGA/MPL-9'

    # _fetch('/home/helena/DR7/cutouts/'+str(ID[j])+".jpg",ra[j],dec[j],scale=R90[j]*0.024)
