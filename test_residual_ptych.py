from keras.models import load_model
import h5py
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sys
from skimage.measure import compare_psnr

def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]

def main():

    # path_test = '/home/sushobhan/caffe/data/ptychography/databases/Test42_Set91_img512_patch48/test_images/'
    # path_test =  "/home/sushobhan/Documents/data/ptychography/Test42_Set91_img512_patch48/test_images/"
    # path_test = "/home/sushobhan/Documents/data/ptychography/"
    path_test = "/home/sushobhan/Documents/data/ptychography/Test40_Set91_img512_patch48/test_images/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]
    crop_size = 4

    border_mode = 'valid'

    # file_name = 'lena_1.h5'
    # file_name = 'resChart.h5'
    file_name = 'set_1.h5'

    file = h5py.File(path_test+ file_name,'r')
    ks = file.keys()
    print ks

    data = file['data']
    label =file['label']
    # data = np.max(data) - data
    # label = np.max(label) - label


    if file_name=="lena.h5" or file_name=="resChart.h5" or file_name=="lena_1.h5":

        data = np.expand_dims(file['data'], axis=0)
        label = np.expand_dims(np.expand_dims(file['label'], axis=0),axis=0)
    # label = np.transpose(label,(0,1,3,2))

    # im.imsave('label.png',label[0,0,],cmap=plt.cm.gray)
    # im.imsave('data.png',data[0,24,],cmap=plt.cm.gray)
        
    model = load_model(home+'models/'+model_name+'.h5')
    y_output = np.array(model.predict(data))

    if border_mode=='valid':
        data = crop(data,crop_size)
        label = crop(label,crop_size)

    print np.max(data), np.max(label)

    print y_output.shape, np.max(y_output)
    print data.shape , label.shape

    im.imsave('label.png',label[0,0,],cmap=plt.cm.gray)
    im.imsave('data.png',data[0,24,],cmap=plt.cm.gray)
    im.imsave('output.png',y_output[0,0,],cmap=plt.cm.gray)

    fig = plt.figure(0)
    m,n = 2,2
    for i in range(0,1):
        # print i
        j,k = i//n, i%n
        # print j,k
        plt.subplot2grid((m,n), (j, k))
        plt.imshow(label[i,0,],cmap=plt.cm.gray)
        # print j+2, k
        plt.subplot2grid((m,n), (j+1, k))
        plt.imshow(y_output[i,0,],cmap=plt.cm.gray)

        plt.subplot2grid((m,n), (j, k+1))
        plt.imshow(data[i,24,],cmap=plt.cm.gray)

        print compare_psnr(label[i,0,],y_output[i,0,])
        print compare_psnr(label[i,0,],data[i,24,])
    plt.subplot_tool()
    plt.savefig(model_name+'.jpg')

    psnr_center = []
    psnr_output = []

    for i in range(data.shape[0]):
        psnr_center.append(compare_psnr(label[i,0,],data[i,24,]))
        psnr_output.append(compare_psnr(label[i,0,],y_output[i,0,]))

    print np.mean(psnr_output)
    print np.mean(psnr_center)

    




if __name__ == '__main__':
    main()