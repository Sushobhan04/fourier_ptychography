from keras.models import load_model
import h5py
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import sys
from skimage.measure import compare_psnr
import cv2

def crop(set,N):
    h = set.shape[2]
    w = set.shape[3]

    return set[:,:,N:h-N,N:w-N]


def normalize(arr):
    ma = np.max(arr,axis=(2,3))
    mi = np.min(arr,axis=(2,3))
    ma = np.expand_dims(np.expand_dims(ma,axis=2),axis=3)
    mi = np.expand_dims(np.expand_dims(mi,axis=2),axis=3)

    arr = np.divide((arr - mi),(ma-mi))

    return arr

def BatchGenerator(files,batch_size,dtype = 'train', N=0):
    while 1:
        for file in files:
            curr_data = h5py.File(file,'r')
            data = np.array(curr_data[dtype]['data'][()])
            label = np.array(curr_data[dtype]['label'][()])
            # print data.shape, label.shape

            for i in range((data.shape[0]-1)//batch_size + 1):
                # print 'batch: '+ str(i)
                data_bat = data[i*batch_size:(i+1)*batch_size,]
                label_bat = label[i*batch_size:(i+1)*batch_size,]
                yield (data_bat, crop(label_bat,N))

def main():

    path_test = "/home/sushobhan/Documents/data/fourier_ptychography/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]

    dataset = []
    batch_size = 64
    N = 8

    dataset = h5py.File(path_test+'test40.h5','r')
    # data = normalize(np.expand_dims(dataset['data'],axis=0))
    data = np.array(dataset['data'])
    label = crop(np.array(dataset['label']),N)
    print label.shape, data.shape

    # test_generator = BatchGenerator(dataset, batch_size,dtype = 'test')
        
    model = load_model(path_test+'models/'+model_name+'.h5')
    y_output = model.predict(data)

    print y_output.shape

    

    # for i,x in enumerate(data[0]):
    #     print i
    #     print (x[0]*255)//1
    #     print np.max(x*255), np.mean(x*255)
    #     if i==24:
    #         print (x*255)//1
    #     cv2.imwrite('data/'+str(i)+'.png',(x*255)//1)

    data = crop(data[:,24:25],N)

    y_output = np.clip(y_output,0.0,1.0)
    data = np.clip(data,0.0,1.0)
    label = np.clip(label,0.0,1.0)

    y_output = y_output
    data = data
    label = label**2

    print y_output.shape, data.shape, label.shape

    psnr_input = []
    psnr_output = []

    for i in range(label.shape[0]):
        psnr_output.append(compare_psnr(label[i,0],y_output[i,0]))
        psnr_input.append(compare_psnr(label[i,0],data[i,0]))
        cv2.imwrite(str(i)+'_output.png',(y_output[i,0]*255)//1)
        cv2.imwrite(str(i)+'_data.png',(data[i,0]*255)//1)
        cv2.imwrite(str(i)+'_original.png',(label[i,0]*255)//1)

    print "psnr_output :"+str(psnr_output)
    print "psnr_input :"+str(psnr_input)

    print "outpur mean: " + str(np.mean(psnr_output))
    print "input mean: " + str(np.mean(psnr_input))

    




if __name__ == '__main__':
    main()
