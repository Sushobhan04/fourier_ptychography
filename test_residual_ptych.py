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

def BatchGenerator(files,batch_size,dtype = 'train'):
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
                yield (data_bat, label_bat)

def main():

    path_test = "/home/sushobhan/Documents/data/fourier_ptychography/"
    home = "/home/sushobhan/Documents/research/ptychography/"
    model_name = sys.argv[1]

    dataset = []
    batch_size = 64

    for i in range(1,6):
        dataset.append(path_test+'datasets/pcp_ptych/set_'+str(i)+'.h5')

    test_generator = BatchGenerator(dataset, batch_size,dtype = 'test')
        
    model = load_model(path_test+'models/'+model_name+'.h5')
    y_output = model.predict_generator(test_generator, steps=13, max_q_size=100,verbose=1)
    evalulate = model.evaluate_generator(test_generator, steps=13, max_q_size=100)

    print evalulate

    print y_output.shape

    # im.imsave('label.png',label[0,0,],cmap=plt.cm.gray)
    # im.imsave('data.png',data[0,24,],cmap=plt.cm.gray)
    # im.imsave('output.png',y_output[0,0,],cmap=plt.cm.gray)

    # fig = plt.figure(0)
    # m,n = 2,2
    # for i in range(0,1):
    #     # print i
    #     j,k = i//n, i%n
    #     # print j,k
    #     plt.subplot2grid((m,n), (j, k))
    #     plt.imshow(label[i,0,],cmap=plt.cm.gray)
    #     # print j+2, k
    #     plt.subplot2grid((m,n), (j+1, k))
    #     plt.imshow(y_output[i,0,],cmap=plt.cm.gray)

    #     plt.subplot2grid((m,n), (j, k+1))
    #     # plt.imshow(data[i,24,],cmap=plt.cm.gray)

    #     print compare_psnr(label[i,0,],y_output[i,0,])
    #     print compare_psnr(label[i,0,],data[i,24,])
    # plt.subplot_tool()
    # plt.savefig(model_name+'.jpg')

    # psnr_center = []
    # psnr_output = []
    j=0

    for x in test_generator:
        # psnr_center.append(compare_psnr(label[i,0,],data[i,24,]))
        for i in range(batch_size):
            psnr_output.append(compare_psnr(x[i,0,],y_output[j,0,]))
            j+=1

    print psnr_output
    print np.mean(psnr_center)

    




if __name__ == '__main__':
    main()