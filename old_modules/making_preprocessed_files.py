import h5py
import glob
import numpy as np
import os
import pickle
from scipy.ndimage.interpolation import zoom
def resample(image, vs, new_spacing=[1,1,1]):
    # Determine current pixel spacing.
    spacing=np.array(vs[2], vs[0], vs[1], dtype=np.float32)

    resize_factor=spacing/new_spacing
    new_real_shape=image.shape* resize_factor
    new_shape=np.round(new_real_shape)
    real_resize_factor=new_shape/image.shape
    new_spacing=spacing/real_resize_factor

    image=zoom(image, real_resize_factor,mode='nearest')

    return image, new_spacing, new_shape


# with open('/home/cougarnet.uh.edu/Desktop/methodist_train.pkl','rb') as f:
#     data= pickle.load(f)
#     for filename in data:
#         h5f=h5py.File('/home/cougarnet.uh.edu/mpadmana/PycharmProjects/Faster_rcnn__pytorch/data/LungNoduledevkit/LungNodule/lung_nodule_detection/data_preparation/ct_lung_nodule/'+filename+'.h5','r'):
filename = "/Users/yuan_pengyu/Downloads/download20180607111951download20180607111900001_1_2_840_1.h5"
h5f = h5py.File(filename, "r")
im=h5f['image'][:]
old_shape=im.shape

# Shrink the image based on volume size.
vs= h5f['volume_size'][:]
resampled_im, new_spacing, new_shape= resample(im,vs,[1,1,1])

# Change the labels based on volume size.
labels=h5f['nodule_info'][:]
if len(labels) ==0:
    labels=np.array([[0,0,0,0]])
else:
    for label in labels:
        label[0]=label[0]*(new_shape[1]/old_shape[1])
        label[1]=label[1]*(new_shape[2]/old_shape[2])
        label[2] = label[2] * (new_shape[0] / old_shape[0])
        label[3] = label[3] * (new_shape[1] / old_shape[1])
        label=[label[2], label[1], label[0], label[3]]
# Normalizing the shrunk image.
maximum= np.amax(resampled_im)
minimum=np.amin(resampled_im)
resampled_im=((resampled_im-minimum)/(maximum-minimum))*255.0
### Not zero centering because the preprocessed luna is not ###############
resampled_im=resampled_im[np.newaxis,:,:,:]
        # np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_train_files/'+filename+'_clean', resampled_im)
        # np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_train_files/' + filename + '_labels', labels)

# with open('/home/cougarnet.uh.edu/Desktop/methodist_val.pkl','rb') as f:
#     data= pickle.load(f)
#     for filename in data:
#         h5f=h5py.File('/home/cougarnet.uh.edu/mpadmana/PycharmProjects/Faster_rcnn__pytorch/data/LungNoduledevkit/LungNodule/lung_nodule_detection/data_preparation/ct_lung_nodule/'+filename+'.h5','r'):
#         im=h5f['image'][:]
#         old_shape=im.shape
#
#         # Shrink the image based on volume size.
#         vs= h5f['volume_size'][:]
#         resampled_im, new_spacing, new_shape= resample(im,vs,[1,1,1])
#
#         # Change the labels based on volume size.
#         labels=h5f['nodule_info'][:]
#         if len(labels) ==0:
#             labels=np.array([[0,0,0,0]])
#         else:
#             for label in labels:
#                 label[0]=label[0]*(new_shape[1]/old_shape[1])
#                 label[1]=label[1]*(new_shape[2]/old_shape[2])
#                 label[2] = label[2] * (new_shape[0] / old_shape[0])
#                 label[3] = label[3] * (new_shape[1] / old_shape[1])
#                 label=[label[2], label[1], label[0], label[3]]
#         # Normalizing the shrunk image.
#         maximum= np.amax(resampled_im)
#         minimum=np.amin(resampled_im)
#         resampled_im=((resampled_im-minimum)/(maximum-minimum))*255.0
#         ### Not zero centering because the preprocessed luna is not ###############
#         resampled_im=resampled_im[np.newaxis,:,:,:]
#         np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_val_files/'+filename+'_clean', resampled_im)
#         np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_val_files/' + filename + '_labels', labels)
#
#
# with open('/home/cougarnet.uh.edu/Desktop/methodist_test.pkl','rb') as f:
#     data= pickle.load(f)
#     for filename in data:
#         h5f=h5py.File('/home/cougarnet.uh.edu/mpadmana/PycharmProjects/Faster_rcnn__pytorch/data/LungNoduledevkit/LungNodule/lung_nodule_detection/data_preparation/ct_lung_nodule/'+filename+'.h5','r'):
#         im=h5f['image'][:]
#         old_shape=im.shape
#
#         # Shrink the image based on volume size.
#         vs= h5f['volume_size'][:]
#         resampled_im, new_spacing, new_shape= resample(im,vs,[1,1,1])
#
#         # Change the labels based on volume size.
#         labels=h5f['nodule_info'][:]
#         if len(labels) ==0:
#             labels=np.array([[0,0,0,0]])
#         else:
#             for label in labels:
#                 label[0]=label[0]*(new_shape[1]/old_shape[1])
#                 label[1]=label[1]*(new_shape[2]/old_shape[2])
#                 label[2] = label[2] * (new_shape[0] / old_shape[0])
#                 label[3] = label[3] * (new_shape[1] / old_shape[1])
#                 label=[label[2], label[1], label[0], label[3]]
#         # Normalizing the shrunk image.
#         maximum= np.amax(resampled_im)
#         minimum=np.amin(resampled_im)
#         resampled_im=((resampled_im-minimum)/(maximum-minimum))*255.0
#         ### Not zero centering because the preprocessed luna is not ###############
#         resampled_im=resampled_im[np.newaxis,:,:,:]
#         np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_test_files/'+filename+'_clean', resampled_im)
#         np.save('/home/cougarnet.uh.edu/mpadmana/Desktop/methodist_test_files/' + filename + '_labels', labels)