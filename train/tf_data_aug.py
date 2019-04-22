import random
import numpy as np
import scipy.ndimage


def random_rotation_2d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    '''
    Add noise to the images in a batch
    :param batch: batch images of shape (batchSize, hight, width, numChannel)
    :param mean: mean value
    :param var: variance
    :param amount: percentage of pixels in one image
    :param mode: noise mode {'gaussian', 'pepper', 's&p'}
    :return:
    '''
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[tuple(coords)] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[tuple(coords)] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[tuple(coords)] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)

if __name__ == '__main__':
    import h5py
    fileName = "/home/cougarnet.uh.edu/pyuan2/Datasets/Lung_nodule_2d_32*32/Lung_Nodule_2d.h5"
    with h5py.File(fileName, 'r') as f:
        X_train = f['X_train'][:].astype(np.float32)
        Y_train = f['Y_train'][:].astype(np.int64)

    a = random_rotation_2d(X_train, 90)
    b = add_noise(X_train)
    x_train = np.concatenate((X_train, a, b), axis=0)
    y_train = np.concatenate((Y_train, Y_train, Y_train), axis=0)