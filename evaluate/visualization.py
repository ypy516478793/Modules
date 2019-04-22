
import matplotlib.pyplot as plt


def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('sample %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()


if __name__ == '__main__':

    import h5py

    fileName = "/home/cougarnet.uh.edu/pyuan2/Datasets/Lung_nodule_2d_32*32/Lung_Nodule_2d.h5"
    with h5py.File(fileName, 'r') as f:
        X_train = f['X_train'][:]

    sample_stack(X_train)