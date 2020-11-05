import numpy as np

A = np.random.randint(10, size=[3,2,4])
print("A before indexing: ")
print(A)

# expected output size: (3,1,2)
dim1_idx = np.repeat(np.arange(3), 2).reshape(3, 1, 2)
# array([[[0, 0]],
#        [[1, 1]],
#        [[2, 2]]])

dim2_idx = np.array([0, 0, 0, 1, 0, 0]).reshape(3, 1, 2)
# array([[[0, 0]],
#        [[0, 1]],
#        [[0, 0]]])

dim3_idx = np.array([0, 2, 2, 0, 0, 1]).reshape(3, 1, 2)
# array([[[0, 2]],
#        [[2, 0]],
#        [[0, 1]]])

print("A after indexing: ")
print(A[dim1_idx, dim2_idx, dim3_idx])

# h, w, d = batch_x.shape[0], num_classes*FLAGS.update_batch_size, batch_x.shape[2]
# task_idx = np.repeat(np.arange(h), w*d).reshape(h,w,d)
# a_idx = np.zeros([h, w, d]).astype(np.int)
# for i in range(batch_x.shape[0]):
#     a_idx[i] = np.random.choice(batch_x.shape[1], [w, d], replace=False)
# feature_idx = np.tile(np.arange(d), h*w).reshape(h,w,d)
# inputa = batch_x[task_idx, a_idx, feature_idx]
# labela = batch_y[task_idx, a_idx, feature_idx]
# inputb = batch_x
# labelb = batch_y

# in main.py of maml project