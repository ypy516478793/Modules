import tensorflow as tf

def create_writer(save_dir, sess):
    # writer = tf.summary.FileWriter(logdir=save_dir, sess.graph)

    writer = tf.summary.FileWriter(logdir=save_dir)
    writer.add_graph(sess.graph)


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print()