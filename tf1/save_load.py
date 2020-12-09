import tensorflow as tf

def save_main():
    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

    inc_v1 = v1.assign(v1 + 1)
    dec_v2 = v2.assign(v2 - 1)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.
        inc_v1.op.run()
        dec_v2.op.run()
        # Save the variables to disk.
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

def load_main():
    tf.reset_default_graph()

    # Create some variables.
    v1 = tf.get_variable("v1", shape=[3])
    v2 = tf.get_variable("v2", shape=[5])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./tmp/model.ckpt")  # model_path = "./tmp/model.ckpt"
        print("Model restored.")
        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


def load(sess, save_dir, restore_training=True):
    saver = tf.train.Saver()
    if restore_training:
        ckpt = tf.train.get_checkpoint_state(save_dir)
        model_path = ckpt.model_checkpoint_path
        saver.restore(sess, model_path)
        print("Load model from: {:s}".format(model_path))
        # last_episode = int(str(ckpt.model_checkpoint_path).split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())