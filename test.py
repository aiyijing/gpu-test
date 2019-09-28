import tensorflow as tf
import datetime

def cpu_test():

    # running
    # Creates a graph.(cpu version)

    print('cpu version')
    start = datetime.datetime.now()
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6, 9], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[9, 6], name='b')
        c = tf.matmul(a, b)
        c = tf.matmul(c, a)
        c = tf.matmul(c, b)
    # Creates a session with log_device_placement set to True.
    sess1 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    for i in range(100):
        sess1.run(c)
    print(sess1.run(c))
    sess1.close()
    end = datetime.datetime.now()

    return (end - start).microseconds


def gpu_test():

    # running
    # Creates a graph.(gpu version)

    print('gpuversion')
    start = datetime.datetime.now()
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[6, 9], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[9, 6], name='b')
        c = tf.matmul(a, b)
        c = tf.matmul(c, a)
        c = tf.matmul(c, b)
    # Creates a session with log_device_placement set to True.
    sess2 = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    for i in range(100):
        sess2.run(c)
    print(sess2.run(c))
    sess2.close()
    end = datetime.datetime.now()
    return (end - start).microseconds


if __name__ == "__main__":

    while True:
        time1 = cpu_test()
        time2 = gpu_test()
        print('CPU time:', time1)
        print('GPU time:', time2)
