from __future__ import print_function
import os
#import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
learning_rate = 0.001
training_iters = 3000
batch_size = 10
display_step = 3

# Network Parameters
n_input = 128*128 # MNIST data input (img shape: 128*128 )
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, 128, 128, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture  x.shape:(128,128,3)
    x = tf.reshape(x, shape=[-1, 128, 128, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv2, k=2)
    print(pool1.shape) #(64,64,64)

    # Convolution Layer
    conv3 = conv2d(pool1, weights['wc3'], biases['bc3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv4, k=2)
    print(pool2.shape) #(32,32,128)

    # Convolution Layer
    conv5 = conv2d(pool2, weights['wc5'], biases['bc5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7'])
    # Max Pooling
    pool3 = maxpool2d(conv7, k=2)
    print(pool3.shape) #(16,16,256)

    # Convolution Layer
    conv8 = conv2d(pool3, weights['wc8'], biases['bc8'])
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
    conv10 = conv2d(conv9, weights['wc10'], biases['bc10'])
    # Max Pooling
    pool4 = maxpool2d(conv10, k=2)
    print(pool4.shape) #(8,8,512)

    conv11 = conv2d(pool4, weights['wc11'], biases['bc11'])
    conv12 = conv2d(conv11, weights['wc12'], biases['bc12'])
    conv13 = conv2d(conv12, weights['wc13'], biases['bc13'])
    # Max Pooling
    pool5 = maxpool2d(conv13, k=2)
    print(pool5.shape) #(4,4,512)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    #fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    '''
    fc3 = tf.reshape(fc2, [-1, weights['out'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc2, weights['out']), biases['bd2'])
    fc3 = tf.nn.relu(fc2)
    '''
    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 3 input, 24 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),

    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),

    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc6': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    'wc7': tf.Variable(tf.random_normal([3, 3, 128, 128])),

    'wc8': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc9': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc10': tf.Variable(tf.random_normal([3, 3, 256, 256])),

    'wc11': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4*4*512, 512])),
    'wd2': tf.Variable(tf.random_normal([512, 64])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([64, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([128])),
    'bc6': tf.Variable(tf.random_normal([128])),
    'bc7': tf.Variable(tf.random_normal([128])),
    'bc8': tf.Variable(tf.random_normal([256])),
    'bc9': tf.Variable(tf.random_normal([256])),
    'bc10': tf.Variable(tf.random_normal([256])),
    'bc11': tf.Variable(tf.random_normal([512])),
    'bc12': tf.Variable(tf.random_normal([512])),
    'bc13': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'bd2': tf.Variable(tf.random_normal([64])),
    'out': tf.Variable(tf.random_normal([10]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    list = os.listdir("./resize_image/")
    print(list)
    print(len(list))
    count=0
    while count < 50:
        count = count + 1
        #print('count:',count)
        batch_xs = []
        batch_ys = []
        for image in list:
            id_tag = image.find("-")
            score = image[0:id_tag]
            # print(score)
            img = Image.open("./resize_image/" + image)
            img_ndarray = numpy.asarray(img, dtype='float32')
            img_ndarray = numpy.reshape(img_ndarray, [128, 128, 3])
            # print(img_ndarray.shape)
            batch_x = img_ndarray
            batch_xs.append(batch_x)
            # print(batch_xs)
            batch_y = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            # print(type(score))
            batch_y[int(score) - 1] = 1
            # print(batch_y)
            batch_y = numpy.reshape(batch_y, [10, ])
            batch_ys.append(batch_y)
        # print(batch_ys)
        batch_xs = numpy.asarray(batch_xs)
        #print(batch_xs.shape)
        batch_ys = numpy.asarray(batch_ys)
        #print(batch_ys.shape)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       keep_prob: dropout})
        if count % 1 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs,
                                                              y: batch_ys,
                                                              keep_prob: 1.})
            print("count = " + str(count) + ", Minibatch Loss= " + str(loss) + ", Training Accuracy= " + str(acc) )
    print("Optimization Finished!")
    saver.save(sess,"./model-vgg/model.ckpt")

'''
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    list = os.listdir("./resize_image/")
    print(list)
    print(len(list))
    count=0
    while count<200:
        count = count+1
        #print("count:",count)
        #batch_xs = []
        #batch_ys = []
        for batch_id in range(0, 10):
            batch = list[batch_id * 50:batch_id * 50 + 50]
            batch_xs = []
            batch_ys = []
            for image in batch:
                id_tag = image.find("-")
                score = int(image[0:id_tag])
                score = round(score)
                # print(score)
                img = Image.open("./resize_image/" + image)
                img_ndarray = numpy.asarray(img, dtype='float32')
                img_ndarray = numpy.reshape(img_ndarray, [128, 128, 3])
                # print(img_ndarray.shape)
                batch_x = img_ndarray
                batch_xs.append(batch_x)
                # print(batch_xs)
                batch_y = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                # print(type(score))
                batch_y[int(score) - 1] = 1
                # print(batch_y)
                batch_y = numpy.reshape(batch_y, [10, ])
                batch_ys.append(batch_y)
                #print(batch_ys)
            batch_xs = numpy.asarray(batch_xs)
            #print(batch_xs.shape)
            batch_ys = numpy.asarray(batch_ys)
            #print(batch_ys.shape)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                           keep_prob: dropout})
        if count % 5 == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_xs,
                                                                  y: batch_ys,
                                                                  keep_prob: 1.})
                print("Count: " + str(count) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss) + ", Training Accuracy= " + \
                        "{:.5f}".format(acc))

    print("Optimization Finished!")
    saver.save(sess,"./model-vgg/model.ckpt")
'''