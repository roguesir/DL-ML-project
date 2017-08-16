
from __future__ import print_function
import os
import matplotlib.pyplot as plt
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
display_step = 2

# Network Parameters
n_input = 128*128 # MNIST data input (img shape: 28*28)
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
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1, k=2)
    print(pool1.shape) #(64,64,64)

    # Convolution Layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2, k=2)
    print(pool2.shape) #(32,32,128)

    # Convolution Layer
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    # Max Pooling
    pool5 = maxpool2d(conv5, k=2)
    print(pool5.shape) #(16,16,256)

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
    #out = tf.nn.softmax(out)
    return out

# Store layers weight & bias
weights = {
    # 3x3 conv, 3 input, 24 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 16])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 96])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 96, 128])),
    # fully connected, 32*32*96 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*128, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([4096, 10]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([96])),
    'bc5': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([10]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
pred_result = tf.argmax(pred, 1)
# print(pred)
#pred = tf.argmax(pred, 1)
# Define loss and optimizer
#cost = tf.reduce_mean((tf.argmax(pred, 1)-tf.argmax(y, 1))^2)
cost = -tf.reduce_sum(y *tf.log(tf.clip_by_value(pred ,1e-10,1.0)))
#cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver=tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, "./model-alex/model.ckpt")
    step = 1
    # Keep training until reach max iterations
    list = os.listdir("./test_resize/")
    # print(list)
    print(len(list))
    input_xs = []
    for image in list:
        print(image)
        img = Image.open("./test_resize/"+image)
        img_ndarray = numpy.asarray(img,dtype='float32')
        img_ndarray = numpy.reshape(img_ndarray,[128,128,3])
        # print(type(img_ndarray.tolist()))
        input_xs.append(img_ndarray)
    input_xs = numpy.asarray(input_xs)
    prediction_test = sess.run(pred_result,feed_dict={x:input_xs,keep_prob:1})
    print(prediction_test+1)

    print('Test Finished')
    '''
    for batch_id in range(0, 2):
        batch = list[batch_id * 10:batch_id * 10 + 10]
        batch_xs = []
        batch_ys = []
        for image in batch:
            id_tag = image.find("-")
            score = image[0:id_tag]
            # print(score)
            img = Image.open("./test_resize/" + image)
            img_ndarray = numpy.asarray(img, dtype='float32')
            img_ndarray = numpy.reshape(img_ndarray, [128, 128, 3])
            # print(img_ndarray.shape)
            batch_x = img_ndarray
            batch_xs.append(batch_x)
        print(batch_xs)
        # print(batch_ys)
        batch_xs = numpy.asarray(batch_xs)
        #print(batch_xs.shape)

        # Run optimization op (backprop)
        pred_result_test=sess.run(pred_result, feed_dict={x: batch_xs,keep_prob: 1.})
        print(pred_result_test)
    print("Test Finished!")
    '''