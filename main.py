import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Neurocat import Loading
from Neurocat import im2col


# activation function
def activation(inp):
    return tf.nn.sigmoid(inp)


def gen_layer(name, input, shape, dev_w=0.5, dev_b=0.5, activate=True,
              res=None):
    with tf.name_scope(name) as scope:
        W = tf.Variable(
            tf.random_normal([shape[0], shape[1]],
                             stddev=dev_w, dtype=tf.float32), name="W")
        b = tf.Variable(
            tf.random_normal([1, shape[1]],
                             stddev=dev_b, dtype=tf.float32), name="b")

        out = tf.matmul(input, W) + b
        if activate:
            out = activation(out)

        # residual
        if res is not None:
            out += res

        tf.summary.histogram("weights", W)
        tf.summary.histogram("bias", b)

    return W, b, out


# num of training epochs
train_epoch = 2000

# horizont for timeembedding
pasHor = 100
futHor = 5
infHor = 1

# noisefector for training
noise_fac = 0.05

# number of mixtures
mixtures = 4

# number of units in layer
layer = [pasHor] \
        + [73] * 3 \
        + [42] * 3 \
        + [37] * 3 \
        + [mixtures * futHor * 3]

# standard deviation for weight initialisation
std_dev = 0.5

# num of samples that should be generated
gen_epoch = 2500

# choose the training data (zerocentered)
data = np.float32(np.loadtxt("./examples/sinus.txt")[None, :])

# timeembedding of the data
x_data = im2col(data[:, :-futHor], (1, pasHor), 1).T
y_data = im2col(data[:, pasHor:], (1, futHor), 1).T

# amount of training data
train_len = x_data.shape[0]

with tf.name_scope("x") as scope:
    x = tf.placeholder(dtype=tf.float32, shape=[None, pasHor], name="x")
with tf.name_scope("y") as scope:
    y = tf.placeholder(dtype=tf.float32, shape=[None, futHor], name="y")

# input layer
Wi, bi, out = gen_layer("input", x,
                        (layer[0], layer[1]), std_dev, std_dev)

# hidden layer
for i in range(1, len(layer) - 2):
    res = None
    if i % 3 == 1:
        residual = out
    if i % 3 == 2:
        res = residual
    Wh, bh, out = gen_layer("hidden", out,
                            (layer[i], layer[i + 1]), std_dev, std_dev,
                            res=res)

# output layer
Wo, bo, out = gen_layer("output", out,
                        (layer[-2], layer[-1]), std_dev, std_dev,
                        activate=False)
with tf.name_scope("mixture_model") as scope:
    pi, sigma, mu = tf.split(out, 3, 1, name="mixture")

new_shape = (-1, mixtures, futHor)

with tf.name_scope("Pi") as scope:
    pi = tf.reshape(pi, shape=new_shape, name="horizont")
    max_pi = tf.reduce_max(pi, 1, keep_dims=True)
    pi = tf.subtract(pi, max_pi)
    pi = tf.exp(pi)
    norm = tf.reduce_sum(pi, 1, keep_dims=True)
    norm = tf.reciprocal(norm)
    pi = tf.multiply(norm, pi)

with tf.name_scope("Sigma") as scope:
    sigma = tf.reshape(sigma, shape=new_shape, name="horizont")
    sigma = tf.exp(sigma)

with tf.name_scope("Mu") as scope:
    mu = tf.reshape(mu, shape=new_shape, name="horizont")

# normalisation factor for gaussian, not needed.#
norm_fac = 1. / np.sqrt(2. * np.pi)
# don't forget to normalize over mixtures
mix_norm = tf.constant(np.float32(1 / mixtures))

# make it feedable for tensorflow
gauss_norm = tf.constant(np.float32(norm_fac), name="Gaussnormalizer")

# calculate the loss, see Bishop 1994
with tf.name_scope("loss") as scope:
    lab = tf.reshape(y, shape=(-1, 1, futHor), name="horizont")
    with tf.name_scope("normal") as scope:
        normal = tf.subtract(lab, mu)
        normal = tf.multiply(normal, tf.reciprocal(sigma))
        normal = -tf.square(normal)
        normal = tf.multiply(normal, tf.reciprocal(tf.constant(2.)))
        normal = tf.multiply(tf.exp(normal), tf.reciprocal(sigma))
        normal = tf.multiply(normal, gauss_norm)
    with tf.name_scope("cond_average") as scope:
        lossfunc = tf.multiply(normal, pi)
        lossfunc = tf.reduce_sum(lossfunc, 1)
        lossfunc = tf.multiply(lossfunc, mix_norm)
    lossfunc = -tf.log(lossfunc / mixtures)
    lossfunc = tf.reduce_mean(lossfunc)

with tf.name_scope("trainer") as scope:
    train_op = tf.train.AdamOptimizer().minimize(lossfunc)

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    # toolkit for visual loading
    loader = Loading(train_epoch, 'Training the Network')
    # lossfunction
    loss = np.zeros(loader.treshold())
    for i in loader.range():
        noise = np.float32(np.random.normal(size=(train_len, 1)) * noise_fac)
        # training
        sess.run(train_op, feed_dict={
            x: x_data,
            y: np.add(y_data, noise)
        })
        # loss
        loss[i] = sess.run(lossfunc, feed_dict={
            x: x_data,
            y: y_data
        })

        # update visual loading
        loader.loading(i)

    # generate input
    x_gen = x_data[0][None, :]
    # generate prediction
    prediction = np.empty(shape=(1, 0))
    # get confidence and variance of the prediction
    confidence = np.empty(shape=(1, 0))
    variance = np.empty(shape=(1, 0))

    loader = Loading(gen_epoch, 'Generating Autonomous Wave')
    while loader.in_progress(prediction.shape[1]):
        _pi, _sig, _mu = sess.run(
            [pi, sigma, mu],
            feed_dict={x: x_gen})

        # find argmax of pi such that you can choose optimal mu
        argmax = np.argmax(_pi, axis=1)[0]
        __pi = np.choose(argmax, _pi[0])
        __mu = np.choose(argmax, _mu[0])
        __sig = np.choose(argmax, _sig[0])

        # append new information
        x_gen = np.concatenate((x_gen, __mu[None, :infHor]), axis=1)
        prediction = np.concatenate((prediction, __mu[None, :infHor]), axis=1)
        confidence = np.concatenate((confidence, __pi[None, :infHor]), axis=1)
        variance = np.concatenate((variance, __sig[None, :infHor]), axis=1)

        # delete old input that is not needed anymore
        x_gen = np.delete(x_gen, list(range(infHor)), axis=1)

        loader.loading(prediction.shape[1])
writer.close()

# prepare data, optional, you can create new with size of y_aut
data = data[:, pasHor:]
# little trick that subtracts numpy arrays of possibly different length
error = np.subtract(data[:, :prediction.shape[1]],
                    prediction[:, :data.shape[1]])

# plot
fig = plt.figure(1)
ax1 = fig.add_subplot(311)
ax1.plot(loss, 'r-')
ax1.set_title("errorrate MDN")
ax1.grid(True)
ax2 = fig.add_subplot(312)
ax2.plot(data.T, 'g-')
ax2.plot(prediction.T, 'b-')
ax2.plot(confidence.T, 'r--')
ax2.plot(variance.T, 'y--')
ax2.legend
ax2.set_title("Autonomous")
ax2.grid(True)
ax3 = fig.add_subplot(313, sharex=ax2)
ax3.plot(error.T, 'r-')
ax3.set_title("Difference")
ax3.grid(True)
plt.suptitle('mixtures: %d, '
             'layer: %d, '
             'training: %d, '
             'past horizont: %d, '
             'future horizont: %d, '
             'inference horizont: %d, '
             'noise %.2f' %
             (mixtures, len(layer), train_epoch, pasHor,
              futHor, infHor, noise_fac))
plt.show()
