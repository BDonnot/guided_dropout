import os
import copy
import re

import numpy as np
import tensorflow as tf

import pdb

DTYPE_USED = tf.float32 # default tensorflow type used for variable


class EncodingRaw:
    def __init__(self, num_elem, nullElemenEnc):
        """Encode one dimension only, for onehot only
        Usage outside of "SpecificGDCEncoding" is not recommended.
        :param num_elem: the number of element in the variable encoded
        :param nullElemenEnc: the encoding of the null element
        """
        self.enco = {}
        self.nullElemenEnc = nullElemenEnc
        self.size = nullElemenEnc.shape
        self.num_elem = num_elem
        self.nullelemenArr = np.zeros(num_elem, dtype=np.int)
        self.nullElemenkey = self.nullelemenArr.tostring()
        self.enco[self.nullElemenkey] = copy.deepcopy(nullElemenEnc)
        self.num = len(self.nullElemenkey)

    def __getitem__(self, item):
        """
        Return the mask associated with the item given in input
        :param item: the item to be queried
        :return: the proper mask
        """
        if not item in self.enco.keys():
            if len(item) != self.num:
                raise RuntimeError(
                    "EncodingRaw.__getitem__: unsuported querying for object of size {} (normal size should be {})".format(
                        len(item), self.num))
            # tmp = self.nullelemenArr
            res = self.nullElemenEnc
            arr_tmp = np.fromstring(item, dtype=np.int)
            for id_el, el in enumerate(arr_tmp):
                if el != 0:
                    tmp_ = copy.copy(self.nullelemenArr)
                    tmp_[id_el] = el
                    tmp_ = self.__getitem__(tmp_.tostring())
                    res = np.logical_or(res, tmp_)
            self.enco[item] = res.reshape(self.size).astype(np.float32)
        return self.enco[item]

    def __setitem__(self, key, value):
        self.enco[key] = value

    def keys(self):
        return self.enco.keys()

    def save(self, path, name):
        """
        save the masks for every key, for a later ussage
        :param path: the path where the masks should be stored
        :param name: the name used to save (all mask will be stored in path/name/*.npy)
        :return: 
        """
        mypath = os.path.join(path, name)
        if not os.path.exists(mypath):
            os.mkdir(os.path.join(mypath))
        for key, mask in self.enco.items():
            arr = np.fromstring(key, dtype=np.int)
            nm = []
            for i, v in enumerate(arr):
                if v != 0:
                    nm.append("{}".format(i))
            if len(nm) == 0:
                nm = "bc"
            else:
                nm = "_".join(nm)
            nm += ".npy"
            np.save(file=os.path.join(mypath, nm), arr=mask)

    def reload(self, path, name):
        """
        reload the mask for every keys
        :param path: the path from where the data should be restored
        :param name: 
        :return: 
        """
        # pdb.set_trace()
        path = re.sub("\"", "", path)
        name = re.sub("\"", "", name)
        mypath = os.path.join(path, name)
        if not os.path.exists(mypath):
            msg = "E: EncodingRaw.reload: the directory {} doest not exists".format(mypath)
            raise RuntimeError(msg)
        del self.enco
        self.enco = {}
        # self.nullelemenArr = np.zeros(num_elem, dtype=np.int)
        # self.nullElemenkey = self.nullelemenArr.tostring()

        if not os.path.exists(os.path.join(mypath, "bc.npy")):
            msg = "E: EncodingRaw.reload: the file in bc.npy is nto located at {}, but it should".format(mypath)
            raise RuntimeError(msg)

        arr = np.load(os.path.join(mypath, "bc.npy"))
        self.enco[self.nullElemenkey] = copy.deepcopy(arr)
        self.nullElemenEnc = copy.deepcopy(arr)
        for fn in os.listdir(mypath):
            if not self._isfnOK(fn):
                continue
            key = self._extractkey(fn)
            self.enco[key] = np.load(os.path.join(mypath, fn))

    def _isfnOK(self, fn):
        """
        return true if the file is a 'mask' file
        :param fn: 
        :return: 
        """
        return re.match("([0-9]+(\_[0-9]+)*)\.npy", fn) is not None

    def _extractkey(self, fn):
        """
        Extract the key to the stored array
        :param fn: 
        :return: 
        """
        fn = re.sub("\.npy", "", fn)
        fns = fn.split("_")
        arr = np.zeros(self.num_elem, dtype=np.int)
        for el in fns:
            arr[int(el)] = 1
        # print("fn: {}\n key: {}".format(fn, arr))
        return arr.tostring()


class SpecificGDCEncoding:
    def __init__(self, sizeinputonehot, nrow, ncol,
                 name="guided_dropconnect_encoding",
                 nbconnections=None,
                 path=".", reload=False,
                 keep_prob=None):
        """
        Encoding for guided dropconnect, a more generic case of guided dropout, but that we don't recommend using.
        :param nrow: number of rows of the matrix
        :param ncol: number of columns of the matrix to be masked
        :param sizeinputonehot: size of the tau vector
        :param name: name of the operation
        :param nbconnections: number of connections used for each dimension of the one-hot input vector
        :param path: path where mask will be stored
        :param reaload: do you want to reload (T) or build (F) the mask
        :param keep_prob: the keeping probability for regular dropout (applied only in the units not "guided dropout'ed")
        """
        self.size_in = nrow
        self.size_out = ncol

        self.nb_connections = nbconnections # number of connections per dimension of the one-hot input vector data
        self.sizeinputonehot = sizeinputonehot  # for now works only with one-hot data
        self.keep_prob = keep_prob
        if self.keep_prob == 1.:
            self.keep_prob = None
        if keep_prob is not None:
            if not 0 < keep_prob <= 1:
                    raise ValueError("SpecificGDCEncoding.__init__ keep_prob must be a float in the "
                                     "range (0, 1], got %g" % keep_prob)

        self.nb_neutral = None  # init in the method bellows
        choices = self.which_goes_where()

        # build the null element encoding:
        nullElemenEnc = np.zeros(shape=self._proper_size_init(), dtype=np.float32)
        nullElemenEnc[choices == self.sizeinputonehot] = 1.
        self.masks = EncodingRaw(num_elem=self.sizeinputonehot, nullElemenEnc=nullElemenEnc)
        self.name_op = name
        if not reload:
            null_key = np.zeros(self.sizeinputonehot, dtype=np.int)
            # build the other masks
            for i in range(self.sizeinputonehot):
                tmp_key = copy.deepcopy(null_key)
                tmp_key[i] = 1
                tmp_val = copy.deepcopy(nullElemenEnc)
                tmp_val[choices == i] = 1
                self.masks[tmp_key.tostring()] = tmp_val
            self.masks.save(path=path, name=name)
        else:
            self.masks.reload(path=path, name=name)

    def __call__(self, x):
        """
        Convert the input 'x' in the associated guided dropconnect mask
        :param x: a tensorflow tensor: the input
        :return: the associated mask
        """
        res = tf.py_func(func=self._guided_drop, inp=[x], Tout=tf.float32, name=self.name_op)
        res.set_shape(self._proper_size_tf())
        return res

    def _guided_drop(self, x):
        """
        Function that implement guided dropconnect: all element should have the same "tau" values
        :param x: a numpy two dimensional array
        :return:
        """
        test_same_vect = np.apply_along_axis(lambda x: x.tostring(), 1, x)
        if len(np.unique(test_same_vect)) != 1:
            msg = "guided_dropconnect : different vector use for getting the mask. This is for now unsupported."
            raise RuntimeError(msg)
        x = x[0, :]  # only the first line is relevant here, the other should be equal (guided dropconnect only)
        if x.shape != (self.sizeinputonehot,):
            msg = "guided_dropconnect: you give a vector with size {}, masks are defined with input of size {}".format(
                x.shape[0], self.sizeinputonehot)
            raise RuntimeError(msg)
        x = x.reshape(self.sizeinputonehot)
        x = x.astype(np.int)
        xhash = x.tostring()
        res = self.masks[xhash]
        if self.keep_prob is not None:
            res = self.dropout(res)
        return res

    def dropout(self, res):
        """
        Implement regular dropout in units always present
        inspired from tensorflow dropout implmentation at
        https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/python/ops/nn_ops.py
        :param res:
        :return:
        """
        noise_shape = res.shape
        noise_shape = (noise_shape[0], self.nb_neutral)
        random_tensor = self.keep_prob
        random_tensor += np.random.uniform(noise_shape)
        binary_tensor = np.concatenate((np.floor(random_tensor),
                                        np.ones((noise_shape[0], res.shape[1] - noise_shape[1]))),
                                       axis=1
                                       )
        res = res / self.keep_prob * binary_tensor
        return res

    def which_goes_where(self):
        """
        Decides at random which connection is assign to which masks.
        :return: a vector of the proper shape, with integer between 0 and "ndim*" stating which connection is assigned
        :return: to which mask. (connections numbered ndim are assign to the "null element" encoding)
        """

        ## try to equilibrate the number of connection per dimension
        numberofconnections = self.size_in * self.size_out
        # if self.nb_connections is None:
        #     self.nb_neutral = int((1 - self.proba_select) * numberofconnections)
        # else:
        self.nb_neutral = numberofconnections-self.nb_connections*self.sizeinputonehot

        rest_to_fill = numberofconnections - self.nb_neutral
        distributed = list(range(self.sizeinputonehot))

        if rest_to_fill // self.sizeinputonehot == 0:
            msg = "W /!\ guided_dropout / dropconnect: There are 0 connections assigned to some masks.\n"
            msg += "W /!\ Masking will not work properly!\n"
            msg += "W /!\ Consider adding more units to the masks"
            msg += " (or decreasing nbconnections).\n"
            print(msg)
        choices = distributed * (rest_to_fill // self.sizeinputonehot)
        choices += distributed[:(rest_to_fill % self.sizeinputonehot)]
        choices = np.array(choices)
        for i in range(self.size_out):
            # make sur the shuffling shuffles properly
            np.random.shuffle(choices)
        choices = np.concatenate((np.array([self.sizeinputonehot] * self.nb_neutral), choices)) #neutral element always at the beginning
        choices = choices.reshape(self._proper_size_init())
        return choices

    def _proper_size_tf(self):
        return self.size_in, self.size_out

    def _proper_size_init(self):
        return self.size_in, self.size_out


class SpecificGDOEncoding(SpecificGDCEncoding):
    def __init__(self, sizeinputonehot, sizeout, nbconnections=1,
                 name="guided_dropout_encoding",
                 path=".", reload=False,
                 keep_prob=None):
        """
        Encoding for guided dropout.
        :param sizeinputonehot: size of the tau vector
        :param name: name of the operation
        :param nbconnections: number of connections used for each dimension of the one-hot input vector (overwrite proba_select)
        :param path: path where mask will be stored
        :param reload: do you want to reload (T) or build (F) the mask
        :param keep_prob: the keeping probability for regular dropout (applied only in the units not "guided dropout'ed")
        """
        # this is simply used for saving at the moment :
        SpecificGDCEncoding.__init__(self, sizeinputonehot=sizeinputonehot, nrow=1,
                                     ncol=sizeout, name=name,
                                     path=path, reload=reload, nbconnections=nbconnections,
                                     keep_prob=keep_prob)  # this is simply used for saving at the moment

        eye = np.identity(sizeinputonehot, dtype=np.float32)
        eye = np.concatenate([eye for _ in range(nbconnections)], axis=1)
        self.nb_connections_used = nbconnections*sizeinputonehot
        self.nb_ones = sizeout - nbconnections*sizeinputonehot
        if self.nb_ones < 0:
            msg = "SpecificGDOEncoding : the output size \"size_out\" is to small for the number of connection"
            msg += " \"nbconnections\" required for a size of \"sizeinputonehot\". Meaning that \"nbconnections\" * "
            msg += "\"sizeinputonehot\" is bigger than \"sizeout\". Try increases the size of your neural network "
            msg += "or reducing (if possible) the number of unit per dimension."
            raise RuntimeError(msg)

        self.my_one = tf.ones(self.nb_ones)
        self.mat = tf.convert_to_tensor(eye, DTYPE_USED)
        self.mask = None

    def __call__(self, h):
        """
        :param h:
        :return:
        """
        if self.mask is None:
            msg = "Mask not set, call \"get_mask\" before applying guided dropout"
            raise RuntimeError(msg)
        tmp = tf.multiply(h[:, self.nb_ones:], self.mask)
        self.mask = None
        return tf.concat((h[:, :self.nb_ones], tmp), axis=1)

    def set_mask(self, x):
        """
        Convert the input 'x' in the associated guided dropconnect mask
        :param x: a tensorflow tensor: the one hot input
        :return: nothing
        """
        res = tf.matmul(x, self.mat, name="building_mask")
        self.mask = res

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda *i, **kwargs: i[0]  # pylint:disable=invalid-name

    # define parameters for the model
    # learning parameters
    nb_iteration = 20000
    batch_size = 50
    lr = 1e-3

    # architecture of the neural networks
    use_sigmoid = True  # use sigmoid as non linearity in the network
    nbUnit = 20
    guided_dropout_encoding = False  # use the guided dropout


    # load some test data
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from Generate_Test_data import pi, generate
    np.random.seed(42)  # for reproducible experiments

    NB_UNARY_ACTIONS = 2
    # generate training data
    x_I, S_I, tau_I, \
    x_I_test, S_I_test, tau_I_test \
        = generate(nb_dim=NB_UNARY_ACTIONS,
                               n_train=200,
                               n_test=100,
                               angle_range_S=5 * pi / 3,
                               radius_S=1.,
                               S_0_start_ph=0,     # The start phase of data
                               S_0_precision=10.,  # width of the "C's"
                               S_0_origin=[0, 0],  # The origin of source domain with label = 0
                               nb_unit_per_dim=1)

    inputSize = x_I.shape[-1]
    inputSize_real = inputSize
    output_size = S_I.shape[-1]
    sizeinputonehot = tau_I.shape[-1]

    latentdim = nbUnit+sizeinputonehot
    # neural network weights
    if not guided_dropout_encoding:
        inputSize_real += sizeinputonehot

    ## first layer (encoding the data)
    w_input = tf.get_variable(name="w_input",
                         shape=[inputSize_real, latentdim],
                         initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                         trainable=True)
    ## second layer (the one with guided dropout)
    w_gd = tf.get_variable(name="w_gd",
                         shape=[latentdim, latentdim],
                         initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                         trainable=True)
    ## third layer (decoding the data)
    w_output = tf.get_variable(name="w_output",
                               shape=[latentdim, output_size],
                               initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False),
                               trainable=True)

    ## defining the guided dropout operator, and the mask
    gd_op = SpecificGDOEncoding(sizeinputonehot=sizeinputonehot,
                                sizeout=latentdim,
                                )

    ## place holders for data
    phInput = tf.placeholder("float32", shape=[None, inputSize])  # x in the paper
    phOutput = tf.placeholder("float32", shape=[None, output_size])  # y_hat in the paper
    phGD = tf.placeholder("float32", shape=[None, sizeinputonehot])  # tau in the paper

    if use_sigmoid:
        activation = tf.nn.sigmoid
    else:
        activation = tf.nn.relu

    data_input = phInput
    if not guided_dropout_encoding:
        data_input = tf.concat((phInput, phGD), axis=1)

    # computation graph
    h_0 = activation(tf.matmul(data_input, w_input))
    h_1 = tf.matmul(h_0, w_gd)

    if guided_dropout_encoding:
        ##### This is the part where guided dropout take place ###########
        # get the proper mask
        gd_op.set_mask(phGD)
        # now perform the mask operation
        h_1_after_gd = activation(gd_op(h_1))
        ##### End of the part where guided dropout take place ###########
    else:
        h_1_after_gd = activation(h_1)

    # and retrieve the output
    y_hat = tf.matmul(h_1_after_gd, w_output)


    # the loss
    rec_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y_hat - phOutput, 2), axis=1))

    # the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    optimize_glob = optimizer.minimize(loss=rec_loss, name="optimizer_glob")

    all_indxs = list(range(x_I.shape[0]))
    training_curve = []
    iterations_num = []

    samples = list(range(5))
    samples = [0]+[2**i for i in range(NB_UNARY_ACTIONS)]
    inputs = x_I[samples].reshape(len(samples) * x_I.shape[1], x_I.shape[-1]).copy()
    outputs = S_I[samples].reshape(len(samples) * x_I.shape[1], S_I.shape[-1]).copy()
    masks = tau_I[samples].reshape(len(samples) * x_I.shape[1], tau_I.shape[-1]).copy()
    all_indxs = list(range(outputs.shape[0]))


    # now fit the model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in tqdm(range(nb_iteration)):
            idxs = np.random.choice(all_indxs, size=batch_size)

            # batch input data
            in_batch = inputs[idxs, :]  # if not forget_input else np.random.normal(size=(batch_size, nvar))
            _, l = sess.run([optimize_glob, rec_loss],
                            feed_dict={phInput: in_batch,
                                       phGD: masks[idxs, :],
                                       phOutput: outputs[idxs, :]})
            test_tmp = sess.run(h_1_after_gd,
                            feed_dict={phInput: in_batch,
                                       phGD: masks[idxs, :],
                                       phOutput: outputs[idxs, :]})
            # pdb.set_trace()
            if (i + 1) % 100 == 0 or i == nb_iteration - 1:
                training_curve.append(l)
                iterations_num.append(i + 1)

        inputs_test = x_I_test[samples].reshape(len(samples) * x_I_test.shape[1], x_I_test.shape[-1]).copy()
        outputs_test = S_I_test[samples].reshape(len(samples) * x_I_test.shape[1], S_I_test.shape[-1]).copy()
        masks_test = tau_I_test[samples].reshape(len(samples) * x_I_test.shape[1], tau_I_test.shape[-1]).copy()
        y_hat_final_test = sess.run(y_hat,
                               feed_dict={phInput: inputs_test,
                                          phGD: masks_test})
        samples_set = set(samples)
        samples_supertest = [i for i in range(2**NB_UNARY_ACTIONS) if not i in samples_set]
        inputs_supertest = x_I_test[samples_supertest].reshape(len(samples_supertest) * x_I_test.shape[1], x_I_test.shape[-1]).copy()
        outputs_supertest = S_I_test[samples_supertest].reshape(len(samples_supertest) * x_I_test.shape[1], S_I_test.shape[-1]).copy()
        masks_supertest = tau_I_test[samples_supertest].reshape(len(samples_supertest) * x_I_test.shape[1], tau_I_test.shape[-1]).copy()
        y_hat_final_supertest = sess.run(y_hat,
                               feed_dict={phInput: inputs_supertest,
                                          phGD: masks_supertest,
                                          phOutput: outputs_supertest})

    xlim = [np.min(S_I_test.reshape(S_I_test.shape[0] * S_I_test.shape[1], S_I_test.shape[-1])[:, 0]),
            np.max(S_I_test.reshape(S_I_test.shape[0] * S_I_test.shape[1], S_I_test.shape[-1])[:, 0])]
    ylim = [np.min(S_I_test.reshape(S_I_test.shape[0] * S_I_test.shape[1], S_I_test.shape[-1])[:, 1]),
            np.max(S_I_test.reshape(S_I_test.shape[0] * S_I_test.shape[1], S_I_test.shape[-1])[:, 1])]
    plt.figure(1)
    plt.plot(iterations_num, training_curve)
    plt.show()

    plt.figure(1)
    plt.plot(y_hat_final_test[:, 0], y_hat_final_test[:, 1], ".", color="red", label="pred")
    plt.plot(outputs_test[:, 0], outputs_test[:, 1], ".", color="blue", label="true")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if NB_UNARY_ACTIONS == 2:
        plt.text(0, 0, s=r"$\tau = [0,0]$", horizontalalignment="center", verticalalignment="bottom")
        plt.text(0, 0, s=r"no line disconnected", horizontalalignment="center", verticalalignment="top")
        plt.text(0, 3, s=r"$\tau = [0,1]$", horizontalalignment="center", verticalalignment="bottom")
        plt.text(0, 3, s="line 1 disconnected", horizontalalignment="center", verticalalignment="top")
        plt.text(3, 0, s=r"$\tau = [1,0]$", horizontalalignment="center", verticalalignment="bottom")
        plt.text(3, 0, s="line 2 disconnected", horizontalalignment="center", verticalalignment="top")

    plt.title("On the test set")
    plt.show()

    plt.figure(1)
    plt.plot(y_hat_final_supertest[:, 0], y_hat_final_supertest[:, 1], ".", color="red", label="pred")
    plt.plot(outputs_supertest[:, 0], outputs_supertest[:, 1], ".", color="blue", label="true")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    if NB_UNARY_ACTIONS == 2:
        plt.text(3, 3, s=r"$\tau = [1,1]$", horizontalalignment="center", verticalalignment="bottom")
        plt.text(3, 3, s="lines 1 and 2 disconnected", horizontalalignment="center", verticalalignment="top")
    plt.title("On the super test set")
    plt.show()





