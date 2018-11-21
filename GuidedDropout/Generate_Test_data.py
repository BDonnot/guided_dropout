import numpy as np
from numpy import pi

# parameters of the data generation process
NB_UNARY_ACTIONS = 15
n_train = 100
angle_range_S = 5 * pi / 3
radius_S = 1.
S_0_start_ph = 0  # The start phase of data
S_0_precision = 10.
S_0_origin = [0, 0]  # The origin of source domain with label = 0
nb_unit_per_dim = 1

class PertAdd:
    def __init__(self, value=None, nb_dim=10, pert_num=None):
        """
        Represent an additive pertubation
        :param value: the value to add
        """

        self.nb_dim = nb_dim
        if value is None:
            center = np.random.uniform(low=-1, high=1, size=(2))*nb_dim*2.5
            tmp = np.random.normal(size=(2)) + center
            self.val = tmp.reshape(1, 2)
            if pert_num is not None and nb_dim == 2:
                # for a better representation of the results in the github repository
                if pert_num == 0:
                    self.val = np.array([0, 3]).reshape(1, 2)
                if pert_num == 1:
                    self.val = np.array([3, 0]).reshape(1, 2)
        else:
            self.val = value

    def __call__(self, x):
        return np.broadcast_to(self.val, x.shape).copy()

    def __add__(self, other):
        res = PertAdd(self.val + other.val)
        return res

    def assign_null(self):
        """
        :return: the null element of this king of function
        """
        res = PertAdd(value=np.zeros(2), nb_dim=self.nb_dim)
        return res

def F(center, x, pertubartions=PertAdd() ):
    return np.add(center, x[:, [1]] * np.append(np.cos(x[:, [0]]), np.sin(x[:, [0]]), axis=1)) + pertubartions(x)

def sampleX(n_samples, S_0_start_ph, angle_range_S, radius_S, S_0_precision):
    x1 = np.random.uniform(low=S_0_start_ph, high=S_0_start_ph + angle_range_S, size=(n_samples, 1))
    x2 = np.random.normal(loc=radius_S, scale=radius_S / S_0_precision, size=(n_samples, 1))
    return np.append(x1, x2, axis=1)


def generate(nb_dim=10,
             n_train=100,
             n_test=100 , # number of point in the test set
             angle_range_S=5 * pi / 3,
             radius_S=1.,
             S_0_start_ph=0,  # The start phase of data
             S_0_precision=10.,
             S_0_origin=[0, 0],  # The origin of source domain with label = 0
             nb_unit_per_dim=1,
             pertType=PertAdd):

    str_ref_tau = "{}".format(nb_dim)
    str_ref_tau = "{:0"+str_ref_tau+"b}"
    # nb_layer_added_gd = nb_unit_per_dim * nb_dim

    # generate the alpha^i's
    alphas = [None for i in range(nb_dim)]

    for i in range(nb_dim):
        alphas[i] = pertType(nb_dim=nb_dim, pert_num=i)


    x_I_test = np.zeros((2**nb_dim, n_test, 2))
    S_I_test = np.zeros((2**nb_dim, n_test, 2))
    tau_I_test = np.zeros((2**nb_dim, n_test, nb_dim))

    x_I = np.zeros((2**nb_dim, n_train, 2))
    S_I = np.zeros((2**nb_dim, n_train, 2))
    tau_I = np.zeros((2**nb_dim, n_train, nb_dim))
    for i in range(2**nb_dim):
        x_I[i] = sampleX(n_train, S_0_start_ph, angle_range_S, radius_S, S_0_precision)
        x_I_test[i] = sampleX(n_test, S_0_start_ph, angle_range_S, radius_S, S_0_precision)

        pertubartions = pertType(nb_dim=nb_dim).assign_null()
        binary_str = str_ref_tau.format(i)
        tmp_tau = np.zeros((1, nb_dim))
        for ith, char in enumerate(binary_str):
            if char == "1":
                tmp = alphas[ith]
                pertubartions = pertubartions + tmp
                tmp_tau[0, ith] = 1
        tau_I[i] = np.repeat(tmp_tau, axis=0, repeats=n_train)
        tau_I_test[i] = np.repeat(tmp_tau, axis=0, repeats=n_test)
        S_I[i] = F(center=S_0_origin, x=x_I[i], pertubartions=pertubartions)
        S_I_test[i] = F(center=S_0_origin, x=x_I_test[i], pertubartions=pertubartions)
    return x_I, S_I, tau_I, x_I_test, S_I_test, tau_I_test


# if __name__ == "__main__":
#     # parameters for saving the plots
#     save_plt = False
#     path_plot = "/home/bdonnot/Documents/nips-2018/plot"
#
#     nb_dim = NB_UNARY_ACTIONS  # don't change for now!
#     np.random.seed(42)  # for reproducible experiments
#
#     pertType = PertHeteroScedastik  # the pertubation type
#     x_I, S_I, tau_I = generate(nb_dim=nb_dim,
#                                n_train=n_train,
#                                # n_test=100 , # number of point in the test set of the second stuff
#                                angle_range_S=angle_range_S,
#                                radius_S=radius_S,
#                                S_0_start_ph=S_0_start_ph,  # The start phase of data
#                                S_0_precision=S_0_precision,
#                                S_0_origin=S_0_origin,  # The origin of source domain with label = 0
#                                nb_unit_per_dim=nb_unit_per_dim,
#                                pertType=pertType)
#
#     str_ref_tau = "{}".format(nb_dim)
#     str_ref_tau = "{:0"+str_ref_tau+"b}"
#
#     fig1 = plt.figure()
#     ax = fig1.add_subplot(1, 1, 1)
#     # ax.plot(S_I[0][:,0], S_I[0][:,1], "+", label="source")
#     for i in range(2**nb_dim):
#         binary_str = str_ref_tau.format(i)
#         ax.plot(S_I[i,:,0], S_I[i,:,1], "+", label="domain {}".format(binary_str))
#     # fig1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
#     ax.set_xlabel(r"$y_1$")
#     ax.set_ylabel(r"$y_2$")
#     ax.legend()
#     # plt.title("Inputs")
#     fig1.show()
#     if save_plt:
#         plt.savefig(os.path.join(path_plot, "moon_x_{}_multistuff.png".format("lin" if linear else "rot")))