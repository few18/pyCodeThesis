#!/usr/bin/python3

# import the required libraries
import numpy as np
import scipy.stats
import functools
import argparse
import scipy as sc
import numpy.matlib as npm
import sys
import random
# from sklearn import preprocessing


# def surrogates(x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"):
#     """
#     Returns iAAFT surrogates of given time series.
#
#     Parameter
#     ---------
#     x : numpy.ndarray, with shape (N,)
#         Input time series for which IAAFT surrogates are to be estimated.
#     ns : int
#         Number of surrogates to be generated.
#     tol_pc : float
#         Tolerance (in percent) level which decides the extent to which the
#         difference in the power spectrum of the surrogates to the original
#         power spectrum is allowed (default = 5).
#     verbose : bool
#         Show progress bar (default = `True`).
#     maxiter : int
#         Maximum number of iterations before which the algorithm should
#         converge. If the algorithm does not converge until this iteration
#         number is reached, the while loop breaks.
#     sorttype : string
#         Type of sorting algorithm to be used when the amplitudes of the newly
#         generated surrogate are to be adjusted to the original data. This
#         argument is passed on to `numpy.argsort`. Options include: 'quicksort',
#         'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
#         information. Note that although quick sort can be a bit faster than
#         merge sort or heap sort, it can, depending on the data, have worse case
#         spends that are much slower.
#
#     Returns
#     -------
#     xs : numpy.ndarray, with shape (ns, N)
#         Array containing the IAAFT surrogates of `x` such that each row of `xs`
#         is an individual surrogate time series.
#
#     See Also
#     --------
#     numpy.argsort
#
#     """
#     # as per the steps given in Lancaster et al., Phys. Rep (2018)
#     nx = x.shape[0]
#     xs = np.zeros((ns, nx))
#     maxiter = 10000
#     ii = np.arange(nx)
#
#     # get the fft of the original array
#     x_amp = np.abs(np.fft.fft(x))
#     x_srt = np.sort(x)
#     r_orig = np.argsort(x)
#
#     # loop over surrogate number
#     for k in range(ns):
#
#         # 1) Generate random shuffle of the data
#         count = 0
#         r_prev = np.random.permutation(ii)
#         r_curr = r_orig
#         z_n = x[r_prev]
#         percent_unequal = 100.
#
#         # core iterative loop
#         while (percent_unequal > tol_pc) and (count < maxiter):
#             r_prev = r_curr
#
#             # 2) FFT current iteration yk, and then invert it but while
#             # replacing the amplitudes with the original amplitudes but
#             # keeping the angles from the FFT-ed version of the random
#             y_prev = z_n
#             fft_prev = np.fft.fft(y_prev)
#             phi_prev = np.angle(fft_prev)
#             e_i_phi = np.exp(phi_prev * 1j)
#             z_n = np.fft.ifft(x_amp * e_i_phi)
#
#             # 3) rescale zk to the original distribution of x
#             r_curr = np.argsort(z_n, kind=sorttype)
#             z_n[r_curr] = x_srt.copy()
#             percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx
#
#             # 4) repeat until number of unequal entries between r_curr and
#             # r_prev is less than tol_pc percent
#             count += 1
#
#         if count >= (maxiter - 1):
#             print("maximum number of iterations reached!")
#
#         xs[k] = np.real(z_n)
#
#     return xs


class DeepESN():

    '''
    Deep Echo State Network (DeepESN) class:
    this class implement the DeepESN model suitable for
    time-serie prediction and sequence classification.
    Reference paper for DeepESN model:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99

    Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
    C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
    Neural Networks, 2018, vol. 108, pp. 33-47

    ----
    This file is a part of the DeepESN Python Library (DeepESNpy)
    Luca Pedrelli
    luca.pedrelli@di.unipi.it
    lucapedrelli@gmail.com
    Department of Computer Science - University of Pisa (Italy)
    Computational Intelligence & Machine Learning (CIML) Group
    http://www.di.unipi.it/groups/ciml/
    ----
    '''

    def __init__(self, Nu,Nr,Nl, configs, verbose=0):
        # initialize the DeepESN model

        if verbose:
            sys.stdout.write('init DeepESN...')
            sys.stdout.flush()

        rhos = np.array(configs.rhos) # spectral radius (maximum absolute eigenvalue)
        lis = np.array(configs.lis) # leaky rate
        iss = np.array(configs.iss) # input scale
        IPconf = configs.IPconf # configuration for Deep Intrinsic Plasticity
        reservoirConf = configs.reservoirConf # reservoir configurations

        if len(rhos.shape) == 0:
            rhos = npm.repmat(rhos, 1,Nl)[0]

        if len(lis.shape) == 0:
            lis = npm.repmat(lis, 1,Nl)[0]

        if len(iss.shape) == 0:
            iss = npm.repmat(iss, 1,Nl)[0]

        self.W = {} # recurrent weights
        self.Win = {} # recurrent weights
        self.Gain = {} # activation function gain
        self.Bias = {} # activation function bias

        self.Nu = Nu # number of inputs
        self.Nr = Nr # number of units per layer
        self.Nl = Nl # number of layers
        self.rhos = rhos.tolist() # list of spectral radius
        self.lis = lis # list of leaky rate
        self.iss = iss # list of input scale

        self.IPconf = IPconf

        self.readout = configs.readout

        # sparse recurrent weights init
        if reservoirConf.connectivity < 1:
            for layer in range(Nl):
                self.W[layer] = np.zeros((Nr,Nr))
                for row in range(Nr):
                    number_row_elements = round(reservoirConf.connectivity * Nr)
                    row_elements = random.sample(range(Nr), number_row_elements)
                    self.W[layer][row,row_elements] = np.random.uniform(-1,+1, size = (1,number_row_elements))

        # full-connected recurrent weights init
        else:
            for layer in range(Nl):
                self.W[layer] = np.random.uniform(-1,+1, size = (Nr,Nr))

        # layers init
        for layer in range(Nl):

            target_li = lis[layer]
            target_rho = rhos[layer]
            input_scale = iss[layer]

            if layer==0:
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nr,Nu+1))
            else:
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nr,Nr+1))

            Ws = (1-target_li) * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]) + target_li * self.W[layer]
            eig_value,eig_vector = np.linalg.eig(Ws)
            actual_rho = np.max(np.absolute(eig_value))

            Ws = (Ws *target_rho)/actual_rho
            self.W[layer] = (target_li**-1) * (Ws - (1.-target_li) * np.eye(self.W[layer].shape[0], self.W[layer].shape[1]))

            self.Gain[layer] = np.ones((Nr,1))
            self.Bias[layer] = np.zeros((Nr,1))

        if verbose:
            print('done.')
            sys.stdout.flush()

    def computeLayerState(self, input, layer, initialStatesLayer = None, DeepIP = 0):
        # compute the state of a layer with pre-training if DeepIP == 1

        state = np.zeros((self.Nr, input.shape[1]))

        if initialStatesLayer is None:
            initialStatesLayer = np.zeros(state[:,0:1].shape)

        input = self.Win[layer][:,0:-1].dot(input) + np.expand_dims(self.Win[layer][:,-1],1)

        if DeepIP:
            state_net = np.zeros((self.Nr, input.shape[1]))
            state_net[:,0:1] = input[:,0:1]
            state[:,0:1] = self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], state_net[:,0:1]) + self.Bias[layer])
        else:
            #state[:,0:1] = self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], input[:,0:1]) + self.Bias[layer])
            state[:,0:1] = (1-self.lis[layer]) * initialStatesLayer + self.lis[layer] * np.tanh( np.multiply(self.Gain[layer], self.W[layer].dot(initialStatesLayer) + input[:,0:1]) + self.Bias[layer])

        for t in range(1,state.shape[1]):
            if DeepIP:
                state_net[:,t:t+1] = self.W[layer].dot(state[:,t-1:t]) + input[:,t:t+1]
                state[:,t:t+1] = (1-self.lis[layer]) * state[:,t-1:t] + self.lis[layer] * np.tanh(np.multiply(self.Gain[layer], state_net[:,t:t+1]) + self.Bias[layer])

                eta = self.IPconf.eta
                mu = self.IPconf.mu
                sigma2 = self.IPconf.sigma**2

                # IP learning rule
                deltaBias = -eta*((-mu/sigma2)+ np.multiply(state[:,t:t+1], (2*sigma2+1-(state[:,t:t+1]**2)+mu*state[:,t:t+1])/sigma2))
                deltaGain = eta / npm.repmat(self.Gain[layer],1,state_net[:,t:t+1].shape[1]) + deltaBias * state_net[:,t:t+1]

                # update gain and bias of activation function
                self.Gain[layer] = self.Gain[layer] + deltaGain
                self.Bias[layer] = self.Bias[layer] + deltaBias

            else:
                state[:,t:t+1] = (1-self.lis[layer]) * state[:,t-1:t] + self.lis[layer] * np.tanh( np.multiply(self.Gain[layer], self.W[layer].dot(state[:,t-1:t]) + input[:,t:t+1]) + self.Bias[layer])

        return state

    def computeDeepIntrinsicPlasticity(self, inputs):
        # we incrementally perform the pre-training (deep intrinsic plasticity) over layers

        len_inputs = range(len(inputs))
        states = []

        for i in len_inputs:
            states.append(np.zeros((self.Nr*self.Nl, inputs[i].shape[1])))

        for layer in range(self.Nl):

            for epoch in range(self.IPconf.Nepochs):
                Gain_epoch = self.Gain[layer]
                Bias_epoch = self.Bias[layer]


                if len(inputs) == 1:
                    self.computeLayerState(inputs[0][:,self.IPconf.indexes], layer, DeepIP = 1)
                else:
                    for i in self.IPconf.indexes:
                        self.computeLayerState(inputs[i], layer, DeepIP = 1)


                if (np.linalg.norm(self.Gain[layer]-Gain_epoch,2) < self.IPconf.threshold) and (np.linalg.norm(self.Bias[layer]-Bias_epoch,2)< self.IPconf.threshold):
                    sys.stdout.write(str(epoch+1))
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    break

                if epoch+1 == self.IPconf.Nepochs:
                    sys.stdout.write(str(epoch+1))
                    sys.stdout.write('.')
                    sys.stdout.flush()

            inputs2 = []
            for i in range(len(inputs)):
                inputs2.append(self.computeLayerState(inputs[i], layer))

            for i in range(len(inputs)):
                states[i][(layer)*self.Nr: (layer+1)*self.Nr,:] = inputs2[i]

            inputs = inputs2

        return states

    def computeState(self,inputs, DeepIP = 0, initialStates = None, verbose=0):
        # compute the global state of DeepESN with pre-training if DeepIP == 1

        if self.IPconf.DeepIP and DeepIP:
            if verbose:
                sys.stdout.write('compute state with DeepIP...')
                sys.stdout.flush()
            states = self.computeDeepIntrinsicPlasticity(inputs)
        else:
            if verbose:
                sys.stdout.write('compute state...')
                sys.stdout.flush()
            states = []

            for i_seq in range(len(inputs)):
                states.append(self.computeGlobalState(inputs[i_seq], initialStates))

        if verbose:
            print('done.')
            sys.stdout.flush()

        return states

    def computeGlobalState(self,input, initialStates):
        # compute the global state of DeepESN

        state = np.zeros((self.Nl*self.Nr,input.shape[1]))

        initialStatesLayer = None


        for layer in range(self.Nl):
            if initialStates is not None:
                initialStatesLayer = initialStates[(layer)*self.Nr: (layer+1)*self.Nr,:]
            state[(layer)*self.Nr: (layer+1)*self.Nr,:] = self.computeLayerState(input, layer, initialStatesLayer, 0)
            input = state[(layer)*self.Nr: (layer+1)*self.Nr,:]

        return state

    def trainReadout(self,trainStates,trainTargets,lb, verbose=0):
        # train the readout of DeepESN

        trainStates = np.concatenate(trainStates,1)
        trainTargets = np.concatenate(trainTargets,1)

        # add bias
        X = np.ones((trainStates.shape[0]+1, trainStates.shape[1]))
        X[:-1,:] = trainStates
        trainStates = X

        if verbose:
            sys.stdout.write('train readout...')
            sys.stdout.flush()

        if self.readout.trainMethod == 'SVD': # SVD, accurate method
            U, s, V = np.linalg.svd(trainStates, full_matrices=False);
            s = s/(s**2 + lb)

            self.Wout = trainTargets.dot(np.multiply(V.T, np.expand_dims(s,0)).dot(U.T));

        else:  # NormalEquation, fast method
            B = trainTargets.dot(trainStates.T)
            A = trainStates.dot(trainStates.T)

            self.Wout = np.linalg.solve((A + np.eye(A.shape[0], A.shape[1]) * lb), B.T).T

        if verbose:
            print('done.')
            sys.stdout.flush()

    def computeOutput(self,state):
        # compute a linear combination between the global state and the output weights
        state = np.concatenate(state,1)
        return self.Wout[:,0:-1].dot(state) + np.expand_dims(self.Wout[:,-1],1) # Wout product + add bias

# define deep esn function for getting correlation
def Causal_ESN_deep(X,Y):

    class Struct(object): pass

    def config_data(IP_indexes):

        configs = Struct()

        configs.rhos = 0.1 # set spectral radius 0.1 for all recurrent layers
        configs.lis = [0.1,0.5,0.9] # set li to 0.1,0.5,0.9 for the recurrent layers
        configs.iss = 1 # set insput scale 0.1 for all recurrent layers

        configs.IPconf = Struct()
        configs.IPconf.DeepIP = 0 # activate pre-train
        configs.IPconf.threshold = 0.1 # threshold for gradient descent in pre-train algorithm
        configs.IPconf.eta = 10e-5 # learning rate for IP rule
        configs.IPconf.mu = 0 # mean of target gaussian function
        configs.IPconf.sigma = 0.1 # std of target gaussian function
        configs.IPconf.Nepochs = 10 # maximum number of epochs
        configs.IPconf.indexes = IP_indexes # perform the pre-train on these indexes

        configs.reservoirConf = Struct()
        configs.reservoirConf.connectivity = 1 # connectivity of recurrent matrix

        configs.readout = Struct()
        configs.readout.trainMethod = 'SVD'
        configs.readout.regularizations = 10.0**np.array(range(-4,-1,1))

        return configs

    def select_indexes(data, indexes, transient=0):

        if len(data) == 1:
            return [data[0][:,indexes][:,transient:]]

        return [data[i][:,transient:] for i in indexes]

    def load_data(X, Y):

        #data = loadmat('MG.mat') # load dataset
        #data = np.loadtxt('MackeyGlass_t17.txt')
        #data = np.reshape(data,(1,10000))
        X = np.reshape(X,(1,X.shape[0]))
        Y = np.reshape(Y,(1,Y.shape[0]))
        #print(data.shape)
        dataset = Struct()
        dataset.name = ['DATA']
        dataset.inputs = [X]
        dataset.targets = [Y]

        # input dimension
        Nu = dataset.inputs[0].shape[0]

        # function used for model evaluation
        # error_function = functools.partial(metric_function)

        # select the model that achieves the maximum accuracy on validation set
        optimization_problem = np.argmin

        tr_ind = int(X.shape[1]*0.8)

        TR_indexes = range(tr_ind) # indexes for training, validation and test set in Piano-midi.de task
        VL_indexes = range(tr_ind,X.shape[1])
        TS_indexes = range(Y.shape[1])

        return dataset, Nu, optimization_problem, TR_indexes, VL_indexes, TS_indexes

    dataset, Nu, optimization_problem, TR_indexes, VL_indexes, TS_indexes = load_data(X, Y)

    # load configuration for pianomidi task
    configs = config_data(list(TR_indexes) + list(VL_indexes))

    # Be careful with memory usage
    Nr = 100 # number of recurrent units
    Nl = 3 # number of recurrent layers
    reg = 10.0e-6
    transient = 5

    deepESN = DeepESN(Nu, Nr, Nl, configs, verbose = 0)
    states = deepESN.computeState(dataset.inputs, deepESN.IPconf.DeepIP)

    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), transient)
    train_targets = select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), transient)
    test_states = select_indexes(states, TS_indexes)
    test_targets = select_indexes(dataset.targets, TS_indexes)
    deepESN.trainReadout(train_states, train_targets, reg)

    train_outputs = deepESN.computeOutput(train_states)

    test_outputs = deepESN.computeOutput(test_states)

    corr = scipy.stats.spearmanr(test_outputs, test_targets[0], axis = 1)

    return corr

# define function to run surrogate time series through deep esn
def compute_confidence(X_data, Y_data, verbose = False):

    n, T = X_data.shape

    lags = np.arange(0,21)
    xmapy = np.zeros((n,21))
    ymapx = np.zeros((n,21))

    for j in range(n):
        if verbose:
            if (j%20 == 0):
                print("Current Iteration {}".format(j))
        for i in range(21):

            if i < 11:
                L1 = 10 - i
                R1 = T
                L2 = 0
                R2 = T - 10 + i

            else:
                L1 = 0
                R1 = T - i + 10
                L2 = i - 10
                R2 = T

            X = X_data[j,L1:R1]
            Y = Y_data[j,L2:R2]
            xmapy[j,i] = Causal_ESN_deep(X,Y)[0]

            Y = Y_data[j,L1:R1]
            X = X_data[j,L2:R2]
            ymapx[j,i] = Causal_ESN_deep(Y,X)[0]

    return xmapy, ymapx


# define function to run actual time series through deep esn
def compute_lags(X_data, Y_data, n, verbose = False):

    T = X_data.shape[0]

    lags = np.arange(0,21)
    xmapy = np.zeros((n,21))
    ymapx = np.zeros((n,21))

    for j in range(n):
        if verbose:
            if (j%20 == 0):
                print("Current Iteration {}".format(j))
        for i in range(21):

            if i < 11:
                L1 = 10 - i
                R1 = T
                L2 = 0
                R2 = T - 10 + i

            else:
                L1 = 0
                R1 = T - i + 10
                L2 = i - 10
                R2 = T

            X = X_data[L1:R1]
            Y = Y_data[L2:R2]
            xmapy[j,i] = Causal_ESN_deep(X,Y)[0]

            Y = Y_data[L1:R1]
            X = X_data[L2:R2]
            ymapx[j,i] = Causal_ESN_deep(Y,X)[0]

    return xmapy, ymapx


def DeepESN_real(X, Y, Xsurr, Ysurr,  n_data = 50, verbose = False):

    # Get length of time series
    # N = X.shape[0]

    # generate surrogates for X and Y
    # Xsurr = surrogates(x=X, ns=n_surrogates, verbose=False)
    # Ysurr = surrogates(x=Y, ns=n_surrogates, verbose=False)

    # run the deepESN of the surrogates
    xmapy_surr, ymapx_surr = compute_confidence(Xsurr, Ysurr, verbose)

    # run deepESN for actual data
    xmapy, ymapx = compute_lags(X, Y, n_data, verbose)

    return xmapy_surr, ymapx_surr, xmapy, ymapx
# import data

combs = np.loadtxt('combs_guadalquivir.csv', delimiter = ',', dtype = str)

#get index of combination to run
parser = argparse.ArgumentParser(description="Guadalquivir")
parser.add_argument(
    "arrayidx", help="Runs deepesn for given combination of species", type=int, default=2
)
args = parser.parse_args()

# get names of columns of data to analyse
X_name = combs[args.arrayidx][0]
Y_name = combs[args.arrayidx][1]

# get the data
X = np.load(X_name + '.npy')
Y = np.load(Y_name + '.npy')

# get the surrogates
Xsurr = np.load(X_name + '_surr.npy')
Ysurr = np.load(Y_name + '_surr.npy')

# standardise data
X = preprocessing.scale(X)
Y = preprocessing.scale(Y)
Xsurr = preprocessing.scale(Xsurr, axis = 1)
Ysurr = preprocessing.scale(Ysurr, axis = 1)

# run the deep esn framework on the data
xmapy_surr, ymapx_surr, xmapy, ymapx = DeepESN_real(X, Y, Xsurr, Ysurr)
correlations = np.dstack((xmapy_surr,ymapx_surr,xmapy,ymapx))

# set the filename to the investigated species
file_name = "{}_{}".format(X_name, Y_name)
# save array into file
np.save(file_name, correlations)
