import numpy as np
import os
import sys
import time
import pickle
import logging
from os import listdir
from easydict import EasyDict as edict
from tqdm import tqdm
from scipy.signal import convolve2d
from sklearn.preprocessing import normalize
from sklearn import linear_model
from skimage.io import imread, imsave
from skimage.color import rgb2ycbcr, ycbcr2rgb, rgb2gray
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt

try:
    from spams import trainDL
except ImportError:
    print("Warning: 'spams' library not found. Dictionary training will not be available.")
    trainDL = None

import dimod
#import dynex
from qubovert import boolean_var
from qubovert.sim import anneal_qubo
try:
    from dwave.cloud import Client
except ImportError:
    print("Warning: 'dwave-ocean-sdk' not found. D-Wave solvers will not be available.")
    Client = None

print("All libraries imported.")



from time import perf_counter as tpc
from dwave.samplers import SimulatedAnnealingSampler
def DWaveBqmSolver(Q,num_reads=2000):
    bqm=dimod.BinaryQuadraticModel.from_qubo(Q)
    t=tpc()
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm,num_of_samples=num_reads)
    return {
        "configuration":list(sampleset.first.sample),
        "energy":sampleset.first.energy,
        "time":tpc()-t
    }


from qdeepsdk import QDeepHybridSolver
import requests
def QDeepSolver(Q,budget=50000,num_reads=10000):
    solver = QDeepHybridSolver()
    solver.token = "akwysie03c"
    solver.m_budget = budget
    solver.num_reads = num_reads
    matrix = np.array(Q)
    # Solve the QUBO problem
    try:
        response = solver.solve(matrix)
        results = response['QdeepHybridSolver']
        return results
    except ValueError as e:
        print(f"Error: {e}")
    except requests.RequestException as e:
        print(f"API Error: {e}")
    return {'configuration':[],'energy':-1,'time':-1}
def QDeepBqmSolver(bqm,budget=50000,num_reads=10000):
    N=0
    for p in bqm:
        N=max(N,p[0],p[1])
    N+=1
    Q=np.zeros((N,N))
    for p in bqm:
        Q[p[0]][p[1]]=bqm[p]
    ans= QDeepSolver(Q,budget,num_reads)
    while ans['time']==-1:
        print('qdeep server error, retrying...')
        ans= QDeepSolver(Q,budget,num_reads)
    return ans
#https://github.com/anabatsh/PROTES

#Protes
import protes

def func_build_qubo_ready(Q):
    """Predefined QUBO matrix optimization."""
    d = Q.shape[0]  # Dimensionality of the problem
    n = 2           # Binary decision variables (0 or 1)
    def func(I):
        """QUBO objective function."""
        return np.array([np.dot(x, Q @ x) for x in I])  # Evaluate QUBO
    return d, n, func

def ProtesSolver(Q,m=10000):
    Q=np.array(Q)
    d, n, f = func_build_qubo_ready(Q)
    t = tpc()
    i_opt, y_opt = protes.protes(f, d, n, m, k=500, k_top=30, log=False)
    x = np.array(i_opt)
    energy = x.T @ Q @ x
    return {
        "configuration": i_opt.tolist(),
        "energy": float(energy),
        "time":tpc()-t
    }


def ProtesBqmSolver(bqm,m=10000):
    N=0
    for p in bqm:
        N=max(N,p[0],p[1])
    N+=1
    Q=np.zeros((N,N))
    for p in bqm:
        Q[p[0]][p[1]]=bqm[p]
    ans= ProtesSolver(Q,m)
    return ans


import warnings
warnings.filterwarnings('ignore')

# --- Training Configuration ---
C_train = edict()
C_train.dict_size   = 128          # dictionary size
C_train.lmbd        = 0.1          # sparsity regularization
C_train.patch_size  = 5            # image patch size
C_train.nSmp        = 100000       # number of patches to sample
C_train.upscale     = 3            # upscaling factor

C_train.root_dir = os.path.realpath(".")
exp_time_train = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C_train.exp_name = 'exp_train_'+exp_time_train
C_train.output_dir = os.path.join(*[C_train.root_dir,'output', C_train.exp_name])
print(f"Training output directory: {C_train.output_dir}")


# Inference Configuration
C_run = edict()

# Select the algorithm for sparse coding sklearn_lasso, qubo_bsc, dynex, dynex_bsc, qubo_bsc_dwave1
C_run.sc_algo = ["sklearn_lasso","qubo_dwave","qubo_qdeep","qubo_protes"][0]
# C_run.sc_algo = "qubo_bsc_dwave1"

# General parameters (this should match the dictionaries used)
C_run.patch_size = 5
C_run.overlap = 3
C_run.upscale = 3
C_run.lmbd = 0.

C_run.Dl_path = "data/dicts/Dl_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"
C_run.Dh_path = "data/dicts/Dh_128_US3_L0.1_PS5_test_exp_train_2022_11_22_17_08_59.pkl"

# Parameters for sparse coding algorithms
C_run.lasso_alpha = 1e-5 # For sklearn_lasso
C_run.bsc_alpha = 0.1    # For QUBO-based methods
C_run.bsc_mu = 0.05      # For QUBO-based methods

# Parameters for D-Wave / Dynex
C_run.num_passes = 1
C_run.num_reads = 100
C_run.qubo_size = 512
C_run.subproblem_size = 32
 # Inverse temperature for Gibbs entropy calculation
C_run.beta = 1.0

# fall back paths in case of problems in data
mode_val = "1img_small"
#PATH
img_path="data/val_single/"
val_hr_path = {
    "1img_small":img_path+"HR",
}
C_run.val_hr_path = val_hr_path[mode_val]
val_lr_path = {
    "1img_small":img_path+"LR",
}
C_run.val_lr_path = val_lr_path[mode_val]


C_run.root_dir = os.path.realpath(".")
exp_time_run = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C_run.exp_name = f'exp_inference_{C_run.sc_algo}_{exp_time_run}'
C_run.output_dir = os.path.join(*[C_run.root_dir, 'output', C_run.exp_name])
print(f"Inference output directory: {C_run.output_dir}")

config = C_run # this comes from the main part
print(f"\nActive algorithm: {config.sc_algo}")

# helper functions
def sample_patches(img, patch_size, patch_num, upscale):
    """Samples random patches from a single image."""
    if img.ndim == 3 and img.shape[2] == 3:
        hIm = rgb2gray(img)
    else:
        hIm = img

    # Generate low resolution counter parts
    lIm = rescale(hIm, 1 / upscale, anti_aliasing=True)
    lIm = resize(lIm, hIm.shape, anti_aliasing=True)
    nrow, ncol = hIm.shape

    x = np.random.permutation(range(nrow - 2 * patch_size)) + patch_size
    y = np.random.permutation(range(ncol - 2 * patch_size)) + patch_size

    xrow, ycol = np.meshgrid(x, y)
    xrow = np.ravel(xrow, order='F')
    ycol = np.ravel(ycol, order='F')

    if patch_num < len(xrow):
        xrow = xrow[0 : patch_num]
        ycol = ycol[0 : patch_num]

    patch_num = len(xrow)

    HP = np.zeros((patch_size ** 2, patch_num))
    LP = np.zeros((4 * patch_size ** 2, patch_num))

    # Compute the first and second order gradients
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T

    lImG11 = convolve2d(lIm, hf1, 'same')
    lImG12 = convolve2d(lIm, vf1, 'same')
    lImG21 = convolve2d(lIm, hf2, 'same')
    lImG22 = convolve2d(lIm, vf2, 'same')

    for i in tqdm(range(patch_num), desc="Sampling Patches"):
        row, col = xrow[i], ycol[i]

        Hpatch = hIm[row : row + patch_size, col : col + patch_size].flatten('F')
        HP[:, i] = Hpatch - np.mean(Hpatch)

        Lpatch1 = lImG11[row : row + patch_size, col : col + patch_size].flatten('F')
        Lpatch2 = lImG12[row : row + patch_size, col : col + patch_size].flatten('F')
        Lpatch3 = lImG21[row : row + patch_size, col : col + patch_size].flatten('F')
        Lpatch4 = lImG22[row : row + patch_size, col : col + patch_size].flatten('F')

        LP[:, i] = np.concatenate((Lpatch1, Lpatch2, Lpatch3, Lpatch4))

    return HP, LP


def rnd_smp_patch(img_path, patch_size, num_patch, upscale):
    """Samples patches from a directory of images."""
    img_dir = [f for f in listdir(img_path) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
    if not img_dir:
        print(f"Error: No images found in {img_path}")
        return None, None

    img_num = len(img_dir)
    nper_img = np.zeros((img_num, 1))

    print("Calculating patch distribution per image...")
    for i in range(img_num):
        img = imread(os.path.join(img_path, img_dir[i]))
        nper_img[i] = img.shape[0] * img.shape[1]

    nper_img = np.floor(nper_img * num_patch / np.sum(nper_img))

    Xh_list, Xl_list = [], []
    print("Reading images and sampling patches...")
    for i in tqdm(range(img_num), desc="Processing Images"):
        patch_num = int(nper_img[i])
        if patch_num == 0:
            continue
        img = imread(os.path.join(img_path, img_dir[i]))
        H, L = sample_patches(img, patch_size, patch_num, upscale)
        Xh_list.append(H)
        Xl_list.append(L)

    if not Xh_list:
        print("Error: No patches were sampled.")
        return None, None

    Xh = np.concatenate(Xh_list, axis=1)
    Xl = np.concatenate(Xl_list, axis=1)

    return Xh, Xl

def patch_pruning(Xh, Xl):
    """Removes patches with low variance."""
    print("Pruning patches...")
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, 10)
    idx = pvars > threshold
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    print(f"Pruned from {len(pvars)} to {Xh.shape[1]} patches.")
    return Xh, Xl


# code from train_dict.py no updates

def train_dictionaries():
    # Set active config to training config
    config = C_train
    os.makedirs(config.output_dir, exist_ok=True)

    # Check for prerequisites
    if trainDL is None:
        print("Skipping training: 'spams' library is not installed.")
        return

    train_img_path = 'data/train_hr/'
    if not os.path.exists(train_img_path) or not listdir(train_img_path):
        print(f"Skipping training: Training directory '{train_img_path}' is empty or does not exist.")
        return

    # Setup Logging
    log_format = "%(asctime)s | %(message)s"
    # Clear existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p', stream=sys.stdout)
    fh = logging.FileHandler(os.path.join(config.output_dir, 'log_train.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("--- Starting Dictionary Training ---")
    logging.info(f"Config: {config}")

    # Randomly sample image patches
    logging.info("Sampling random patches from training images...")
    Xh, Xl = rnd_smp_patch(train_img_path, config.patch_size, config.nSmp, config.upscale)

    if Xh is None:
        logging.error("Patch sampling failed. Aborting training.")
        return

    # Prune patches with small variances
    Xh, Xl = patch_pruning(Xh, Xl)
    Xh = np.asfortranarray(Xh)
    Xl = np.asfortranarray(Xl)

    # Dictionary learning using SPAMS
    logging.info("Learning high-resolution dictionary (Dh)...")
    Dh = trainDL(Xh, K=config.dict_size, lambda1=config.lmbd, iter=100, mode=2, verbose=False)

    logging.info("Learning low-resolution dictionary (Dl)...")
    Dl = trainDL(Xl, K=config.dict_size, lambda1=config.lmbd, iter=100, mode=2, verbose=False)

    # Saving dictionaries to files
    os.makedirs('data/dicts', exist_ok=True)
    dict_suffix = f"{config.dict_size}_US{config.upscale}_L{config.lmbd}_PS{config.patch_size}_{config.exp_name}.pkl"

    dh_path = os.path.join('data/dicts', 'Dh_' + dict_suffix)
    dl_path = os.path.join('data/dicts', 'Dl_' + dict_suffix)

    with open(dh_path, 'wb') as f:
        pickle.dump(Dh, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Saved Dh to {dh_path}")

    with open(dl_path, 'wb') as f:
        pickle.dump(Dl, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Saved Dl to {dl_path}")

    logging.info("--- Dictionary Training Finished ---")
    print("\nIMPORTANT: Update the C_run.Dh_path and C_run.Dl_path in the configuration cell with these new file paths to use them for inference.")


# To run it uncomess
train_dictionaries()

# Utility functions for the core algorithm from the follwoing files ScSR.py, backprojection.py, and qubo_algorithms.py

def extract_lr_feat(img_lr):
    """Extracts first and second order gradient features."""
    h, w = img_lr.shape
    img_lr_feat = np.zeros((h, w, 4))

    # First order gradient filters
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T
    img_lr_feat[:, :, 0] = convolve2d(img_lr, hf1, 'same')
    img_lr_feat[:, :, 1] = convolve2d(img_lr, vf1, 'same')

    # Second order gradient filters
    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T
    img_lr_feat[:, :, 2] = convolve2d(img_lr, hf2, 'same')
    img_lr_feat[:, :, 3] = convolve2d(img_lr, vf2, 'same')

    return img_lr_feat

def create_list_step(start, stop, step):
    """Creates a list of numbers with a specific step."""
    return np.arange(start, stop, step)

def lin_scale(xh, us_norm):
    """Linearly scales a patch."""
    hr_norm = np.sqrt(np.sum(np.multiply(xh, xh)))
    if hr_norm > 0:
        lin_scale_factor = 1.2
        s = us_norm * lin_scale_factor / hr_norm
        xh = np.multiply(xh, s)
    return xh

def backprojection(img_hr, img_lr, maxIter=8):
    """Refines the high-res image by minimizing reconstruction error."""
    p = gauss2D((5, 5), 1)

    for i in range(maxIter):
        img_lr_ds = resize(img_hr, img_lr.shape, anti_aliasing=1)
        img_diff = img_lr - img_lr_ds
        img_diff = resize(img_diff, img_hr.shape)
        img_hr += convolve2d(img_diff, p, 'same')
    return img_hr

def gauss2D(shape=(5,5),sigma=1):
    """Generates a 2D gaussian mask."""
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



# QUBO formulation and solver interface functions I took them from qubo_algorithms.py

def qubo_bsc(X, y, alpha, mu): # Classical Annealing
    """Binary sparse coding using classical simulated annealing."""
    D = X.shape[1]
    w = np.ones(D) * mu
    m = update_m(X, y, w, alpha)
    return m * w

def update_m(X, y, w, alpha):
    """Helper for classical annealing via qubovert."""
    D = X.shape[1]
    m = {i: boolean_var(f'm({i})') for i in range(D)}
    A = np.linalg.multi_dot([np.diag(w), X.T, X, np.diag(w)])
    b = -2 * np.linalg.multi_dot([np.diag(w), X.T, y])
    b = b + alpha * w * np.sign(w)

    model = 0
    for i in range(D):
        for j in range(D):
            model += m[i] * (A[i, j] + 1e-9) * m[j]
        model += (b[i] + 1e-9) * m[i]

    res = anneal_qubo(model, num_anneals=1)
    model_solution = res.best.state
    return np.array(list(model_solution.values()))

def qubo_dynex(X, y, alpha, mu):
    """Binary sparse coding using Dynex update rule."""
    D = X.shape[1]
    w = np.ones(D) * mu
    m = update_m_dynex(X, y, w, alpha)
    return m * w

def update_m_dynex(X, y, w, alpha):
    """Constructs and solves a QUBO using the Dynex platform."""
    D = X.shape[1]
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    A = np.linalg.multi_dot([np.diag(w), X.T, X, np.diag(w)])
    b = -2 * np.linalg.multi_dot([np.diag(w), X.T, y])
    b = b + alpha * w * np.sign(w)

    # Add quadratic terms
    for i in range(D):
        for j in range(i, D):
            if i == j:
                # Add to linear term
                bqm.add_linear(i, A[i, i])
            else:
                # BQM API handles the factor of 2 for symmetric matrices
                bqm.add_quadratic(i, j, A[i, j] + A[j, i])

    # Add linear terms
    bqm.add_linear_from({i: b[i] for i in range(D)})

    model = dynex.BQM(bqm, logging=False)
    sampler = dynex.DynexSampler(model, mainnet=False, logging=False, description='Dynex SDK Super-Resolution')
    sampleset = sampler.sample(num_reads=10000, annealing_time=300)

    return np.array(list(sampleset.first.sample.values()))

def create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo):
    """Creates a batch of QUBOs for parallel submission (e.g., to Dynex)."""
    mu = config.bsc_mu
    alpha = config.bsc_alpha
    D = Dl.shape[1]
    X = Dl
    w = np.ones(D) * mu
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size -1, patch_size - overlap), img_us_width - patch_size -1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size -1, patch_size - overlap), img_us_height - patch_size -1)

    num_patches = len(gridx) * len(gridy)
    num_qubos = int(np.ceil(num_patches / n_patches_per_qubo))
    Q_dicts = [{} for _ in range(num_qubos)]

    count = 0
    for m in range(len(gridx)):
        for n in range(len(gridy)):
            xx, yy = int(gridx[m]), int(gridy[n])
            feat_patch = img_lr_y_feat[yy:yy+patch_size, xx:xx+patch_size, :].flatten('F')
            feat_norm = np.linalg.norm(feat_patch)
            y = feat_patch / feat_norm if feat_norm > 1 else feat_patch

            A = np.linalg.multi_dot([np.diag(w), X.T, X, np.diag(w)])
            b = -2 * np.linalg.multi_dot([np.diag(w), X.T, y])
            b += alpha * w * np.sign(w)

            patch_idx_in_qubo = count % n_patches_per_qubo
            qubo_idx = count // n_patches_per_qubo

            for j in range(D):
                var1 = patch_idx_in_qubo * D + j
                # Linear term
                Q_dicts[qubo_idx][(var1, var1)] = Q_dicts[qubo_idx].get((var1, var1), 0) + A[j, j] + b[j] + 1e-9
                # Quadratic terms
                for k in range(j + 1, D):
                    var2 = patch_idx_in_qubo * D + k
                    Q_dicts[qubo_idx][(var1, var2)] = Q_dicts[qubo_idx].get((var1, var2), 0) + 2 * A[j, k] + 1e-9

            count += 1

    return Q_dicts



# main Super-Resolution function from ScSR.py

def ScSR(img_lr_y, size, upscale, Dh, Dl, lmbd, overlap, quantum_objects=None):
    patch_size = config.patch_size

    img_us = resize(img_lr_y, size)
    img_us_height, img_us_width = img_us.shape
    img_hr = np.zeros(img_us.shape)
    img_hr_entropy = np.zeros(img_us.shape)
    cnt_matrix = np.zeros(img_us.shape)

    img_lr_y_feat = extract_lr_feat(img_us)

    gridx = np.append(create_list_step(0, img_us_width - patch_size - 1, patch_size - overlap), img_us_width - patch_size - 1)
    gridy = np.append(create_list_step(0, img_us_height - patch_size - 1, patch_size - overlap), img_us_height - patch_size - 1)

    count = 0
    cardinality = np.zeros(len(gridx)*len(gridy))

    if config.sc_algo=="qubo_dwave" or config.sc_algo=="qubo_qdeep" or config.sc_algo=="qubo_protes": #Quantum Annealing (Hybrid Solvers), submit to a hybrid solver
        logging.info("running "+config.sc_algo+" in ScSR")
        n_patches_per_qubo = 8
        qubo_size = n_patches_per_qubo*Dl.shape[1]

        create_qubo_start_time = time.time()
        Q_dicts = create_qubo1(img_lr_y, size, Dl, overlap, n_patches_per_qubo)
        logging.info("Create QUBO time: "+str(time.time()-create_qubo_start_time))

        flattened_m = np.zeros(qubo_size*len(Q_dicts))
        logging.info("Number of QUBO problems to solve: %d"%(len(Q_dicts)))

        total_qpu_access_time = 0
        solve_qubo_start_time = time.time()
        for i in range(len(Q_dicts)):
            logging.info("i=%d"%i)
            computation=0
            if config.sc_algo=="qubo_dwave":
                computation=DWaveBqmSolver(Q_dicts[i])
            elif config.sc_algo=="qubo_qdeep":
                computation=QDeepBqmSolver(Q_dicts[i])
            else:
                computation=ProtesBqmSolver(Q_dicts[i])
            #ans1=QDeepBqmSolver(Q_dicts[i])
            #ans2=ProtesBqmSolver(Q_dicts[i])
            #print(ans1['energy'],ans2['energy'])
            flattened_m[i*qubo_size:i*qubo_size+len(computation['configuration'])] = computation['configuration']
        logging.info("Solve QUBO time: "+str(time.time()-solve_qubo_start_time))

    total_opt_time = 0
    for m in tqdm(range(0, len(gridx))):
        for n in range(0, len(gridy)):
            xx = int(gridx[m])
            yy = int(gridy[n])

            us_patch = img_us[yy : yy + patch_size, xx : xx + patch_size]
            us_mean = np.mean(np.ravel(us_patch, order='F'))
            us_patch = np.ravel(us_patch, order='F') - us_mean
            us_norm = np.sqrt(np.sum(np.multiply(us_patch, us_patch)))

            feat_patch = img_lr_y_feat[yy : yy + patch_size, xx : xx + patch_size, :]
            feat_patch = np.ravel(feat_patch, order='F')
            feat_norm = np.sqrt(np.sum(np.multiply(feat_patch, feat_patch)))

            if feat_norm > 1:
                y = np.divide(feat_patch, feat_norm)
            else:
                y = feat_patch

            if config.sc_algo=="sklearn_lasso": #Lasso Regression
                opt_time_start = time.time()
                reg = linear_model.Lasso(alpha=config.lasso_alpha,max_iter=1000)
                reg.fit(Dl,y)
                w = reg.coef_
                total_opt_time += time.time()-opt_time_start
            elif config.sc_algo=="qubo_bsc": #Classical Annealing
                opt_time_start = time.time()
                w = qubo_bsc(Dl,y,alpha=config.bsc_alpha,mu=config.bsc_mu)
                total_opt_time += time.time()-opt_time_start
            elif config.sc_algo=="qubo_dwave" or config.sc_algo=="qubo_qdeep" or config.sc_algo=="qubo_protes": #Quantum Annealing (Hybrid Solvers)
                w = flattened_m[count*Dl.shape[1]:(count+1)*Dl.shape[1]]*config.bsc_mu


            cardinality[count] = np.matmul(np.where(np.abs(w)>0,1,0),np.ones(len(w)))

            gibbs_entropy = 0
            hr_patch = np.dot(Dh, w)

            hr_patch = lin_scale(hr_patch, us_norm)
            hr_patch = np.reshape(hr_patch, (patch_size, -1))
            hr_patch += us_mean

            img_hr[yy : yy + patch_size, xx : xx + patch_size] += hr_patch
            img_hr_entropy[yy : yy + patch_size, xx : xx + patch_size] += gibbs_entropy
            cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] = cnt_matrix[yy : yy + patch_size, xx : xx + patch_size] + 1.0

            count += 1

    logging.info("total_opt_time="+str(total_opt_time))

    index_y,index_x = np.where(cnt_matrix < 1)
    assert len(index_y)==len(index_x)
    for i in range(len(index_y)):
        yy = index_y[i]
        xx = index_x[i]
        img_hr[yy][xx] = img_us[yy][xx]
        cnt_matrix[yy][xx] = 1.0

    img_hr = np.divide(img_hr, cnt_matrix)
    img_hr_entropy = np.divide(img_hr_entropy, cnt_matrix)

    logging.info("avg_cardinality="+str(np.mean(cardinality)))

    return img_hr,img_hr_entropy


from sklearn.metrics import mean_squared_error
from skimage import exposure

def normalize_signal(img, channel,img_lr_ori):
    if np.mean(img[:, :, channel]) * 255 > np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    elif np.mean(img[:, :, channel]) * 255 < np.mean(img_lr_ori[:, :, channel]):
        ratio = np.mean(img_lr_ori[:, :, channel]) / (np.mean(img[:, :, channel]) * 255)
        img[:, :, channel] = np.multiply(img[:, :, channel], ratio)
    return img[:, :, channel]


def normalize_max(img):
    for m in range(img.shape[0]):
        for n in range(img.shape[1]):
            if img[m, n, 0] > 1:
                img[m, n, 0] = 1
            if img[m, n, 1] > 1:
                img[m, n, 1] = 1
            if img[m, n, 2] > 1:
                img[m, n, 2] = 1
    return img
import math
def run_inference():
    # Set active config
    config = C_run
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.val_hr_path, exist_ok=True)
    os.makedirs(config.val_lr_path, exist_ok=True)

    # Setup Logging
    log_format = "%(asctime)s | %(message)s"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p', stream=sys.stdout)
    fh = logging.FileHandler(os.path.join(config.output_dir, 'log_inference.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info(f"--- Starting Super-Resolution Inference using '{config.sc_algo}' ---")

    # 1. Load Dictionaries
    logging.info(f"Loading dictionaries Dh='{config.Dh_path}' and Dl='{config.Dl_path}'")
    try:
        with open(config.Dh_path, 'rb') as f:
            Dh = pickle.load(f)
        with open(config.Dl_path, 'rb') as f:
            Dl = pickle.load(f)
    except FileNotFoundError:
        logging.error("Dictionary files not found. Please run Part 1 to train them or provide correct paths.")
        print("\nERROR: Dictionary files not found. Creating dummy data and folders for a demonstration run.")
        os.makedirs(os.path.dirname(config.Dl_path), exist_ok=True)
        dummy_dict = np.random.rand(25, 128)
        with open(config.Dh_path, 'wb') as f: pickle.dump(dummy_dict, f)
        with open(config.Dl_path, 'wb') as f: pickle.dump(np.random.rand(100, 128), f)
        Dh, Dl = dummy_dict, np.random.rand(100, 128)

    # 2. Load Image
    try:
        lr_img_name = listdir(config.val_lr_path)[0]
        # Try to find a matching HR image, otherwise use the first one
        try:
            hr_img_name = [f for f in listdir(config.val_hr_path) if os.path.splitext(f)[0] in os.path.splitext(lr_img_name)[0]][0]
        except IndexError:
             hr_img_name = listdir(config.val_hr_path)[0]

        lr_path = os.path.join(config.val_lr_path, lr_img_name)
        hr_path = os.path.join(config.val_hr_path, hr_img_name)

        logging.info(f"Loading LR image: {lr_path}")
        img_lr = imread(lr_path) / 255.0
        logging.info(f"Loading HR ground truth: {hr_path}")
        img_hr_ground_truth = imread(hr_path) / 255.0
    except (FileNotFoundError, IndexError):
        logging.warning("Input image not found. Creating a dummy image for demonstration.")
        img_lr = np.random.rand(50, 75, 3)
        img_hr_ground_truth = np.random.rand(150, 225, 3)
        imsave(os.path.join(config.val_lr_path, "dummy.png"), (img_lr * 255).astype(np.uint8))
        imsave(os.path.join(config.val_hr_path, "dummy_hr.png"), (img_hr_ground_truth * 255).astype(np.uint8))


    # 3. Prepare Image for Processing
    size_hr = (img_lr.shape[0] * config.upscale, img_lr.shape[1] * config.upscale)
    img_lr_ycbcr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr_ycbcr[:, :, 0]
    img_lr_cb = resize(img_lr_ycbcr[:, :, 1], size_hr, anti_aliasing=True)
    img_lr_cr = resize(img_lr_ycbcr[:, :, 2], size_hr, anti_aliasing=True)

    # 4. Run Super-Resolution Algorithm on Y channel
    logging.info("Starting ScSR algorithm...")
    start_time = time.time()
    img_hr_y, _ = ScSR(img_lr_y, size_hr, config.upscale, Dh, Dl, config.lmbd, config.overlap)
    logging.info(f"ScSR finished in {time.time() - start_time:.2f} seconds.")

    # 5. Apply Back-projection
    logging.info("Applying back-projection...")
    img_hr_y = backprojection(img_hr_y, img_lr_y)

    # 6. Reconstruct Final HR Image
    img_hr_final_ycbcr = np.stack([img_hr_y, img_lr_cb, img_lr_cr], axis=2)
    img_hr_final = ycbcr2rgb(img_hr_final_ycbcr)
    img_hr_final = np.clip(img_hr_final, 0, 1)

    result_path = os.path.join('out', f"result_{config.sc_algo}.png")
    imsave(result_path, (img_hr_final * 255).astype(np.uint8))
    logging.info(f"Result saved to {result_path}")

    # 7. Display Results
    img_bicubic = resize(img_lr, size_hr, anti_aliasing=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_lr)
    axes[0].set_title('Original Low-Res')
    axes[0].axis('off')

    axes[1].imshow(img_bicubic)
    axes[1].set_title('Bicubic Upscaled')
    axes[1].axis('off')

    axes[2].imshow(img_hr_final)
    axes[2].set_title(f'Result ({config.sc_algo})')
    axes[2].axis('off')

    axes[3].imshow(img_hr_ground_truth)
    axes[3].set_title('Ground Truth High-Res')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    maxIter=100

    noisy_dir=os.path.join('out', f"result_{config.sc_algo}.png")
    signal_dir=hr_path

    img_lr = imread( noisy_dir )
    logging.info("img_lr shape: "+str(img_lr.shape))
    # Read and save ground truth image
    img_hr = imread( signal_dir )
    logging.info("img_hr shape: "+str(img_hr.shape))

    img_hr_y = rgb2ycbcr(img_hr)[:, :, 0]
    # Change color space
    img_lr_ori = img_lr
    temp = img_lr
    img_lr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr[:, :, 0]
    img_lr_cb = img_lr[:, :, 1]
    img_lr_cr = img_lr[:, :, 2]
    # Upscale chrominance to color SR images
    img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
    img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)
    # Super Resolution via Sparse Representation
    start_time = time.time()
    img_sr_y,img_sr_unc = ScSR(img_lr_y, img_hr_y.shape, C_train.upscale, Dh, Dl, C_train.lmbd, config.overlap)
    logging.info("ScSR time: "+str(time.time()-start_time))
    img_sr_y = backprojection(img_sr_y, img_lr_y, maxIter)
    img_sr_hmatched_y = exposure.match_histograms(image=img_sr_y,reference=img_lr_y,channel_axis=None)
    # Bicubic interpolation for reference
    img_bc = resize(img_lr_ori, (img_hr.shape[0], img_hr.shape[1]))
    #imsave(os.path.join(*[config.output_dir,'%04d_1bicubic.png'%i]), img_bc)
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]
    #########################################################################################
    # Compute RMSE for the illuminance
    rmse_bc_hr = np.sqrt(mean_squared_error(img_hr_y, img_bc_y))
    rmse_bc_hr = np.zeros((1,)) + rmse_bc_hr
    rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
    rmse_sr_hr = np.zeros((1,)) + rmse_sr_hr
    rmse_srhm_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_hmatched_y))
    rmse_srhm_hr = np.zeros((1,)) + rmse_srhm_hr
    logging.info('bicubic RMSE: '+str(rmse_bc_hr))
    logging.info('SR RMSE: '+str(rmse_sr_hr))
    logging.info('SR Histogram-matched RMSE: '+str(rmse_srhm_hr))
    y_psnr_bc_hr = 20*math.log10(255.0/rmse_bc_hr)
    y_psnr_sr_hr = 20*math.log10(255.0/rmse_sr_hr)
    y_psnr_srhm_hr = 20*math.log10(255.0/rmse_srhm_hr)
    logging.info('bicubic Y-Channel PSNR: '+str(y_psnr_bc_hr))
    logging.info('SR Y-Channel PSNR: '+str(y_psnr_sr_hr))
    logging.info('SRHM Y-Channel PSNR: '+str(y_psnr_srhm_hr))
    #########################################################################################
    # Create colored SR images
    img_sr = np.stack((img_sr_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr = ycbcr2rgb(img_sr)
    # Signal normalization
    for channel in range(img_sr.shape[2]):
        img_sr[:, :, channel] = normalize_signal(img_sr, channel,img_lr_ori)
    # Maximum pixel intensity normalization
    img_sr = normalize_max(img_sr)
    #imsave(os.path.join(*[config.output_dir,'%04d_2SR.png'%i]), img_sr)
    img_sr_hmatched = np.stack((img_sr_hmatched_y, img_sr_cb, img_sr_cr), axis=2)
    img_sr_hmatched = ycbcr2rgb(img_sr_hmatched)
    #imsave(os.path.join(*[config.output_dir,'%04d_2SRHM.png'%i]), img_sr_hmatched)
    img_sr_final = (np.clip(img_sr_hmatched,0,1)*255).astype(np.uint8)
    #imsave(os.path.join(*[config.output_dir,'%04d_2SR_final.png'%i]), img_sr_final)




def calc_psnr(img_name):
    config=C_run
    logging.info(f"--- Starting Super-Resolution Inference using '{config.sc_algo}' ---")

    D_size = 128
    US_mag = 3
    lmbd = config.lmbd
    patch_size = config.patch_size

    # Set which dictionary you want to use
    with open(config.Dh_path, 'rb') as f:
        Dh = pickle.load(f)
    Dh = normalize(Dh)
    with open(config.Dl_path, 'rb') as f:
        Dl = pickle.load(f)
    Dl = normalize(Dl)
    img_lr_dir = config.val_lr_path
    img_hr_dir = config.val_hr_path
    overlap = config.overlap
    upscale = 3
    maxIter = 100
    img_type = '.png'
    logging.info("image name: "+img_name)
    # Read test image

    img_name_dir = list(img_name)
    img_name_dir = np.delete(np.delete(np.delete(np.delete(img_name_dir, -1), -1), -1), -1)
    img_name_dir = ''.join(img_name_dir)
    img_lr = imread( os.path.join(*[img_lr_dir, img_name]) )
    logging.info("img_lr shape: "+str(img_lr.shape))

    # Read and save ground truth image
    img_hr = imread( os.path.join(*[img_hr_dir, img_name]) )
    logging.info("img_hr shape: "+str(img_hr.shape))
    #imsave(os.path.join(*[config.output_dir,"%04d_3HR.png"%i]), img_hr)
    img_hr_y = rgb2ycbcr(img_hr)[:, :, 0]

    # Change color space
    img_lr_ori = img_lr
    temp = img_lr
    img_lr = rgb2ycbcr(img_lr)
    img_lr_y = img_lr[:, :, 0]
    img_lr_cb = img_lr[:, :, 1]
    img_lr_cr = img_lr[:, :, 2]

    # Upscale chrominance to color SR images
    img_sr_cb = resize(img_lr_cb, (img_hr.shape[0], img_hr.shape[1]), order=0)
    img_sr_cr = resize(img_lr_cr, (img_hr.shape[0], img_hr.shape[1]), order=0)

    # Super Resolution via Sparse Representation
    start_time = time.time()
    img_sr_y,img_sr_unc = ScSR(img_lr_y, img_hr_y.shape, upscale, Dh, Dl, lmbd, overlap)
    logging.info("ScSR time: "+str(time.time()-start_time))
    img_sr_y = backprojection(img_sr_y, img_lr_y, maxIter)
    img_sr_hmatched_y = exposure.match_histograms(image=img_sr_y,reference=img_lr_y,channel_axis=None)

    # Bicubic interpolation for reference
    img_bc = resize(img_lr_ori, (img_hr.shape[0], img_hr.shape[1]))
    #imsave(os.path.join(*[config.output_dir,'%04d_1bicubic.png'%i]), img_bc)
    img_bc_y = rgb2ycbcr(img_bc)[:, :, 0]

##########################################################################################

    # Compute RMSE for the illuminance
    rmse_bc_hr = np.sqrt(mean_squared_error(img_hr_y, img_bc_y))
    rmse_bc_hr = np.zeros((1,)) + rmse_bc_hr
    rmse_sr_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_y))
    rmse_sr_hr = np.zeros((1,)) + rmse_sr_hr
    rmse_srhm_hr = np.sqrt(mean_squared_error(img_hr_y, img_sr_hmatched_y))
    rmse_srhm_hr = np.zeros((1,)) + rmse_srhm_hr

    logging.info('bicubic RMSE: '+str(rmse_bc_hr))
    logging.info('SR RMSE: '+str(rmse_sr_hr))
    logging.info('SR Histogram-matched RMSE: '+str(rmse_srhm_hr))

    y_psnr_bc_hr = 20*math.log10(255.0/rmse_bc_hr)
    y_psnr_sr_hr = 20*math.log10(255.0/rmse_sr_hr)
    y_psnr_srhm_hr = 20*math.log10(255.0/rmse_srhm_hr)
    logging.info('bicubic Y-Channel PSNR: '+str(y_psnr_bc_hr))
    logging.info('SR Y-Channel PSNR: '+str(y_psnr_sr_hr))
    logging.info('SRHM Y-Channel PSNR: '+str(y_psnr_srhm_hr))

# Run the main function

algos= ["qubo_dwave","qubo_qdeep","qubo_protes","sklearn_lasso"]
#run_inference()
for algo in algos:
    C_run.sc_algo=algo
    calc_psnr('000000.png')
    
#notes:
#search for the comment #PATH  , there you can change the path for your photos
#enter the name of your photo in the call of calc_psnr



