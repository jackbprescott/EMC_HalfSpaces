import numpy as np
import numpy.linalg as la

# import matplotlib as mp
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.mplot3d import Axes3D

import torchvision.datasets as datasets

from sklearn.covariance import EmpiricalCovariance

### Data loader

def load_data(dataset, size=70000, dim=784, tr_ratio=0.5, seed=1):
    np.random.seed(seed)

    if dataset == 'norm':
        norm_tr = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=int(tr_ratio * size))
        norm_ts = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.identity(dim), size=int((1 - tr_ratio) * size))
        return norm_tr, norm_ts

    elif dataset == 'mnist':
        mnist_tr = datasets.MNIST('./datalab', train = True, download = True).data.numpy().reshape(-1, 28*28) / 255
        mnist_ts = datasets.MNIST('./datalab', train = False, download = True).data.numpy().reshape(-1, 28*28) / 255
        mnist = np.ma.concatenate([mnist_tr, mnist_ts])
        np.random.shuffle(mnist)
        idx = int(tr_ratio * len(mnist))
        return mnist[:idx], mnist[idx:]

    elif dataset == 'cifar':
        cifar_tr = datasets.CIFAR10('./datalab', train = True, download = True).data.reshape(-1, 32*32*3) / 255
        cifar_ts = datasets.CIFAR10('./datalab', train = False, download = True).data.reshape(-1, 32*32*3) / 255
        cifar = np.ma.concatenate([cifar_tr, cifar_ts])
        np.random.shuffle(cifar)
        idx = int(tr_ratio * len(cifar))
        return cifar[:idx], cifar[idx:]

    else:
        raise('Invalid dataset')


### General utility functions

def l2_dists(X, pc):
    # Make sure l1-norm of PC is 1, so that the dot product is also the l1-norm to the hyperplane
    # orthogonal to pc and passing through the origin
    pc /= la.norm(pc, ord=2)
    X_pc = np.dot(X, pc)
    X_pc.sort()
    return X_pc


def linf_dists(X, pc):
    # Make sure l1-norm of PC is 1, so that the dot product is also the linf-norm to the hyperplane
    # orthogonal to pc and passing through the origin
    pc /= la.norm(pc, ord=1)
    X_pc = np.dot(X, pc)
    X_pc.sort()
    return X_pc


def pc_pow(pc, exponent, negligible=1e-5):
    pc = np.sign(pc) * np.power(np.abs(pc), exponent)
    pc /= la.norm(pc, ord=1)
    pc[np.abs(pc) <= negligible] = 0
    return pc / la.norm(pc, ord=1)


def get_pc(data):
    X = data.reshape(data.shape[0], -1)
    K = EmpiricalCovariance().fit(X).covariance_
    w, v = la.eig(K)
    return v[:,np.flip(np.argsort(w))]


def extract_params(train_output, pc_mtx):
    pc = pc_mtx.T[train_output['PC']]
    exp = train_output['Exponent']
    risk_boundary = train_output['Risk boundary']
    fwd = train_output['Fwd']
    return pc, exp, risk_boundary, fwd


### Functions to measure adversarial risk for a specific hyperplane

def measure_adv_risk_train(X_pc, alpha, eps, fwd):
    adv_risk_count = 0
    risk_boundary_idx = int(len(X_pc) * alpha)

    if fwd:
        risk_boundary = X_pc[risk_boundary_idx]
        adv_risk_boundary = risk_boundary + eps

        for i in range(risk_boundary_idx + 1, len(X_pc)):
            if X_pc[i] <= adv_risk_boundary:
                adv_risk_count += 1
            else:
                break
    else:
        X_pc = np.flip(X_pc)
        risk_boundary = X_pc[risk_boundary_idx]
        adv_risk_boundary = risk_boundary - eps

        for i in range(risk_boundary_idx + 1, len(X_pc)):
            if X_pc[i] >= adv_risk_boundary:
                adv_risk_count += 1
            else:
                break

    return (risk_boundary_idx + 1) / len(X_pc), (risk_boundary_idx + 1 + adv_risk_count) / len(X_pc), risk_boundary


def measure_adv_risk_test(X_pc, alpha, eps, fwd, risk_boundary):
    risk_count = 0
    adv_risk_count = 0

    if fwd:
        adv_risk_boundary = risk_boundary + eps

        for x in X_pc:
            if x <= adv_risk_boundary:
                adv_risk_count += 1
                if x <= risk_boundary:
                    risk_count += 1
            else:
                break
    else:
        adv_risk_boundary = risk_boundary - eps

        for x in np.flip(X_pc):
            if x >= adv_risk_boundary:
                adv_risk_count += 1
                if x >= risk_boundary:
                    risk_count += 1
            else:
                break

    return risk_count / len(X_pc), adv_risk_count / len(X_pc)


def adv_risk_train(pc, X, alpha, eps, norm, title=None):
    if norm == "l2":
        X_pc = l2_dists(X, pc)
    elif norm == "linf":
        X_pc = linf_dists(X, pc)
    else:
        print("Error: Unkown norm specified")
        return

    fwd_risk, fwd_adv_risk, fwd_risk_boundary = measure_adv_risk_train(X_pc, alpha, eps, fwd=True)
    bwd_risk, bwd_adv_risk, bwd_risk_boundary = measure_adv_risk_train(X_pc, alpha, eps, fwd=False)

    if fwd_adv_risk <= bwd_adv_risk:
        fwd = True
        risk = fwd_risk
        adv_risk = fwd_adv_risk
        risk_boundary = fwd_risk_boundary
    else:
        fwd = False
        risk = bwd_risk
        adv_risk = bwd_adv_risk
        risk_boundary = bwd_risk_boundary

    return fwd, risk_boundary, np.round(risk, 5), np.round(adv_risk, 5)


# Functions to search for optimal parameters (principal component index and corresponding exponent) to
# minimize adversarial risk, and to evaluate on test set

def train(pc_mtx, X_tr, alpha, eps, norm, adj=0.005):
    best_risk = 1
    best_adv_risk = 1
    best_pc = -1
    best_exp = 1
    best_risk_boundary = 0
    best_fwd = True
    pc_bests = []
    
    alpha += adj

    for i, pc in enumerate(pc_mtx.T):
        if i % 50 == 0:
            print(f"On PC {i}")
      #  print("PC: {}".format(i))
        pc_best_exp = 1
        pc_best_fwd, pc_best_risk_boundary, pc_best_risk, pc_best_adv_risk = adv_risk_train(pc, X_tr, alpha=alpha, eps=eps, norm=norm)
        
        fwd, risk_boundary, risk, adv_risk = adv_risk_train(pc_pow(pc, pc_best_exp+10), X_tr, alpha=alpha, eps=eps, norm=norm)
        
        while adv_risk < pc_best_adv_risk:
            pc_best_exp += 10
            if np.isnan(pc_pow(pc, pc_best_exp+10)).any():
                break
            pc_best_fwd, pc_best_risk_boundary, pc_best_risk, pc_best_adv_risk = fwd, risk_boundary, risk, adv_risk
            fwd, risk_boundary, risk, adv_risk = adv_risk_train(pc_pow(pc, pc_best_exp+10), X_tr, alpha=alpha, eps=eps, norm=norm)

        if pc_best_exp == 1:
            refined_range = np.linspace(1.5, 10.5, 19)
        else:
            refined_range = np.concatenate([np.linspace(pc_best_exp - 9, pc_best_exp - 1, 9), np.linspace(pc_best_exp + 1, pc_best_exp + 9, 9)])

        for exp_rf in refined_range:
            if np.isnan(pc_pow(pc, exp_rf)).any():
                break
            fwd, risk_boundary, risk, adv_risk = adv_risk_train(pc_pow(pc, exp_rf), X_tr, alpha=alpha, eps=eps, norm=norm)
            
            if adv_risk < pc_best_adv_risk:
                pc_best_fwd, pc_best_risk_boundary, pc_best_risk, pc_best_adv_risk = fwd, risk_boundary, risk, adv_risk
                pc_best_exp = exp_rf
                    
        # Now, after trying all the exponents, if the best achieved adv risk is better than the overall best, replace the overall best values with this newly achieved best        
        if pc_best_adv_risk < best_adv_risk and pc_best_risk >= alpha:
            best_fwd, best_risk_boundary, best_risk, best_adv_risk = pc_best_fwd, pc_best_risk_boundary, pc_best_risk, pc_best_adv_risk
            best_pc = i
            best_exp = pc_best_exp
            #
            #  plt.title(f"New Best PC: {i}")
            #
            #  if norm == "l2":
            #      X_pc = l2_dists(X_tr, pc_pow(pc_mtx.T[i], pc_best_exp))
            #  elif norm == "linf":
            #      X_pc = linf_dists(X_tr, pc_pow(pc_mtx.T[i], pc_best_exp))
            #  else:
            #      print("Error: Unkown norm specified")
            #      return
            #
            #  _ = plt.hist(X_pc, bins=50)
            #  plt.axvline(best_risk_boundary, color='k', linestyle='dashed', linewidth=1)
            #  if fwd:
            #      plt.axvline(best_risk_boundary + eps, color='k', linestyle='dashed', linewidth=1)
            #      print("Adversarial risk region: {} to {}, includes {} points".format(best_risk_boundary, best_risk_boundary + eps, np.round(best_adv_risk * len(X_pc))))
            #  else:
            #      plt.axvline(best_risk_boundary - eps, color='k', linestyle='dashed', linewidth=1)
            #      print("Adversarial risk region: {} to {}, includes {} points".format(best_risk_boundary - eps, best_risk_boundary, np.round(best_adv_risk * len(X_pc))))
            #  plt.ylim(top=2500)
            #  plt.show()
            
            #  print("New PC {} Best:".format(i))
            #  print({
            #      "Risk": pc_best_risk,
            #      "Adv risk": pc_best_adv_risk,
            #      "Exponent": pc_best_exp
            #     })
    
    #print("Done!")
            
    return ({
            "PC": best_pc,
            "Exponent": best_exp,
            "Risk": best_risk,
            "Adv risk": best_adv_risk,
            "Risk boundary": best_risk_boundary,
            "Fwd": best_fwd
            }, pc_bests)


def test(pc, X, alpha, eps, fwd, risk_boundary, norm):
    if norm == "l2":
        X_pc = l2_dists(X, pc)
    elif norm == "linf":
        X_pc = linf_dists(X, pc)
    else:
        print("Error: Unkown norm specified")
        return
    
    risk, adv_risk = measure_adv_risk_test(X_pc, alpha, eps, fwd, risk_boundary)
    
    #  plt.title("Test Results")
    #  _ = plt.hist(X_pc, bins=50)
    #  plt.axvline(risk_boundary, color='k', linestyle='dashed', linewidth=1)
    #  if fwd:
    #      plt.axvline(risk_boundary + eps, color='k', linestyle='dashed', linewidth=1)
    #      #print("Adversarial risk region: {} to {}, includes {} points".format(risk_boundary, risk_boundary + eps, adv_risk * len(X)))
    #  else:
    #      plt.axvline(risk_boundary - eps, color='k', linestyle='dashed', linewidth=1)
    #      #print("Adversarial risk region: {} to {}, includes {} points".format(risk_boundary - eps, risk_boundary, adv_risk * len(X)))
    #  plt.ylim(top=2500)
    #  plt.show()

    return np.round(risk, 5), np.round(adv_risk, 5)


def evaluate(tr, ts, alpha, eps, num_pc=-1):
    pc = get_pc(tr)
    if num_pc >= 0:
        pc = pc[:,:num_pc]
    best, pc_bests = train(pc[:,:num_pc], tr, alpha, eps, "linf", adj=0.5/np.sqrt(len(tr)))
    best_pc, exp, risk_boundary, fwd = extract_params(best, pc)
    test_risk, test_adv_risk = test(pc_pow(best_pc, exp), ts, alpha, eps, fwd, risk_boundary, 'linf')
    print(f'Test risk: {test_risk},\tAdv test risk: {test_adv_risk}')
    return test_risk, test_adv_risk

