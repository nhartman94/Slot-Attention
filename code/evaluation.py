import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import torch
import torch.nn.functional as F

def student_t(Y_pred_sorted, Y_true_sorted, n_rings=2, bs=1000):
    fig, axs = plt.subplots(1, 3,figsize=(12 ,3))
    #fig.suptitle(title)
    div =  (Y_pred_sorted-Y_true_sorted).reshape(bs*n_rings, 3).T
    labels = [r"$x$ resolution", r"$y$ resolution", r"$R$ resolution"]

    fit_params = np.zeros((3,3))
    for i, (ax, label) in enumerate(zip(axs[0:], labels)):
        # fit with student t
        df, loc, scale = t.fit(div[i])
        fit_params[i, 0] = df
        fit_params[i, 1] = loc
        fit_params[i, 2] = scale
        label_fit= f" df={df:.3},\n mean={loc:.3},\n scale={scale:.3}"
        # plot
        n_counts, containers, patches = ax.hist(div[i], bins=100, density=True) #desity=True to return a probability density!
        x_min = np.amax(np.array([containers[0], -0.25]))
        x_max = np.amin(np.array([containers[-1], 0.25]))
        x = np.linspace(x_min, x_max, 200)
        ax.plot(x, t.pdf(x, df, loc=loc, scale=scale), '--', label=' t-distribution \n'+ label_fit)
        ax.set_xlabel(label)
        ax.set_ylabel("counts")
        ax.legend()
    plt.tight_layout()
    plt.show()
    return fit_params
    
def KL_divergence(slots_sorted, rings_sorted):
    l_kl = F.kl_div(torch.log(slots_sorted),rings_sorted,reduction='none').sum(axis=1).mean(axis=-1)
    l_kl = np.nan_to_num(l_kl)
    return np.mean(l_kl)