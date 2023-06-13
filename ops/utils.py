import torch 
import numpy as np

def one_hot_sparse(A,num_classes,value=None):
    A = A.int()
    B = torch.arange(A.shape[0]).to(A.device)
    if value==None:
        C = torch.ones_like(B)
    else:
        C = value
    return torch.sparse_coo_tensor(torch.stack([B,A]),C,size=(A.shape[0],num_classes))


def Histogram_equalization(x, bins=1000):
    x = (x - x.min())/(x.max()-x.min())
    hist, bins = np.histogram(x, bins, [0,1])
    cdf = hist.cumsum()
    cdf_normalized = cdf*float(hist.max())/cdf.max()
    x_equalized = np.interp(x, bins[:-1], cdf_normalized)
    return x_equalized