import numpy as np
import matplotlib.pyplot as plt

def Part_dervs(func, params, eps=1e-8):
    part_der = []
    for i, p in enumerate(params):
        diff_pars = params.copy()
        diff_pars[i] += eps
        part_der.append((func(*diff_pars) - func(*params))/eps)
    return part_der





