import numpy as np
from tools.rsa import compute_rsm
from tools.cka import center_kernel
from tools.stats import apply_transform

def group_level_rsa(data: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "linear":
        print("Standardizing data")
        data = [apply_transform(d, "standardize") for d in data]
    per_subject_rsms = compute_rsm_per_subjects(data, metric=metric)
    if metric == "cosine":
        group_level_rsm =  average_cosine_rsms(per_subject_rsms)
    elif metric == "pearson":
        group_level_rsm = average_pearson_rsm(per_subject_rsms)
        
    elif metric == "linear":
        group_level_rsm = average_linear_rsm(per_subject_rsms)

    else:
        raise ValueError(f"Metric {metric} not supported")

    return group_level_rsm, per_subject_rsms
    

def compute_rsm_per_subjects(subjects: list[np.ndarray], metric: str = "cosine") -> np.ndarray:
    return [compute_rsm(s, metric=metric) for s in subjects]

def average_cosine_rsms(rsms: list[np.ndarray]) -> np.ndarray:
    return np.mean(rsms, axis=0)

def average_linear_rsm(rsms: list[np.ndarray]) -> np.ndarray:
    return np.mean(rsms, axis=0)


def fisher_z(r, epsilon=1e-6):
    r = np.clip(r, -1 + epsilon, 1 - epsilon)
    return np.arctanh(r)

def inverse_fisher_z(z):
    return np.tanh(z)

def average_pearson_rsm(rsms):
    z_list = [fisher_z(rsm) for rsm in rsms]
    
    # Compute the element-wise average of the transformed matrices.
    avg_z = np.mean(z_list, axis=0)
    
    # Convert the averaged z-values back to the original similarity scale.
    avg_rsm = inverse_fisher_z(avg_z)
    return avg_rsm



