import numpy as np

def compute_eisenberg_noe_clearing(
    liabilities: np.ndarray, 
    external_assets: np.ndarray, 
    max_iter: int = 1000, 
    tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the unique Eisenberg-Noe clearing payment vector p* in a financial network.
    """
    n_nodes = liabilities.shape[0]
    total_obligations = liabilities.sum(axis=1)
    
    pi_matrix = np.zeros((n_nodes, n_nodes))
    nonzero_mask = total_obligations > 0
    pi_matrix[nonzero_mask] = liabilities[nonzero_mask] / total_obligations[nonzero_mask, None]
    
    p_current = total_obligations.copy()
    
    for _ in range(max_iter):
        inflows = np.dot(pi_matrix.T, p_current)
        p_next = np.minimum(total_obligations, np.maximum(0.0, inflows + external_assets))
        
        if np.linalg.norm(p_next - p_current, ord=1) < tol:
            break
        p_current = p_next

    default_status = p_current < (total_obligations - tol)
    return p_current, default_status
