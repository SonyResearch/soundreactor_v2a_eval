
import torch
_SIGMA = 10
_SCALE = 1000

def compute_mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)

def _compute_squared_distances(x1, x2):
    x1_sqnorms = torch.sum(x1 * x1, dim=1)
    x2_sqnorms = torch.sum(x2 * x2, dim=1)
    
    # x1: (b1, d), x2: (b2, d) -> matmul: (b1, b2)
    # x1_sqnorms.unsqueeze(1): (b1, 1)
    # x2_sqnorms.unsqueeze(0): (1, b2)
    return -2 * torch.matmul(x1, x2.T) + x1_sqnorms.unsqueeze(1) + x2_sqnorms.unsqueeze(0)

def _compute_symmetric_kernel_sum(X, gamma, batch_size, device):
    n = X.shape[0]
    total_sum = 0.0
    
    for i in range(0, n, batch_size):
        x_batch_i = X[i : i + batch_size]
        
        dists_sq_diag = _compute_squared_distances(x_batch_i, x_batch_i)
        total_sum += torch.exp(-gamma * dists_sq_diag).sum()
        
        for j in range(i + batch_size, n, batch_size):
            x_batch_j = X[j : j + batch_size]
            dists_sq_off_diag = _compute_squared_distances(x_batch_i, x_batch_j)
            total_sum += 2 * torch.exp(-gamma * dists_sq_off_diag).sum()
            
    return total_sum

def _compute_kernel_sum(X, Y, gamma, batch_size, device):
    n = X.shape[0]
    m = Y.shape[0]
    total_sum = 0.0
    
    for i in range(0, n, batch_size):
        x_batch_i = X[i : i + batch_size]
        for j in range(0, m, batch_size):
            y_batch_j = Y[j : j + batch_size]
            dists_sq = _compute_squared_distances(x_batch_i, y_batch_j)
            total_sum += torch.exp(-gamma * dists_sq).sum()
            
    return total_sum

def compute_mmd_optimized(x, y, batch_size=(5096*2), device='cuda' if torch.cuda.is_available() else 'cpu'):
    n = x.shape[0]
    m = y.shape[0]
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    
    gamma = 1.0 / (2 * _SIGMA**2)
    
    sum_k_xx = _compute_symmetric_kernel_sum(x, gamma, batch_size, device)
    sum_k_yy = _compute_symmetric_kernel_sum(y, gamma, batch_size, device)
    
    sum_k_xy = _compute_kernel_sum(x, y, gamma, batch_size, device)
    
    k_xx = sum_k_xx / (n * n)
    k_yy = sum_k_yy / (m * m)
    k_xy = sum_k_xy / (n * m)
    return _SCALE * (k_xx + k_yy - 2 * k_xy)