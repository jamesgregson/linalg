import numpy as np

from .givens        import *
from .householder   import *
from .hessenberg    import *

def eigvalsh( A: np.ndarray, num_iterations=100 ):
    assert( A.shape[0] == A.shape[1] )

    # initial stage, reduce to tridiagonal form
    h,u_new = hessenberg_sym( A )
    U = u_new

    # basic QR iteration with eigenvectors
    for iter in range(num_iterations):
        h,u_new = hessenberg_qr( h )
        U = U@u_new
    
    return np.diag(h), U