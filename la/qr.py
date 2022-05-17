import numpy as np

from .givens import *

def qr_tri( A, inplace=True ):
    '''QR factorization of a tridiagonal matrix
    '''
    assert( A.shape[0] == A.shape[1] )
    A = A if inplace else A.copy()
    m,n = A.shape
    Q = np.eye(m)
    for i in range(m-1):
        G = Givens( A[i,i], A[i+1,i], i, i+1 )
        A = G@A
        Q = Q@G.T

    return Q,A
