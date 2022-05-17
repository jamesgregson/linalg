from .givens      import *
from .householder import *
from .hessenberg  import *

from .qr          import *



def _tridiagonalize( A: np.ndarray, inplace=True ):
    assert( A.shape[0] == A.shape[1] )
    A = A if inplace else A.copy()
    M = A.shape[0]
    for i in range(0,M-1):
        H = Householder(A[i+1:,i])
        A[i+1:,i:] = H@A[i+1:,i:]
        A[i:,i+1:] = A[i:,i+1:]@H
    return A

def _bidiagonalize( A: np.ndarray, inplace=True ):
    assert( A.shape[0] == A.shape[1] )
    A = A if inplace else A.copy()
    M,N = A.shape
    for i in range(0,M-1):
        HL = Householder(A[i:,i])
        A[i:,i:] = HL@A[i:,i:]
        HR = Householder(A[i,i+1:])
        A[i:,i+1:] = A[i:,i+1:]@HR
    return A
