import numpy as np

from .givens      import *
from .householder import *

def bidiagonalize( A: np.ndarray, inplace=True, compute_uv=False ):
    assert( A.shape[0] == A.shape[1] )
    A = A if inplace else A.copy()
    M,N = A.shape
    U,V = (np.eye(M),np.eye(N)) if compute_uv else (None,None)
    for i in range(0,M-1):
        HL = Householder(A[i:,i])
        A[i:,i:] = HL@A[i:,i:]
        HR = Householder(A[i,i+1:])
        A[i:,i+1:] = A[i:,i+1:]@HR

        if compute_uv:
            U[i:,:]   = HL@U[i:,:]
            V[:,i+1:] = V[:,i+1:]@HR

    return (A,U,V) if compute_uv else A

def hessenberg_sym( M: np.ndarray, inplace=True, compute_u=False ):
    assert( M.shape[0] == M.shape[1] )
    A = M if inplace else M.copy()
    m,n = M.shape
    U = np.eye(m) if compute_u else None
    for i in range(m-2):
        H = Householder( A[i+1:,i] )
        A[i+1:,i:] = H@A[i+1:,i:]
        A[i:,i+1:] = A[i:,i+1:]@H
        if compute_u:
            U[i+1:,:] = H@U[i+1:,:]
    return (A,U) if compute_u else A

def hessenberg_qr( H: np.ndarray, inplace=True, compute_u=False ):
    '''https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf'''
    assert( H.shape[0] == H.shape[1] )
    H = H if inplace else H.copy()
    m,n = H.shape
    U = np.eye(m) if compute_u else None
    for i in range(m-1):
        G = Givens( H[i,i], H[i+1,i], i, i+1 )
        H = G@H
        H = H@G.T
        if compute_u:
            U = U@G.T
    return (H,U) if compute_u else H
    
    