import numpy as np

from .givens      import *
from .householder import *

def bidiagonalize( A: np.ndarray ):
    assert( A.shape[0] == A.shape[1] )
    A = A.copy()
    M,N = A.shape
    U,V = (np.eye(M),np.eye(N))
    for i in range(0,M-1):
        HL = Householder(A[i:,i])
        A[i:,i:] = HL@A[i:,i:]
        HR = Householder(A[i,i+1:])
        A[i:,i+1:] = A[i:,i+1:]@HR

        U[i:,:]   = HL@U[i:,:]
        V[:,i+1:] = V[:,i+1:]@HR

    return A,U,V

def hessenberg_sym( A: np.ndarray ):
    '''Decomposes M = U@H@U.T via Householder reflections'''
    assert( A.shape[0] == A.shape[1] )
    A = A.copy()
    m,n = A.shape
    U = np.eye(m)
    for i in range(m-2):
        H = Householder( A[i+1:,i] )
        A[i+1:,i:] = H@A[i+1:,i:]
        A[i:,i+1:] = A[i:,i+1:]@H

        U[:,i+1:] = U[:,i+1:]@H
    return A,U

def hessenberg_qr( H: np.ndarray ):
    '''https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf'''
    assert( H.shape[0] == H.shape[1] )
    H = H.copy()
    m,n = H.shape
    U = np.eye(m)
    for i in range(m-1):
        G = Givens( H[i,i], H[i+1,i], i, i+1 )
        H = G@H
        H = H@G.T

        U = U@G.T
    return H,U
    
    