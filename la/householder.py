import numpy as np

class Householder:
    __array_priority__ = 10000

    def __init__( self, a ):
        v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
        v[0] = 1
        tau = (2 / (v.T @ v))**0.5
        self.v = v*tau

    @property
    def vector( self ):
        return self.v.copy()

    @property
    def matrix( self ):
        return np.eye(self.v.size) - self.v[:,None]*self.v[None,:]

    @property
    def ndim( self ):
        return 2

    @property
    def shape( self ):
        return (self.v.size,self.v.size)

    @property
    def T( self ):
        return self

    def __matmul__( self, M: np.ndarray ):
        return M - self.v[:,None]@(self.v[None,:]@M)

    def __rmatmul__( self, M: np.ndarray ):
        return M - (M@self.v[:,None])@self.v[None,:]   
