import numpy as np

class Givens:
    __array_priority__ = 10000

    def __init__( self, a, b, p, q ):
        self.p = p
        self.q = q
        if( np.abs(b) > np.abs(a) ):
            t = a/b
            self.s = np.copysign(1.0,b)/np.sqrt(1.0+t**2)
            self.c = self.s*t
        else:
            t = b/a
            self.c = np.copysign(1.0,a)/np.sqrt(1.0+t**2)
            self.s = self.c*t

    @property
    def T( self ):
        g = Givens(1,1,0,1)
        g.p,g.q,g.c,g.s = self.q,self.p,self.c,self.s
        return g

    def __matmul__( self, M: np.ndarray ):
        O = M.copy()
        O[self.p,:] =  self.c*M[self.p,:] + self.s*M[self.q,:]
        O[self.q,:] = -self.s*M[self.p,:] + self.c*M[self.q,:]
        return O

    def __rmatmul__( self, M: np.ndarray ):
        O = M.copy()
        O[:,self.p] =  self.c*M[:,self.p] - self.s*M[:,self.q]
        O[:,self.q] =  self.s*M[:,self.p] + self.c*M[:,self.q]
        return O 