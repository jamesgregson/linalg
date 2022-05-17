import unittest
import numpy as np

import la

def test_hessenberg_sym():
    # create a symmetric random matrix
    A = np.random.standard_normal((5,5))
    A = A.T + A

    # tridiagonalize it via householder reflections
    H,U = la.hessenberg_sym( A )

    # check result
    assert( np.allclose( U@H@U.T, A ) )
    print('test_hessenberg_sym passed.')

def test_hessenberg_qr():
    # create a symmetric random matrix
    A = np.random.standard_normal((5,5))
    A = A.T + A

    # tridiagonalize it via householder reflections
    B,_ = la.hessenberg_sym( A )

    # now do a single iteration of the hessenberg qr algorithm
    H,U = la.hessenberg_qr( B )

    # check result
    assert( np.allclose( U@H@U.T, B ) )
    print('test_hessenberg_qr passed.')

def test_eigvalsh():
    A = np.random.standard_normal((5,5))
    A = (A.T + A)/2

    lam, U = la.eigvalsh( A )

    assert( np.allclose( U@np.diag(lam)@U.T, A ) )

if __name__ == '__main__':
    unittest.main()