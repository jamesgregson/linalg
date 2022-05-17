import numpy as np

import la


def hess_test():
    A = np.random.standard_normal((5,5))
    A = (A.T + A)/2
    H,U = la.hessenberg_sym( A, inplace=False, compute_u=True )

    H2,U2 = la.hessenberg_qr( H, inplace=False, compute_u=True )
    for i in range(40):
        H2,U2 = la.hessenberg_qr( H2, inplace=False, compute_u=True )


    print('hello')

def main():
    np.set_printoptions(precision=4,suppress=True,linewidth=1000)
    hess_test()


if __name__ == '__main__':
    main()