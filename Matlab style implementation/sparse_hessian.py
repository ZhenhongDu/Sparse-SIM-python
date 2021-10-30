import cupy as cp
from cupyx.scipy.fft import fftn,ifftn
from util.operation import operation_xx,operation_xy,operation_xz,operation_yy,operation_yz,operation_zz
from util.sparse_iteration import iter_xx,iter_xy,iter_xz,iter_yy,iter_yz,iter_zz,iter_sparse
import gc

def sparse_hessian(f,fidelity=100,contiz=0.8,sparsity=5,mu=1,iteration_num = 100):
    contiz = cp.sqrt(contiz)
    f = cp.asarray(f,dtype='float32')
    # f = cp.divide(f,cp.max(f[:]))
    imgsize = cp.shape(f)

    print("start sparse deconvolution")
    ## calculate derivate
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)

    operationfft = xxfft + yyfft + (contiz**2)*zzfft+ 2*xyfft + 2*(contiz)*xzfft + 2*(contiz)*yzfft

    normlize = (fidelity/mu) + (sparsity**2) + operationfft
    normlize = cp.asarray(normlize,dtype='float32')
    del xxfft,yyfft,zzfft,xyfft,xzfft,yzfft,operationfft
    gc.collect()
    cp.clear_memo()
    ## initialize b
    bxx = cp.zeros(imgsize,dtype='float32')
    byy = bxx
    bzz = bxx
    bxy = bxx
    bxz = bxx
    byz = bxx
    bl1 = bxx

    ## initialize g
    g_update = cp.multiply((fidelity/mu),f)
    ## iteration
    for iter in range(0,iteration_num):

        g_update = fftn(g_update)

        if iter == 1:
            g = ifftn(g_update / (fidelity/mu)).real

        else:
            g = ifftn(cp.divide(g_update,normlize)).real


        g_update = cp.multiply((fidelity/mu),f)

        Lxx,bxx = iter_xx(g,bxx,1,mu)
        g_update = g_update + Lxx

        Lyy,byy = iter_yy(g,byy,1,mu)
        g_update = g_update + Lyy

        Lzz,bzz = iter_zz(g, bzz, contiz**2, mu)
        g_update = g_update + Lzz

        Lxy,bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy

        Lxz,bxz = iter_xz(g, bxz, 2*contiz, mu)
        g_update = g_update + Lxz

        Lyz,byz = iter_yz(g, byz, 2*contiz, mu)
        g_update = g_update + Lyz

        Lsparse,bl1 = iter_sparse(g,bl1,sparsity,mu)
        g_update = g_update + Lsparse

        print('iteration '+str(iter) + ' have done!')


    g[g<0] = 0

    del bxx,byy,bzz,bxy,byz,bl1,f,normlize,g_update
    gc.collect()

    return g
