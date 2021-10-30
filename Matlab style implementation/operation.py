from cupyx.scipy.fft import fftn
import cupy as cp

def operation_xx(gsize):
    delta_xx = cp.array([[[1, -2, 1]]],dtype='float32')
    xxfft = fftn(delta_xx,gsize)*cp.conj(fftn(delta_xx,gsize))
    return xxfft

def operation_xy(gsize):
    delta_xy = cp.array([[[1, -1], [-1, 1]]],dtype='float32')
    xyfft = fftn(delta_xy,gsize)*cp.conj(fftn(delta_xy,gsize))
    return xyfft

def operation_xz(gsize):
    delta_xz = cp.array([[[1, -1]], [[-1, 1]]],dtype='float32')
    xzfft = fftn(delta_xz,gsize)*cp.conj(fftn(delta_xz,gsize))
    return xzfft

def operation_yy(gsize):
    delta_yy = cp.array([[[1], [-2], [1]]],dtype='float32')
    yyfft = fftn(delta_yy,gsize)*cp.conj(fftn(delta_yy,gsize))
    return yyfft

def operation_yz(gsize):
    delta_yz = cp.array([[[1], [-1]], [[-1], [1]]],dtype='float32')
    yzfft = fftn(delta_yz,gsize)*cp.conj(fftn(delta_yz,gsize))
    return yzfft

def operation_zz(gsize):
    delta_zz = cp.array([[[1]], [[-2]], [[1]]],dtype='float32')
    zzfft = fftn(delta_zz,gsize)*cp.conj(fftn(delta_zz,gsize))
    return zzfft
