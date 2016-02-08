import numpy as np
from scipy.optimize import brentq as brentq_original
from copy import deepcopy

def brentq(f, a, b, args={}, xtol=1e-12, rtol=4e-16, maxiter=100, full_output=True): 
    """Vectorized version of a Brent Solver. It operates
    independently on each element in f.
    """

    size = a.shape

    n_iters = 0 
    n_func_cals = 0

    xblk = np.zeros(size)
    fblk = np.zeros(size)
    spre = np.zeros(size)
    scur = np.zeros(size)
    dblk = np.zeros(size)
    dpre = np.zeros(size)
    stry = np.zeros(size)

    xpre = a.copy()
    xcur = b.copy()

    fpre = f(xpre, *args).copy()
    fcur = f(xcur, *args).copy()
    n_func_cals += 2

    if np.any(fpre*fcur > 0):
        raise ValueError('f(a) and f(b) must have different signs')

    while n_iters < maxiter: 

        # fcur = f(xcur, **args)
        # find elements where sign has switch between pre and cur
        opp_sign = fpre*fcur < 0
        if np.any(opp_sign): 
            xblk[opp_sign] = xpre[opp_sign]
            fblk[opp_sign] = fpre[opp_sign]
            spre[opp_sign] = scur[opp_sign] = xcur[opp_sign] - xpre[opp_sign]
        
        # check if any of abs(fblk) < abs(fcur) for any elements 
        abs_check = np.abs(fblk) < np.abs(fcur)
        if np.any(abs_check): 
            xpre[abs_check] = xcur[abs_check]
            xcur[abs_check] = xblk[abs_check]
            xblk[abs_check] = xpre[abs_check]
            fpre[abs_check] = fcur[abs_check] 
            fcur[abs_check] = fblk[abs_check] 
            fblk[abs_check] = fpre[abs_check]
        
        # check for convergence
        tol = xtol + rtol*np.abs(xcur)
        sbis = (xblk - xcur)/2.0
        if (np.all(fcur == 0) or np.all(np.abs(sbis) < tol)):
            return xcur, None

        quadratic_interp = np.logical_and(np.abs(spre) > tol,  np.abs(fcur) < np.abs(fpre))

        # Just do quadratic interp on everyone, and will overwrite the bisection ones after
        # interpolate
        interp = xpre == xblk
        stry[interp] = -fcur[interp]*(xcur[interp] - xpre[interp])/(fcur[interp] - fpre[interp])
           
        # if not interpolate, then extrapolate
        extrap = np.logical_not(interp)
        if extrap[0] == True:
            pass
        dpre[extrap] = (fpre[extrap] - fcur[extrap])/(xpre[extrap] - xcur[extrap])
        dblk[extrap] = (fblk[extrap] - fcur[extrap])/(xblk[extrap] - xcur[extrap])
        stry[extrap] = -fcur[extrap]*(fblk[extrap]*dblk[extrap] - fpre[extrap]*dpre[extrap]) \
            /(dblk[extrap]*dpre[extrap]*(fblk[extrap] - fpre[extrap]))
        
        # good short step
        short_step = 2*np.abs(stry) < np.minimum(np.abs(spre), 3*np.abs(sbis) - tol)
        spre[short_step] = scur[short_step]
        scur[short_step] = stry[short_step]

        # if not short-step, bisection
        bisect = np.logical_not(short_step)
        spre[bisect] = sbis[bisect]
        scur[bisect] = sbis[bisect]
        
        # also bisect anything that wasn't good for the quadratic interp/exterp
        bisect = np.logical_not(quadratic_interp)
        spre[bisect] = sbis[bisect]
        scur[bisect] = sbis[bisect]

        xpre = deepcopy(xcur)
        fpre = deepcopy(fcur)

        too_big = np.abs(scur) > tol
        xcur[too_big] += scur[too_big]
        ok = np.logical_not(too_big)

        plus_tol = np.logical_and(ok, sbis > 0)
        xcur[plus_tol] += tol[plus_tol]
        minus_tol = np.logical_and(ok, sbis < 0)
        xcur[minus_tol] -= tol[minus_tol]

        fcur = f(xcur, *args)
        n_func_cals += 1
        n_iters += 1
        # print "inter", n_iters, xcur


if __name__ == "__main__": 

    aa =1.
    bb =1.
    cc =10.
    n =77.0/27.0

    def resid(u):

        return aa * u**n + bb * u - cc

    size = 1
    a = np.zeros(size)
    b = 10*np.ones(size)
    n = 77.0/27.0 * np.ones(size)
    # n[1] = 77.0/27.5

    import time
    st = time.time()
    for i in xrange(10): 
        brentq(resid, a, b)
    print "Vector brent time ", time.time()-st
    n = 77.0/27.5

    st = time.time()
    for i in xrange(10): 
        print "Original Brent (first element)", brentq_original(resid, a[0], b[0])
    print "Scalar brent time ", time.time()-st

