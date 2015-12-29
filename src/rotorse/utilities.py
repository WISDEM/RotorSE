#!/usr/bin/env python
# encoding: utf-8
"""
utilities.py

Created by Andrew Ning on 2013-05-31.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from scipy.linalg import solve_banded
from openmdao.api import IndepVarComp, Component, Problem, Group

def cosd(value):
    """cosine of value where value is given in degrees"""

    return np.cos(np.radians(value))


def sind(value):
    """sine of value where value is given in degrees"""

    return np.sin(np.radians(value))


def tand(value):
    """tangent of value where value is given in degrees"""

    return np.tan(np.radians(value))


def hstack(vec):
    """stack arrays horizontally.  useful for assemblying Jacobian
    assumes arrays are column vectors (if rows just use concatenate)"""
    newvec = []
    for v in vec:
        if len(v.shape) == 1:
            newvec.append(v[:, np.newaxis])
        else:
            newvec.append(v)

    return np.hstack(newvec)


def vstack(vec):
    """stack arrays vertically
    assumes arrays are row vectors.  if columns use concatenate"""

    newvec = []
    for v in vec:
        if len(v.shape) == 1:
            newvec.append(v[np.newaxis, :])
        else:
            newvec.append(v)

    return np.vstack(newvec)


def _checkIfFloat(x):
    try:
        n = len(x)
    except TypeError:  # if x is just a float
        x = np.array([x])
        n = 1

    return x, n


def linspace_with_deriv(start, stop, num):
    """creates linearly spaced arrays, and derivatives for changing end points"""

    step = (stop-start)/float((num-1))
    y = np.arange(0, num) * step + start
    y[-1] = stop

    # gradients
    const = np.arange(0, num) * 1.0/float((num-1))
    dy_dstart = -const + 1.0
    dy_dstart[-1] = 0.0

    dy_dstop = const
    dy_dstop[-1] = 1.0

    return y, dy_dstart, dy_dstop


def interp_with_deriv(x, xp, yp):
    """linear interpolation and its derivative. To be precise, linear interpolation is not
    differentiable right at the control points, but in general it works well enough"""
    # TODO: put in Fortran to speed up

    x, n = _checkIfFloat(x)

    if np.any(np.diff(xp) < 0):
        raise TypeError('xp must be in ascending order')

    # n = len(x)
    m = len(xp)

    y = np.zeros(n)
    dydx = np.zeros(n)
    dydxp = np.zeros((n, m))
    dydyp = np.zeros((n, m))

    for i in range(n):
        if x[i] < xp[0]:
            j = 0  # linearly extrapolate
        elif x[i] > xp[-1]:
            j = m-2
        else:
            for j in range(m-1):
                if xp[j+1] > x[i]:
                    break
        x1 = xp[j]
        y1 = yp[j]
        x2 = xp[j+1]
        y2 = yp[j+1]

        y[i] = y1 + (y2 - y1)*(x[i] - x1)/(x2 - x1)
        dydx[i] = (y2 - y1)/(x2 - x1)
        dydxp[i, j] = (y2 - y1)*(x[i] - x2)/(x2 - x1)**2
        dydxp[i, j+1] = -(y2 - y1)*(x[i] - x1)/(x2 - x1)**2
        dydyp[i, j] = 1 - (x[i] - x1)/(x2 - x1)
        dydyp[i, j+1] = (x[i] - x1)/(x2 - x1)

    if n == 1:
        y = y[0]

    return y, np.diag(dydx), dydxp, dydyp


def cubic_with_deriv(x, xp, yp):
    """deprecated"""

    x, n = _checkIfFloat(x)

    if np.any(np.diff(xp) < 0):
        raise TypeError('xp must be in ascending order')

    # n = len(x)
    m = len(xp)

    y = np.zeros(n)
    dydx = np.zeros(n)
    dydxp = np.zeros((n, m))
    dydyp = np.zeros((n, m))

    xk = xp[1:-1]
    yk = yp[1:-1]
    xkp = xp[2:]
    ykp = yp[2:]
    xkm = xp[:-2]
    ykm = yp[:-2]

    b = (ykp - yk)/(xkp - xk) - (yk - ykm)/(xk - xkm)
    l = (xk - xkm)/6.0
    d = (xkp - xkm)/3.0
    u = (xkp - xk)/6.0
    # u[0] = 0.0  # non-existent entries
    # l[-1] = 0.0

    # solve for second derivatives
    fpp = solve_banded((1, 1), np.matrix([u, d, l]), b)
    fpp = np.concatenate([[0.0], fpp, [0.0]])  # natural spline

    # find location in vector
    for i in range(n):
        if x[i] < xp[0]:
            j = 0
        elif x[i] > xp[-1]:
            j = m-2
        else:
            for j in range(m-1):
                if xp[j+1] > x[i]:
                    break
        x1 = xp[j]
        y1 = yp[j]
        x2 = xp[j+1]
        y2 = yp[j+1]

        A = (x2 - x[i])/(x2 - x1)
        B = 1 - A
        C = 1.0/6*(A**3 - A)*(x2 - x1)**2
        D = 1.0/6*(B**3 - B)*(x2 - x1)**2

        y[i] = A*y1 + B*y2 + C*fpp[j] + D*fpp[j+1]
        dAdx = -1.0/(x2 - x1)
        dBdx = -dAdx
        dCdx = 1.0/6*(3*A**2 - 1)*dAdx*(x2 - x1)**2
        dDdx = 1.0/6*(3*B**2 - 1)*dBdx*(x2 - x1)**2
        dydx[i] = dAdx*y1 + dBdx*y2 + dCdx*fpp[j] + dDdx*fpp[j+1]

    if n == 1:
        y = y[0]
        dydx = dydx[0]

    return y


def trapz_deriv(y, x):
    """trapezoidal integration and derivatives with respect to integrand or variable."""

    dI_dy = np.gradient(x)
    dI_dy[0] /= 2
    dI_dy[-1] /= 2

    dI_dx = -np.gradient(y)
    dI_dx[0] = -0.5*(y[0] + y[1])
    dI_dx[-1] = 0.5*(y[-1] + y[-2])

    return dI_dy, dI_dx


def _smooth_maxmin(yd, ymax, maxmin, pct_offset=0.01, dyd=None):

    yd, n = _checkIfFloat(yd)

    y1 = (1-pct_offset)*ymax
    y2 = (1+pct_offset)*ymax

    dy1 = (1-pct_offset)
    dy2 = (1+pct_offset)

    if maxmin == 'min':
        f1 = y1
        f2 = ymax
        g1 = 1.0
        g2 = 0.0
        idx_constant = yd >= y2

        df1 = dy1
        df2 = 1.0


    elif maxmin == 'max':
        f1 = ymax
        f2 = y2
        g1 = 0.0
        g2 = 1.0
        idx_constant = yd <= y1

        df1 = 1.0
        df2 = dy2

    f = CubicSplineSegment(y1, y2, f1, f2, g1, g2)

    # main region
    ya = np.copy(yd)
    if dyd is None:
        dya_dyd = np.ones_like(yd)
    else:
        dya_dyd = np.copy(dyd)

    dya_dymax = np.zeros_like(ya)

    # cubic spline region
    idx = np.logical_and(yd > y1, yd < y2)
    ya[idx] = f.eval(yd[idx])
    dya_dyd[idx] = f.eval_deriv(yd[idx])
    dya_dymax[idx] = f.eval_deriv_params(yd[idx], dy1, dy2, df1, df2, 0.0, 0.0)

    # constant region
    ya[idx_constant] = ymax
    dya_dyd[idx_constant] = 0.0
    dya_dymax[idx_constant] = 1.0

    if n == 1:
        ya = ya[0]
        dya_dyd = dya_dyd[0]
        dya_dymax = dya_dymax[0]


    return ya, dya_dyd, dya_dymax


def smooth_max(yd, ymax, pct_offset=0.01, dyd=None):
    """array max, uses cubic spline to smoothly transition.  derivatives with respect to array and max value.
    width of transition can be controlled, and chain rules for differentiation"""
    return _smooth_maxmin(yd, ymax, 'max', pct_offset, dyd)


def smooth_min(yd, ymin, pct_offset=0.01, dyd=None):
    """array min, uses cubic spline to smoothly transition.  derivatives with respect to array and min value.
    width of transition can be controlled, and chain rules for differentiation"""
    return _smooth_maxmin(yd, ymin, 'min', pct_offset, dyd)



def smooth_abs(x, dx=0.01):
    """smoothed version of absolute vaue function, with quadratic instead of sharp bottom.
    Derivative w.r.t. variable of interest.  Width of quadratic can be controlled"""

    x, n = _checkIfFloat(x)

    y = np.abs(x)
    idx = np.logical_and(x > -dx, x < dx)
    y[idx] = x[idx]**2/(2.0*dx) + dx/2.0

    # gradient
    dydx = np.ones_like(x)
    dydx[x <= -dx] = -1.0
    dydx[idx] = x[idx]/dx


    if n == 1:
        y = y[0]
        dydx = dydx[0]

    return y, dydx



def cubic_spline_eval(x1, x2, f1, f2, g1, g2, x):

    spline = CubicSplineSegment(x1, x2, f1, f2, g1, g2)
    return spline.eval(x)



class CubicSplineSegment(object):
    """cubic splines and the their derivatives with with respect to the variables and the parameters"""

    def __init__(self, x1, x2, f1, f2, g1, g2):

        self.x1 = x1
        self.x2 = x2

        self.A = np.array([[x1**3, x1**2, x1, 1.0],
                  [x2**3, x2**2, x2, 1.0],
                  [3*x1**2, 2*x1, 1.0, 0.0],
                  [3*x2**2, 2*x2, 1.0, 0.0]])
        self.b = np.array([f1, f2, g1, g2])

        self.coeff = np.linalg.solve(self.A, self.b)

        self.poly = np.polynomial.Polynomial(self.coeff[::-1])


    def eval(self, x):
        return self.poly(x)


    def eval_deriv(self, x):
        polyd = self.poly.deriv()
        return polyd(x)


    def eval_deriv_params(self, xvec, dx1, dx2, df1, df2, dg1, dg2):

        x1 = self.x1
        x2 = self.x2
        dA_dx1 = np.matrix([[3*x1**2, 2*x1, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [6*x1, 2.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
        dA_dx2 = np.matrix([[0.0, 0.0, 0.0, 0.0],
                  [3*x2**2, 2*x2, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [6*x2, 2.0, 0.0, 0.0]])
        df = np.array([df1, df2, dg1, dg2])
        c = np.matrix(self.coeff).T

        n = len(xvec)
        dF = np.zeros(n)
        for i in range(n):
            x = np.array([xvec[i]**3, xvec[i]**2, xvec[i], 1.0])
            d = np.linalg.solve(self.A.T, x)
            dF_dx1 = -d*dA_dx1*c
            dF_dx2 = -d*dA_dx2*c
            dF_df = np.linalg.solve(self.A.T, x)
            dF[i] = np.dot(dF_df, df) + dF_dx1[0]*dx1 + dF_dx2[0]*dx2

        return dF




def print_vars(comp, list_type='inputs', prefix='', astable=False):

    reserved = ['missing_deriv_policy', 'force_execute', 'directory', 'force_fd']

    if list_type == 'inputs':
        thelist = comp.list_inputs()
    elif list_type == 'outputs':
        thelist = comp.list_inputs()
    elif list_type == 'vars':
        thelist = comp.list_vars()

    for name in thelist:

        if name in reserved:
            continue

        trait = comp.get_trait(name)
        thetype = trait.trait_type.__str__().split()[0].split('.')[-1]

        if thetype == 'VarTree':
            vartree = getattr(comp, name)
            if prefix is not None:
                newprefix = prefix + '.' + name
            else:
                newprefix = name
            print_vars(vartree, 'vars', prefix=newprefix)
            continue

        units = trait.units
        desc = trait.desc
        default = trait.default
        # print trait.category

        if units is None:
            description = '(' + thetype + ')'
        else:
            description = '(' + thetype + ', ' + units + ')'

        if desc is not None:
            description += ': ' + desc


        if prefix is not '':
            name = prefix + '.' + name

        if not astable:
            name = name + ' = ' + str(default)
            print name + '  # ' + description

        else:

            if not units:
                units = ''
            if not desc:
                desc = ''
            strdefault = str(default)
            if strdefault == '<undefined>':
                strdefault = ''

            print '{0:15}\t{1:10}\t{2:15}\t{3:10}\t{4}'.format(name, thetype, strdefault, units, desc)






def _getvar(comp, name):
    vars = name.split('.')
    base = comp
    for var in vars:
        base = getattr(base, var)

    return base


def _setvar(comp, name, value):
    vars = name.split('.')
    base = comp
    for i in range(len(vars)-1):
        base = getattr(base, vars[i])

    setattr(base, vars[-1], value)


def _explodeall(comp, vtype='inputs'):

    if vtype == 'inputs':
        alloutputs = comp.list_inputs()
    else:
        alloutputs = comp.list_outputs()

    maybe_more_vartrees = True

    while maybe_more_vartrees:
        maybe_more_vartrees = False

        for out in alloutputs:
            try:
                vars = getattr(comp, out).list_vars()
                alloutputs.remove(out)
                for var in vars:
                    alloutputs.append(out + '.' + var)
                maybe_more_vartrees = True

            except Exception:
                pass

    return alloutputs


def _getColumnOfOutputs(comp, outputs, m):

    # fill out column of outputs
    m1 = 0
    m2 = 0
    f = np.zeros(m)
    for i, out in enumerate(outputs):

        # get function value at center
        fsub = _getvar(comp, out)
        if np.array(fsub).shape == ():
            lenf = 1
        else:
            fsub = np.copy(fsub)  # so not pointed to same memory address
            lenf = len(fsub)

        m2 += lenf

        f[m1:m2] = fsub

        m1 = m2

    return f


def check_for_missing_unit_tests(modules):
    """A heuristic check to find components that don't have a corresonding unit test
    for its gradients.

    Parameters
    ----------
    modules : list(str)
        a list of modules to check for missin ggradients
    """
    import sys
    import inspect
    import ast

    thisfilemembers = inspect.getmembers(sys.modules['__main__'], lambda member: inspect.isclass(member) and member.__module__ == '__main__')
    tests = [name for name, classname in thisfilemembers]

    totest = []
    tomod = []
    reserved = ['Assembly', 'Slot', 'ImplicitComponent']
    for mod in modules:
        modulemembers = inspect.getmembers(mod, inspect.isclass)
        fname = mod.__file__
        if fname[-1] == 'c':
            fname = fname[:-1]
        f = open(fname, 'r')
        p = ast.parse(f.read())
        f.close()
        nonimported = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
        for name, classname in modulemembers:
            bases = classname.__bases__
            if 'Base' not in name and name not in reserved and name in nonimported and len(bases) > 0:
                base = bases[0].__name__
                if base == 'Component' or 'Base' in base:
                    totest.append(name)
                    tomod.append(mod.__name__)

    for mod, test in zip(tomod, totest):
        if 'Test'+test not in tests:
            print '!!! There does not appear to be a unit test for:', mod + '.' + test


def check_gradient_unit_test(unittest, prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6, comp=None):
    """compare provided analytic gradients to finite-difference gradients with unit testing.
    Same as check_gradient, but provides a unit test for each gradient for convenience.
    the unit tests checks that the error for each gradient is less than tol.

    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings
    """

    J_fd, J_fwd, J_rev = check_gradient(prob, fd, step_size, tol, display, show_missing_warnings,
        show_scaling_warnings, min_grad, max_grad)
    if comp == None:
        comp = prob.root.comp
    if "list_deriv_vars" in dir(comp): #  callable(getattr(comp, 'list_deriv_vars')):
        inputs, outputs = comp.list_deriv_vars()
        for output in outputs:
            for input in inputs:
                J = J_fwd[output, input]
                JFD = J_fd[output, input]
                m, n = J.shape
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)

                        # # display
                        # if error > tol:
                        #     star = ' ***** '
                        # else:
                        #     star = ''
                        #
                        # if display:
                        #     output = '{}{:<20} ({}) {}: ({}, {})'.format(star, error, errortype, name, J[i, j], JFD[i, j])
                        #     print output
                        #
                        # if show_scaling_warnings and J[i, j] != 0 and np.abs(J[i, j]) < min_grad:
                        #     print '*** Warning: The following analytic gradient is very small and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #
                        # if show_scaling_warnings and np.abs(J[i, j]) > max_grad:
                        #     print '*** Warning: The following analytic gradient is very large and may need to be scaled:'
                        #     print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
                        #

                        try:
                            unittest.assertLessEqual(error, tol)
                        except AssertionError, e:
                            print '*** error in:', "\n\tOutput: ", output, "\n\tInput: ", input, "\n\tPosition: ", i, j
                            raise e
    else:
        for key, value in J_fd.iteritems():
                J = J_fwd[key]
                JFD = J_fd[key]
                m, n = J.shape
                for i in range(m):
                    for j in range(n):
                        if np.abs(J[i, j]) <= tol:
                            errortype = 'absolute'
                            error = J[i, j] - JFD[i, j]
                        else:
                            errortype = 'relative'
                            error = 1.0 - JFD[i, j]/J[i, j]
                        error = np.abs(error)
                        try:
                            unittest.assertLessEqual(error, tol)
                        except AssertionError, e:
                            print '*** error in:', "\n\tKey: ", key, "\n\tPosition: ", i, j
                            raise e


def check_gradient(prob, fd='central', step_size=1e-6, tol=1e-6, display=False,
        show_missing_warnings=True, show_scaling_warnings=False, min_grad=1e-6, max_grad=1e6):
    """compare provided analytic gradients to finite-difference gradients

    Parameters
    ----------
    comp : obj
        An OpenMDAO component that provides analytic gradients through provideJ()
    fd : str
        the type of finite difference to use.  options are central or forward
    step_size : float
        step size to use in finite differencing
    tol : float
        tolerance for how close the gradients should agree to
    display : boolean
        if True, display results for each gradient
    show_missing_warnings: boolean
        if True, warn for gradients that were not provided
        (they may be ones that are unecessary, but may be ones that were accidentally skipped)
    show_scaling_warnings: boolean
        if True, warn for gradients that are either very small or very large which may lead
        to challenges in solving the full linear system
    min_grad/max_grad : float
        quantifies what "very small" or "very large" means when using show_scaling_warnings

    Returns
    -------
    names : array(str)
        list of the names of all the gradients
    errorvec : array(float)
        list of all the errors for the gradients.  If the magnitude of the gradient is less than
        tol, then an absolute error is used, otherwise a relative error is used.

    """
    # inputs = comp.list_deriv_vars
    # inputs, outputs = comp.list_deriv_vars()

    # show_missing_warnings = False

    # if show_missing_warnings:
    #     all_inputs = _explodeall(comp, vtype='inputs')
    #     all_outputs = _explodeall(comp, vtype='outputs')
    #     reserved_inputs = ['missing_deriv_policy', 'directory', 'force_fd', 'force_execute', 'eval_only']
    #     reserved_outputs = ['derivative_exec_count', 'itername', 'exec_count']
    #     potential_missed_inputs = list(set(all_inputs) - set(reserved_inputs) - set(inputs))
    #     potential_missed_outputs = list(set(all_outputs) - set(reserved_outputs) - set(outputs))
    #
    #     if len(potential_missed_inputs) > 0 or len(potential_missed_outputs) > 0:
    #         print
    #         print '*** Warning: ' + comp.__class__.__name__ + ' does not supply derivatives for the following'
    #         print '\tinputs:', potential_missed_inputs
    #         print '\toutputs:', potential_missed_outputs
    #         print

    # prob = Problem()
    # prob.root = Group()
    # prob.root.add('comp', comp, promotes=['*'])
    # prob.setup()
    #
    # for i in range(len(inputs)):
    #     prob[inputs[i]] = comp

    prob.run()
    root = prob.root

    # Linearize the model
    root._sys_linearize(root.params, root.unknowns, root.resids)

    data = {}

    # Derivatives should just be checked without parallel adjoint for now.
    voi = None

    # Check derivative calculations for all comps at every level of the
    # system hierarchy.
    for comp in root.components(recurse=True):
        cname = comp.pathname

        # No need to check comps that don't have any derivs.
        if comp.fd_options['force_fd']:
            continue

        # IndepVarComps are just clutter too.
        if isinstance(comp, IndepVarComp):
            continue

        data[cname] = {}
        jac_fwd = {}
        jac_rev = {}
        jac_fd = {}

        # try:
        #     params, unknowns = comp.list_deriv_vars()
        # except:
        #     pass
        params = comp.params
        unknowns = comp.unknowns
        resids = comp.resids
        dparams = comp.dpmat[voi]
        dunknowns = comp.dumat[voi]
        dresids = comp.drmat[voi]

        # Skip if all of our inputs are unconnected.
        # if len(dparams) == 0:
        #     continue

        # if out_stream is not None:
        #     out_stream.write('-'*(len(cname)+15) + '\n')
        #     out_stream.write("Component: '%s'\n" % cname)
        #     out_stream.write('-'*(len(cname)+15) + '\n')

        states = comp.states

        param_list = [item for item in dparams if not \
                      dparams.metadata(item).get('pass_by_obj')]
        param_list.extend(states)

        # Create all our keys and allocate Jacs
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Check dimensions of user-supplied Jacobian
            for u_name in unknowns:

                u_size = np.size(dunknowns[u_name])
                if comp._jacobian_cache:

                    # We can perform some additional helpful checks.
                    if (u_name, p_name) in comp._jacobian_cache:

                        user = comp._jacobian_cache[(u_name, p_name)].shape

                        # User may use floats for scalar jacobians
                        if len(user) < 2:
                            user = (user[0], 1)

                        if user[0] != u_size or user[1] != p_size:
                            msg = "derivative in component '{}' of '{}' wrt '{}' is the wrong size. " + \
                                  "It should be {}, but got {}"
                            msg = msg.format(cname, u_name, p_name, (u_size, p_size), user)
                            raise ValueError(msg)

                jac_fwd[(u_name, p_name)] = np.zeros((u_size, p_size))
                jac_rev[(u_name, p_name)] = np.zeros((u_size, p_size))

        # Reverse derivatives first
        for u_name in dresids:
            u_size = np.size(dunknowns[u_name])

            # Send columns of identity
            for idx in range(u_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dresids._dat[u_name].val[idx] = 1.0
                try:
                    comp.apply_linear(params, unknowns, dparams,
                                      dunknowns, dresids, 'rev')
                finally:
                    dparams._apply_unit_derivatives()

                for p_name in param_list:

                    dinputs = dunknowns if p_name in states else dparams
                    # try:
                    jac_rev[(u_name, p_name)][idx, :] = dinputs._dat[p_name].val
                    # except:
                    #     pass
        # Forward derivatives second
        for p_name in param_list:

            dinputs = dunknowns if p_name in states else dparams
            p_size = np.size(dinputs[p_name])

            # Send columns of identity
            for idx in range(p_size):
                dresids.vec[:] = 0.0
                root.clear_dparams()
                dunknowns.vec[:] = 0.0

                dinputs._dat[p_name].val[idx] = 1.0
                dparams._apply_unit_derivatives()
                comp.apply_linear(params, unknowns, dparams,
                                  dunknowns, dresids, 'fwd')

                for u_name, u_val in dresids.vec_val_iter():
                    jac_fwd[(u_name, p_name)][:, idx] = u_val

        # Finite Difference goes last
        dresids.vec[:] = 0.0
        root.clear_dparams()
        dunknowns.vec[:] = 0.0

        # Component can request to use complex step.
        if comp.fd_options['form'] == 'complex_step':
            fd_func = comp.complex_step_jacobian
        else:
            fd_func = comp.fd_jacobian

        jac_fd = fd_func(params, unknowns, resids)

        # # Assemble and Return all metrics.
        # _assemble_deriv_data(chain(dparams, states), resids, data[cname],
        #                      jac_fwd, jac_rev, jac_fd, out_stream,
        #                      c_name=cname)

    return jac_fd, jac_fwd, jac_rev

    # J = prob.provideJ()
    #
    # # compute size of Jacobian
    # m = 0
    # mvec = []  # size of each output
    # cmvec = []  # cumulative size of outputs
    # nvec = []  # size of each input
    # cnvec = []  # cumulative size of inputs
    # for out in outputs:
    #     f = _getvar(comp, out)
    #     if np.array(f).shape == ():
    #         msub = 1
    #     else:
    #         msub = len(f)
    #     m += msub
    #     mvec.append(msub)
    #     cmvec.append(m)
    # n = 0
    # for inp in inputs:
    #     x = _getvar(comp, inp)
    #     if np.array(x).shape == ():
    #         nsub = 1
    #     else:
    #         nsub = len(x)
    #     n += nsub
    #     nvec.append(nsub)
    #     cnvec.append(n)
    #
    # JFD = np.zeros((m, n))
    #
    # if J.shape != JFD.shape:
    #     raise TypeError('Incorrect Jacobian size. Your provided Jacobian is of shape {}, but it should be ({}, {})'.format(J.shape, m, n))
    #
    #
    # # fill out column of outputs
    # f = _getColumnOfOutputs(comp, outputs, m)
    #
    # n1 = 0
    #
    # for j, inp in enumerate(inputs):
    #
    #     # get x value at center (save location)
    #     x = _getvar(comp, inp)
    #     if np.array(x).shape == ():
    #         x0 = x
    #         lenx = 1
    #     else:
    #         x = np.copy(x)  # so not pointing to same memory address
    #         x0 = np.copy(x)
    #         lenx = len(x)
    #
    #     for k in range(lenx):
    #
    #         # take a step
    #         if lenx == 1:
    #             h = np.abs(step_size*x)
    #             if h < step_size:
    #                 h = step_size
    #             x += h
    #         else:
    #             h = np.abs(step_size*x[k])
    #             if h < step_size:
    #                 h = step_size
    #             x[k] += h
    #         _setvar(comp, inp, x)
    #         comp.run()
    #
    #         # fd
    #         fp = _getColumnOfOutputs(comp, outputs, m)
    #
    #         if fd == 'central':
    #
    #             # step back
    #             if lenx == 1:
    #                 x -= 2*h
    #             else:
    #                 x[k] -= 2*h
    #             _setvar(comp, inp, x)
    #             comp.run()
    #
    #             fm = _getColumnOfOutputs(comp, outputs, m)
    #
    #             deriv = (fp - fm)/(2*h)
    #
    #         else:
    #             deriv = (fp - f)/h
    #
    #
    #         JFD[:, n1+k] = deriv
    #
    #         # reset state
    #         x = np.copy(x0)
    #         _setvar(comp, inp, x0)
    #         comp.run()
    #
    #     n1 += lenx
    #
    #
    # # error checking
    # namevec = []
    # errorvec = []
    #
    # if display:
    #     print '{:<20} ({}) {:<10} ({}, {})'.format('error', 'errortype', 'name', 'analytic', 'fd')
    #     print
    #
    # for i in range(m):
    #     for j in range(n):
    #
    #         # get corresonding variables names
    #         for ii in range(len(mvec)):
    #             if cmvec[ii] > i:
    #                 oname = 'd_' + outputs[ii]
    #
    #                 if mvec[ii] > 1:  # need to print indices
    #                     subtract = 0
    #                     if ii > 0:
    #                         subtract = cmvec[ii-1]
    #                     idx = i - subtract
    #                     oname += '[' + str(idx) + ']'
    #
    #                 break
    #         for jj in range(len(nvec)):
    #             if cnvec[jj] > j:
    #                 iname = 'd_' + inputs[jj]
    #
    #                 if nvec[jj] > 1:  # need to print indices
    #                     subtract = 0
    #                     if jj > 0:
    #                         subtract = cnvec[jj-1]
    #                     idx = j - subtract
    #                     iname += '[' + str(idx) + ']'
    #
    #                 break
    #         name = oname + ' / ' + iname
    #
    #         # compute error
    #         if np.abs(J[i, j]) <= tol:
    #             errortype = 'absolute'
    #             error = J[i, j] - JFD[i, j]
    #         else:
    #             errortype = 'relative'
    #             error = 1.0 - JFD[i, j]/J[i, j]
    #         error = np.abs(error)
    #
    #         # display
    #         if error > tol:
    #             star = ' ***** '
    #         else:
    #             star = ''
    #
    #         if display:
    #             output = '{}{:<20} ({}) {}: ({}, {})'.format(star, error, errortype, name, J[i, j], JFD[i, j])
    #             print output
    #
    #         if show_scaling_warnings and J[i, j] != 0 and np.abs(J[i, j]) < min_grad:
    #             print '*** Warning: The following analytic gradient is very small and may need to be scaled:'
    #             print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
    #
    #         if show_scaling_warnings and np.abs(J[i, j]) > max_grad:
    #             print '*** Warning: The following analytic gradient is very large and may need to be scaled:'
    #             print '\t(' + comp.__class__.__name__ + ') ' + name + ':', J[i, j]
    #
    #
    #         # save
    #         namevec.append(name)
    #         errorvec.append(error)
    #
    # return namevec, errorvec



# if __name__ == '__main__':



    # xpt = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
    # ypt = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])

    # # interpolate  (extrapolation will work, but beware the results may be silly)
    # n = 50
    # x = np.linspace(0.0, 13.0, n)
    # y = cubic_with_deriv(x, xpt, ypt)

    # import matplotlib.pyplot as plt
    # plt.plot(xpt, ypt, 'o')
    # plt.plot(x, y, '-')
    # plt.show()
