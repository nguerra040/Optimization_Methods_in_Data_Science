# -*- coding: utf-8 -*-
"""
Minimization Algorithms

Author: Nicolas Guerra
Due Date: 5/20/2022

This module's purpose is to minimize is function using either gradient
descent method or Newton's method with line search.
"""

# Import dependencies
import numpy as np
import golden_section
import scipy as sp
import scipy.linalg


def init_options():
    '''
    Option initialization for the specified minimization algorithm

    Initialize algorithm options with default values

    Return values:
        options:
            This is a dictionary with fields that correspond to algorithmic
            options of our method.  In particular:

            max_iter:
                Maximum number of iterations
            tol:
                Convergence tolerance
            step_type:
                Different ways to calculate the search direction:
                    ’gradient_descent’
                    ’Newton’
            linesearch:
                Type of line search:
                    ’constant’
                    ’golden_section’
                    ’backtracking’
            alpha_init:
                First trial step size to attempt during line search:
                    ’constant’: Value of the constant step size
                    ’golden_section’: Initial upper bound of search interval
                    ’backtracking’: Initial trial step size
            suff_decrease_factor:
                coefficient in sufficient decrease condition for backtracking
                line search
            linesearch_tol:
                tolerance for golden section search
            max_iter_gs:
                maximum number of iterations golden section algorithm
                is allowed to take
            perturb_init:
                initial perturbation of Hessian matrix to ensure positive
                definiteness
            perturb_inc_factor:
                increase factor for Hessian perturbation
            output_level:
                Amount of output printed
                    0: No output
                    1: Only summary information
                    2: One line per iteration (good for debugging)
                    3: Print iterates in each iteration

    '''
    options = {}
    # Default values
    options['max_iter'] = 1000
    options['tol'] = 1e-8
    options['step_type'] = 'Newton'
    options['linesearch'] = 'backtracking'
    options['alpha_init'] = 1
    options['suff_decrease_factor'] = 1e-4
    options['linesearch_tol'] = 1e-6
    options['max_iter_gs'] = 10000
    options['perturb_init'] = 1e-4
    options['perturb_inc_factor'] = 10
    options['output_level'] = 1

    return options


def minimize(opt_prob, options, x_start=None):
    '''
    Optimization methods for unconstrainted optimization

    This is an implementation of the gradient descent method for unconstrained
    optimization as well as Newton's method with line search.

    Input arguments:
        opt_prob:
            Optimization problem object to define an unconstrained optimization
            problem.  It must have the following methods:

            val = value(x)
                returns value of objective function at x
            grad = gradient(x)
                returns gradient of objective function at x
            x_start = starting_point()
                returns a default starting point
        options:
            This is a structure with options for the algorithm.
            For details see the init_options function.
        x_start:
            Starting point.  If set to None, obtain default starting point
            from opt_problem.

    Return values:
        status:
            Return code indicating reason for termination:
            'success': Critical point found (convergence tolerance satisfied)
            'iteration limit':  Maximum number of iterations exceeded
            'infinite objective value': Objective function returned Inf
            'incomplete': The gradient method has not finished
        x_sol:
            Approximate critical point (or last iterate if there is a failure)
        f_sol:
            function value of x_sol
    '''

    # Get option values
    max_iter = options['max_iter']
    tol = options['tol']
    step_type = options['step_type']
    linesearch = options['linesearch']
    alpha_init = options['alpha_init']
    suff_decrease_factor = options['suff_decrease_factor']
    linesearch_tol = options['linesearch_tol']
    max_iter_gs = options['max_iter_gs']
    perturb_init = options['perturb_init']
    perturb_inc_factor = options['perturb_inc_factor']
    output_level = options['output_level']

    # get starting point. If none is provided explicitly for this call,
    # ask the OptProblem object for it.
    if x_start is None:
        x_start = opt_prob.starting_point()
    x_k = np.copy(x_start)

    # Initialize iteration number
    num_iter = 0

    # current function value, gradient, and gradient norm
    f_k = opt_prob.value(x_k)
    grad_k = opt_prob.gradient(x_k)
    norm_grad = np.linalg.norm(grad_k, np.inf)
    # Initialize number of function evaluations
    num_f = 1
    num_grad = 1
    num_hess = 0
    # Initialize step size
    alpha_k = alpha_init
    # Initialize status
    status = 'incomplete'

    # Calculate initial step direction using specified method
    if step_type == 'Newton':
        # Get Cholesky factorization
        hess_init = opt_prob.hessian(x_k)
        L_k, perturb_k = \
            factorize_with_mod(hess_init, perturb_init, perturb_inc_factor)

        # Clarifying lower triangle Cholesky
        lower = True
        # Cholesky solve for descent direction
        d_k = sp.linalg.cho_solve((L_k, lower), -grad_k)
        norm_d = np.linalg.norm(d_k, np.inf)
        # Initialize number of Hessian evaluations
        num_hess += 1
    elif step_type == 'gradient':
        # Get descent direction
        d_k = -grad_k
        norm_d = np.linalg.norm(d_k, np.inf)
        # Since not perturbing hessian, set to 0
        perturb_k = 0

    # Output initial info
    if output_level == 2:
        print('iter       f                ||d_k||          alpha        '
              '# func    perturb         ||grad||')
        print('%3d       %10.3e       %10.3e       %10.3e     %3d     '
              '%10.3e      %10.3e' % (num_iter, f_k, norm_d, alpha_k, num_f,
                                      perturb_k, norm_grad))
    elif output_level == 3:
        print('iter       f                ||d_k||          alpha        '
              '# func    perturb         ||grad||')
        print('%3d       %10.3e       %10.3e       %10.3e     %3d     '
              '%10.3e      %10.3e' % (num_iter, f_k, norm_d, alpha_k, num_f,
                                      perturb_k, norm_grad))
        print("iterate: ", x_k)

    while norm_grad > tol:
        # Check if we are beyond max iteration limit on next iteration
        if max_iter < num_iter+1:
            status = 'iteration limit'
            break

        # Increment iteration number
        num_iter += 1

        # Calculate next step and step size
        x_k, f_k, grad_k, alpha_k, inc_num_f, inc_num_grad = \
            linesearch_func(x_k, d_k, suff_decrease_factor, linesearch,
                            alpha_init, linesearch_tol, max_iter_gs, opt_prob)

        # Current gradient norm
        norm_grad = np.linalg.norm(grad_k, np.inf)
        # Increment number of function evaluations
        num_f += inc_num_f
        num_grad += inc_num_grad

        # Calculate initial step direction using specified method
        if step_type == 'Newton':
            # Get Cholesky factorization
            hess_init = opt_prob.hessian(x_k)
            L_k, perturb_k = factorize_with_mod(hess_init, perturb_init,
                                                perturb_inc_factor)

            # Cholesky solve for descent direction
            d_k = sp.linalg.cho_solve((L_k, lower), -grad_k)
            norm_d = np.linalg.norm(d_k, np.inf)
            # Increment number of Hessian evaluations
            num_hess += 1
        elif step_type == 'gradient':
            # Get descent direction
            d_k = -grad_k
            norm_d = np.linalg.norm(d_k, np.inf)

        # Output initial info
        if output_level == 2:
            print('iter       f                ||d_k||          alpha        '
                  '# func    perturb         ||grad||')
            print('%3d       %10.3e       %10.3e       %10.3e     %3d     '
                  '%10.3e      %10.3e' % (num_iter, f_k, norm_d, alpha_k,
                                          num_f, perturb_k, norm_grad))
        elif output_level == 3:
            print('iter       f                ||d_k||          alpha        '
                  '# func    perturb         ||grad||')
            print('%3d       %10.3e       %10.3e       %10.3e     %3d     '
                  '%10.3e      %10.3e' % (num_iter, f_k, norm_d, alpha_k,
                                          num_f, perturb_k, norm_grad))
            print("iterate: ", x_k)

        # Check if f is infinite
        if f_k == float('inf') or f_k == float('-inf'):
            status = 'infinite objective function'
            break

    # Check if we converged before iteration limit
    if status != 'iteration limit' and status != 'infinite objective function':
        status = 'success'

    # Finalize results
    x_sol = np.copy(x_k)
    f_sol = f_k

    if output_level > 0:
        print('\nSummary')
        print(f'Status: {status}')
        print('Number of Iterations: %3d' % (num_iter+1))
        print('Number of Objective Function Evaluations: %3d' % num_f)
        print('Number of Gradient Evaluations: %3d' % num_grad)
        print('Number of Hessian Evaluations: %3d' % num_hess)
        print('Objective value: %10.3e' % f_sol)
        print('Infinity-norm of gradient: %10.3e' % norm_grad)
        print('Computed Solution: ', x_sol)

    return status, x_sol, f_sol


def factorize_with_mod(hess_init, perturb_init, perturb_inc_factor,
                       isperturbed=False, hess_k=0):
    '''
    Cholesky factorization

    This function finds the Cholesky factorization of the given matrix. If
    given matrix is not positive definite, the diagonal of the given matrix
    is perturbed in order to get a positive definite matrix to find
    a Cholesky factorization.

    Input arguments:
        hess_init:
            The matrix for which to extract the Cholesky factorization
        perturb_init:
            Amount to perturb the diagonal of the given matrix if given matrix
            is not positive definite
        perturb_inc_factor:
            Factor to increase the perturbation amount if the previous
            perturbed matrix was still not positive definite
        isperturbed:
            (Default - False)
            If False, given matrix was not perturbed.
            If True, given matrix has been perturbed.
        hess_k:
            The perturbed matrix, initially equaling zero since not useful
            until after a perturbation has been made

    Return values:
        L:
            Cholesky factorization lower triangle matrix
        final_pert:
            Amount by which the initial given matrix was perturbed in order
            to get a Cholesky factorization
    '''
    try:
        # Check if hessian has been perturbed yet (if hess_k is == 0)
        if isinstance(hess_k, int) and hess_k == 0:
            # Try to get cholesky factorization
            L = sp.linalg.cholesky(hess_init, lower=True)
        else:
            L = sp.linalg.cholesky(hess_k, lower=True)

        # If perturbed, return amount eigenvalues were shifted
        if isperturbed:
            final_pert = perturb_init/perturb_inc_factor  # Gets prev. perturb
            return L, final_pert
        else:
            return L, 0
    except sp.linalg.LinAlgError:
        # Perturb eigenvalues of hessian
        hess_k = hess_init + perturb_init*np.identity(hess_init.shape[0])
        perturb_init *= perturb_inc_factor

        # Call same function to loop until we get L
        return factorize_with_mod(hess_init, perturb_init, perturb_inc_factor,
                                  isperturbed=True, hess_k=hess_k)


def linesearch_func(x_k, d_k, suff_decrease_factor, linesearch,
                    alpha_init, linesearch_tol, max_iter_gs, opt_prob):
    '''
    Line Search

    This function finds returns the step size by either golden section search,
    constant step size, or backtracking. This function also returns the
    objective function value, gradient, and current step after taking the
    step of the found step size.

    Input arguments:
        x_k:
            Current location
        d_k:
            Descent direction
        suff_decrease_factor:
            Factor used in backtracking condition to
            determine sufficient decrease
        linesearch:
            Determines the method for which to find the step size "alpha_k"
            Type of line search:
                ’constant’
                ’golden_section’
                ’backtracking’
        alpha_init:
            Initial step size to consider
        linesearch_tol:
            Tolerance to terminate golden section algorithm
        max_iter_gs:
            Maximum iterations golden section algorithm is allowed to take
        opt_prob:
            Optimization problem object to define an unconstrained optimization
            problem.  It must have the following methods:

            val = value(x)
                returns value of objective function at x
            grad = gradient(x)
                returns gradient of objective function at x
            x_start = starting_point()
                returns a default starting point

    Return values:
        x_k:
            Point after taking a step of size 'alpha_k' in the direction 'd_k'
        f_k:
            Objective function value at x_k
        grad_k:
            Gradient at x_k
        alpha_k:
            Computed step size
        inc_num_f:
            The number of times the objective function (opt_prob.value(x))
            was called
        inc_num_grad:
            The number of times the gradient function (opt_prob.gradient(x))
            was called
    '''

    if linesearch == 'constant':
        # Keep initial step size
        alpha_k = alpha_init
        # Get current point, objective value, and gradient
        x_k = x_k + alpha_k*d_k
        f_k = opt_prob.value(x_k)
        grad_k = opt_prob.gradient(x_k)

        # Increment number of times objective f and gradient were called
        inc_num_f = 1
        inc_num_grad = 1
        return x_k, f_k, grad_k, alpha_k, inc_num_f, inc_num_grad
    elif linesearch == 'golden_section':
        verbose = 0  # No output from golden section
        max_iter = max_iter_gs  # Keep golden section running

        # Define function for golden section in linesearch
        def opt_prob_slice(alpha):
            return opt_prob.value(x_k + alpha*d_k)

        # Call golden section
        status, alpha_k, f_k, inc_num_f = \
            golden_section.minimize(opt_prob_slice, 0, alpha_init,
                                    linesearch_tol, verbose, max_iter)

        # Get current point and gradient
        x_k = x_k + alpha_k*d_k
        grad_k = opt_prob.gradient(x_k)

        # Increment number of times gradient was called
        # Note: inc_num_f was outputted from golden_section
        inc_num_grad = 1
        return x_k, f_k, grad_k, alpha_k, inc_num_f, inc_num_grad
    elif linesearch == 'backtracking':
        alpha_k = alpha_init

        # Left and right side of sufficient decrease condition
        left = opt_prob.value(x_k + alpha_k*d_k)
        transposed_grad = opt_prob.gradient(x_k).T
        right = opt_prob.value(x_k) + \
            alpha_k*suff_decrease_factor*np.dot(transposed_grad, d_k)

        # Increment number of times objective f and gradient were called
        inc_num_f = 2
        inc_num_grad = 1
        while left > right:
            # Cut step size in half
            alpha_k = alpha_k/2

            # Left and right side of condition
            left = opt_prob.value(x_k + alpha_k*d_k)
            transposed_grad = opt_prob.gradient(x_k).T
            right = opt_prob.value(x_k) + \
                alpha_k*suff_decrease_factor*np.dot(transposed_grad, d_k)

            # Increment number of times objective f and gradient were called
            inc_num_f += 2
            inc_num_grad += 1

        # Get current point, objective value, and gradient
        f_k = left
        x_k = x_k + alpha_k*d_k
        grad_k = opt_prob.gradient(x_k)

        # Increment number of time gradient was called
        inc_num_grad += 1
        return x_k, f_k, grad_k, alpha_k, inc_num_f, inc_num_grad
