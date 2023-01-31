# -*- coding: utf-8 -*-
"""
Gradient Descent Algorithm

Author: Nicolas Guerra
Due Date: 5/6/2022

This python file uses the Gradient Descent algorithm to find the minimum of a
function.
"""

# Import dependencies
import numpy as np


def init_options():
    '''
    Option initialization for the gradient descent algorithm

    Initialize algorithm options with default values

    Return values:
        options:
            This is a dictionary with fields that correspond to algorithmic
            options of our method.  In particular:

            max_iter:
                Maximum number of iterations
            tol:
                Convergence tolerance
                Algorithm stops if ||gradient||_inf <= tol
            step_size:
                Step size in the gradient descent algorithm
            output_level:
                Amount of output to be printed:
                    0 : No output
                    1 : Only summary information
                    2 : One line per iteration (good for debugging)
                    3 : Print iterates in each iteration
    '''
    options = {}
    # Default values
    options['max_iter'] = 1000
    options['tol'] = 1e-6
    options['step_size'] = 0.1
    options['output_level'] = 1

    return options


def minimize(opt_prob, options, x_start=None):
    '''
    Optimization methods for unconstrainted optimization

    This is an implementation of the gradient descent method for unconstrained
    optimization.

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
    step_size = options['step_size']
    output_level = options['output_level']

    # return flag
    # set to incomplete so that status has to be set explicitly in method
    status = 'incomplete'

    # get starting point. If none is provided explicitly for this call,
    # ask the OptProblem object for it.
    if x_start is None:
        x_start = opt_prob.starting_point()
    x_k = np.copy(x_start)

    # current function value, gradient, and gradient norm
    f_k = opt_prob.value(x_k)
    grad_k = opt_prob.gradient(x_k)
    norm_grad = np.linalg.norm(grad_k, np.inf)

    # Initialize iteration number
    num_iter = 0
    if output_level == 2:
        print('iter       f                ||grad||_inf')
        print('%3d       %10.3e       %10.3e' % (num_iter, f_k, norm_grad))
    elif output_level == 3:
        print('iter       f                ||grad||_inf')
        print('%3d       %10.3e       %10.3e' % (num_iter, f_k, norm_grad))
        print("iterate: ", x_k)

    # Gradient method body
    while norm_grad > tol:

        # Check if we are beyond max iteration limit on next iteration
        if max_iter < num_iter+1:
            status = 'iteration limit'
            break
        # Increment iteration number
        num_iter += 1

        # Calculate new step
        x_k = x_k - step_size * grad_k
        # current function value, gradient, and gradient norm
        f_k = opt_prob.value(x_k)
        grad_k = opt_prob.gradient(x_k)
        norm_grad = np.linalg.norm(grad_k, np.inf)

        # Print out progress so far
        if output_level == 2:
            # Print header every 10 iterations
            if num_iter % 10 == 0:
                print('iter       f                ||grad||_inf')
            print('%3d       %10.3e       %10.3e' % (num_iter, f_k, norm_grad))
        elif output_level == 3:
            # Print header every 10 iterations
            if num_iter % 10 == 0:
                print('iter       f                ||grad||_inf')
            print('%3d       %10.3e       %10.3e' % (num_iter, f_k, norm_grad))
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
        print('Number of Iterations: %3d' % num_iter)
        print('Objective value: %10.3e' % f_sol)
        print('Infinity-norm of gradient: %10.3e' % norm_grad)
        print('Computed Solution: ', x_sol)

    # Return output arguments
    return status, x_sol, f_sol
