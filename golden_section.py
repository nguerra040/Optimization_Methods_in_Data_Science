#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Minimize Module

This file is a module which contains the function 'minimize' which can
find the local minimizer of any stictly unimodal, continuous function.

Author: Nicolas Guerra
Date: April 6, 2022
'''

# Make NumPy module available for numerical computations
import numpy as np


def minimize(obj_func, x_L, x_U, tol, verbose, max_iter):
    '''
    This is a function which tries to find the local minimizer of any strictly
    unimodal, continuous function using the golden section algorithm.

    Input arguments:
        obj_func:
            Function to compute values of the objective function.
            signature:

            obj_func(x)
                Input: scalar number x at which function is evaluated
                Return: value of function
        x_L:
            lower bound of search interval
        x_U:
            upper bound of search interval
        tol:
            tolerance at which to terminate algorithm
        verbose:
            determines amount of output:
                0: no output
                1: one line of output per iteration
        max_iter:
            maximum number of iterations

    Return values:
        status:
            reason for termination:
                0: termination tolerance satisfied
                -1: maximum number of iterations exceeded
        x_sol:
            approximate minimizer of obj_func if status = 0, otherwise the
            most recent iterate
        f_sol
            value of objective function at x_sol
        num_iter:
            number of iteration taken by the algorithm
    '''

    # Sanity check of some input values:
    # Checking x_L < x_U
    if x_L >= x_U:
        # The following is a command one can use to have the execution
        # of the program termination with an error message.
        raise Exception('Invalid input: x_L must be less than x_U.')
    # Checking tol is positive
    if tol <= 0:
        raise Exception('Invalid input: tol must be positive.')

    # Fraction to compute x_1 and x_2 and maintains symmetry
    alpha = (np.sqrt(5)-1)/2
    # Computing x_1
    x_1 = x_U - alpha*(x_U-x_L)
    # Computing x_2
    x_2 = x_L + alpha*(x_U-x_L)

    # compute function value at lower bound
    f_L = obj_func(x_L)
    # compute function value at upper bound
    f_U = obj_func(x_U)
    # compute function value at x_1
    f_1 = obj_func(x_1)
    # compute function value at x_2
    f_2 = obj_func(x_2)

    # initialize the iteration counter
    num_iter = 0

    # compute initial search interval
    length = x_U - x_L

    # Assume for now that algorithm will terminate successfully
    status = 0
    # output for first iteration
    if verbose > 0:
        print('\nOptimizing function with golden section search.\n')
        print('Iter|    x_L       f(x_L)   |    x_1       f(x_1)   '
              '|    x_2       f(x_2)   |    x_U       f(x_U)   |   length')
        print('%3d | %10.3e %10.3e | %10.3e %10.3e |'
              ' %10.3e %10.3e | %10.3e %10.3e | %10.3e'
              % (num_iter, x_L, f_L, x_1, f_1, x_2, f_2, x_U, f_U, length))

    # continue until search interval is less than/equal to tolerance
    while (length > tol):

        # increase iteration counter by one (note the Python syntax)
        num_iter += 1

        # Interrupt while loop if the maximum number of iterations is exceeded
        if num_iter > max_iter:  # I put > since comment above says "exceeded"
            status = -1
            # The following command interrupts the current loop (here the while
            # loop) and continues execution after the loop
            break

        # check if x_1 has worse objective value
        if f_1 > f_2:
            # reassign x's and f's accordingly; eliminate left-most segment
            x_L = x_1
            f_L = f_1
            x_1 = x_2
            f_1 = f_2

            # create new point in right-most segment
            x_2 = x_L + alpha*(x_U-x_L)
            # compute function at x_2
            f_2 = obj_func(x_2)

        # else x_2 has worse objective value
        else:
            # reassign x's and f's accordingly; eliminate right-most segment
            x_U = x_2
            f_U = f_2
            x_2 = x_1
            f_2 = f_1

            # create new point in left-most segment
            x_1 = x_U - alpha*(x_U-x_L)
            # compute function at x_1
            f_1 = obj_func(x_1)

        # reevaluate search interval length
        length = x_U - x_L

        # output for this iteration
        if verbose > 0:
            # Print header all 10 iterations
            if num_iter % 10 == 0:
                print('Iter|    x_L       f(x_L)   |    x_1       f(x_1)   '
                      '|    x_2       f(x_2)   |    x_U       f(x_U)   '
                      '|   length')

            print('%3d | %10.3e %10.3e | %10.3e %10.3e |'
                  ' %10.3e %10.3e | %10.3e %10.3e | %10.3e'
                  % (num_iter, x_L, f_L, x_1, f_1, x_2, f_2, x_U, f_U, length))

    # calculate middle of bound
    x_sol = 0.5*(x_U+x_L)
    # compute function at x_sol
    f_sol = obj_func(x_sol)

    # Final output
    if verbose > 0:
        print('')
        if status == 0:
            # successful termination
            print('Successful termination!')
            print('Optimal solution........: %f' % x_sol)
            print('Optimal objective value : %f' % f_sol)
            print('Number of iterations....: %d' % num_iter)
        elif status == -1:
            # maximum number of iterations exceeded
            print('Convergence failure: Maximum number of iterations exceeded')
            print('Last iterate............: %f' % x_sol)
            print('Last objective value... : %f' % f_sol)
            print('Number of iterations....: %d' % num_iter)
        else:
            # Check if status has an unintended value (sanity checks like this
            # help a lot when debugging code, particularly if it is large and
            # evolving code)
            raise Exception('Internal bug: invalid value of status')

    # Return output arguments
    return status, x_sol, f_sol, num_iter
