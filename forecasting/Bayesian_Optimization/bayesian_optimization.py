# pylint: disable-all
import numbers
import random
import warnings
from math import log
from typing import Any, Callable, List, Optional, Union

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import clone, is_regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state
from skopt import Space
from skopt.acquisition import _gaussian_acquisition, gaussian_acquisition_1D
from skopt.callbacks import VerboseCallback, check_callback
from skopt.learning import (ExtraTreesRegressor, GaussianProcessRegressor,
                            GradientBoostingQuantileRegressor,
                            RandomForestRegressor)
from skopt.learning.gaussian_process.kernels import (ConstantKernel,
                                                     HammingKernel, Matern)
from skopt.optimizer import Optimizer
from skopt.utils import (check_x_in_space, create_result, eval_callbacks,
                         is_2Dlistlike, is_listlike, normalize_dimensions)

GPR_CHOLESKY_LOWER = True


class BO_with_Dropout:
    def __init__(self, p, d, D):
        self.p = p
        self.d = d  # number of dimensions considered in acquisition function
        self.D = D  # total number of dimensions
        self.best_x = None
        self.best_y = None
        self.dims = None
        self.res_x = []
        self.res_y = []
        self.gl_dimensions = None
        self.gl_base_estimator_str = None

    def adapt_next_x(self, next_x: List[Any], space: Space, rnd: int) -> List[Any]:
        """
        Applies "fill-up"-strategy when conducting BO with dimension dropout
        and adapts next promising parameter configuration accordingly

        Parameters:
        next_x: list: List[Any]
            List with parameter values obtained by BO
        space: Space
            Space object from which values are randomly drawn
        rnd: int
            random_state used for drawing values from Space

        Returns:
        next_x_new: List[Any]
            List with adapted parameter values based on fill-up-strategy
        """
        next_x_new = []
        rnd_p = random.uniform(0, 1)
        if rnd_p < self.p:
            ref_arr = self.best_x  # use incumbent parameter values
        else:
            ref_arr = space.rvs(random_state=rnd)[0]  # draw completely random values
        o = 0
        # generate new next_x instance
        for i in range(len(self.dims)):
            if not self.dims[i]:
                next_x_new.append(ref_arr[i])
            else:
                next_x_new.append(next_x[o])
                o += 1

        return next_x_new

    def draw_dimensions(self) -> np.ndarray:
        """
        Sets dimensions considered in acquisition function.

        Returns
        dim: np.ndarray
            boolean-array stating which dimensions to consider
        """
        dim = np.zeros(self.D)  # D: global parameter, set to total number of dimensions
        while sum(dim) < self.d:
            dim[
                random.randrange(self.D)
            ] = 1  # d: global parameter, set to number of dimensions considered in acquisition function
        dim = dim.astype(bool)
        return dim

    def custom_base_minimize(
        self,
        func: Callable,
        dimensions: List[Any],
        base_estimator: Any,
        n_calls: int = 100,
        n_random_starts: Optional[int] = None,
        n_initial_points: int = 10,
        initial_point_generator: str = "random",
        acq_func: str = "EI",
        acq_optimizer: str = "lbfgs",
        x0: Optional[List[Any]] = None,
        y0: Union[List[Any], float, None] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        callback: Union[None, Callable, List[Callable]] = None,
        n_points: int = 10000,
        n_restarts_optimizer: int = 5,
        xi: float = 0.01,
        kappa: float = 1.96,
        n_jobs: int = 1,
        model_queue_size: Optional[int] = None,
    ):
        """Base optimizer class

        Parameters
        ----------
        func : callable
            Function to minimize. Should take a single list of parameters
            and return the objective value.
        
            If you have a search-space where all dimensions have names,
            then you can use :func:`skopt.utils.use_named_args` as a decorator
            on your objective function, in order to call it directly
            with the named arguments. See `use_named_args` for an example.

        dimensions : list, shape (n_dims,)
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
            dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
            dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
            `Categorical`).

            .. note:: The upper and lower bounds are inclusive for `Integer`
                dimensions.

        base_estimator : sklearn regressor
            Should inherit from `sklearn.base.RegressorMixin`.
            In addition, should have an optional `return_std` argument,
            which returns `std(Y | x)` along with `E[Y | x]`.

        n_calls : int, default: 100
            Maximum number of calls to `func`. An objective function will
            always be evaluated this number of times; Various options to
            supply initialization points do not affect this value.

        n_random_starts : int, default: None
            Number of evaluations of `func` with random points before
            approximating it with `base_estimator`.

            .. deprecated:: 0.8
                use `n_initial_points` instead.

        n_initial_points : int, default: 10
            Number of evaluations of `func` with initialization points
            before approximating it with `base_estimator`. Initial point
            generator can be changed by setting `initial_point_generator`.

        initial_point_generator : str, InitialPointGenerator instance, \
                default: `"random"`
            Sets a initial points generator. Can be either

            - `"random"` for uniform random numbers,
            - `"sobol"` for a Sobol' sequence,
            - `"halton"` for a Halton sequence,
            - `"hammersly"` for a Hammersly sequence,
            - `"lhs"` for a latin hypercube sequence,
            - `"grid"` for a uniform grid sequence

        acq_func : string, default: `"EI"`
            Function to minimize over the posterior distribution. Can be either

            - `"LCB"` for lower confidence bound,
            - `"EI"` for negative expected improvement,
            - `"PI"` for negative probability of improvement.
            - `"EIps"` for negated expected improvement per second to take into
            account the function compute time. Then, the objective function is
            assumed to return two values, the first being the objective value and
            the second being the time taken in seconds.
            - `"PIps"` for negated probability of improvement per second. The
            return type of the objective function is assumed to be similar to
            that of `"EIps"`

        acq_optimizer : string, `"sampling"` or `"lbfgs"`, default: `"lbfgs"`
            Method to minimize the acquisition function. The fit model
            is updated with the optimal value obtained by optimizing `acq_func`
            with `acq_optimizer`.

            - If set to `"sampling"`, then `acq_func` is optimized by computing
            `acq_func` at `n_points` randomly sampled points and the smallest
            value found is used.
            - If set to `"lbfgs"`, then

            - The `n_restarts_optimizer` no. of points which the acquisition
                function is least are taken as start points.
            - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
            - The optimal of these local minima is used to update the prior.

        x0 : list, list of lists or `None`
            Initial input points.

            - If it is a list of lists, use it as a list of input points. If no
            corresponding outputs `y0` are supplied, then len(x0) of total
            calls to the objective function will be spent evaluating the points
            in `x0`. If the corresponding outputs are provided, then they will
            be used together with evaluated points during a run of the algorithm
            to construct a surrogate.
            - If it is a list, use it as a single initial input point. The
            algorithm will spend 1 call to evaluate the initial point, if the
            outputs are not provided.
            - If it is `None`, no initial input points are used.

        y0 : list, scalar or `None`
            Objective values at initial input points.

            - If it is a list, then it corresponds to evaluations of the function
            at each element of `x0` : the i-th element of `y0` corresponds
            to the function evaluated at the i-th element of `x0`.
            - If it is a scalar, then it corresponds to the evaluation of the
            function at `x0`.
            - If it is None and `x0` is provided, then the function is evaluated
            at each element of `x0`.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        verbose : boolean, default: False
            Control the verbosity. It is advised to set the verbosity to True
            for long optimization runs.

        callback : callable, list of callables, optional
            If callable then `callback(res)` is called after each call to `func`.
            If list of callables, then each callable in the list is called.

        n_points : int, default: 10000
            If `acq_optimizer` is set to `"sampling"`, then `acq_func` is
            optimized by computing `acq_func` at `n_points` randomly sampled
            points.

        n_restarts_optimizer : int, default: 5
            The number of restarts of the optimizer when `acq_optimizer`
            is `"lbfgs"`.

        xi : float, default: 0.01
            Controls how much improvement one wants over the previous best
            values. Used when the acquisition is either `"EI"` or `"PI"`.

        kappa : float, default: 1.96
            Controls how much of the variance in the predicted values should be
            taken into account. If set to be very high, then we are favouring
            exploration over exploitation and vice versa.
            Used when the acquisition is `"LCB"`.

        n_jobs : int, default: 1
            Number of cores to run in parallel while running the lbfgs
            optimizations over the acquisition function and given to
            the base_estimator. Valid only when
            `acq_optimizer` is set to "lbfgs". or when the base_estimator
            supports n_jobs as parameter and was given as string.
            Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
            to number of cores.

        model_queue_size : int or None, default: None
            Keeps list of models only as long as the argument given. In the
            case of None, the list has no capped length.

        Returns
        -------
        res : `OptimizeResult`, scipy object
            The optimization result returned as a OptimizeResult object.
            Important attributes are:

            - `x` [list]: location of the minimum.
            - `fun` [float]: function value at the minimum.
            - `models`: surrogate models used for each iteration.
            - `x_iters` [list of lists]: location of function evaluation for each
            iteration.
            - `func_vals` [array]: function value for each iteration.
            - `space` [Space]: the optimization space.
            - `specs` [dict]`: the call specifications.
            - `rng` [RandomState instance]: State of the random state
            at the end of minimization.

            For more details related to the OptimizeResult object, refer
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        """

        specs = {"args": locals(), "function": "base_minimize"}

        acq_optimizer_kwargs = {
            "n_points": n_points,
            "n_restarts_optimizer": n_restarts_optimizer,
            "n_jobs": n_jobs,
        }
        acq_func_kwargs = {"xi": xi, "kappa": kappa}

        # Initialize optimization
        # Suppose there are points provided (x0 and y0), record them

        # check x0: list-like, requirement of minimal points
        if x0 is None:
            x0 = []
        elif not isinstance(x0[0], (list, tuple)):
            x0 = [x0]
        if not isinstance(x0, list):
            raise ValueError("`x0` should be a list, but got %s" % type(x0))

        # Adaptation Start
        # document results and update incumbent parameter configuration used for fill-up strategy
        if x0 != None and len(x0) > 0 and isinstance(x0[0], (list, tuple)):
            self.res_x.extend(x0)
            self.best_x = x0[np.argmin(y0)]
            if y0 is not None:
                self.res_y.extend(y0)
                self.best_y = min(y0)
        # Adaptation End

        # Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(
                (
                    "n_random_starts will be removed in favour of "
                    "n_initial_points. It overwrites n_initial_points."
                ),
                DeprecationWarning,
            )
            n_initial_points = n_random_starts

        if n_initial_points <= 0 and not x0:
            raise ValueError("Either set `n_initial_points` > 0," " or provide `x0`")
        # check y0: list-like, requirement of maximal calls
        if isinstance(y0, Iterable):
            y0 = list(y0)
        elif isinstance(y0, numbers.Number):
            y0 = [y0]
        required_calls = n_initial_points + (len(x0) if not y0 else 0)
        if n_calls < required_calls:
            raise ValueError(
                "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls)
            )
        # calculate the total number of initial points
        n_initial_points = n_initial_points + len(x0)

        # Build optimizer

        # create optimizer class
        optimizer = Optimizer(
            dimensions,
            base_estimator,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            n_jobs=n_jobs,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer,
            random_state=random_state,
            model_queue_size=model_queue_size,
            acq_optimizer_kwargs=acq_optimizer_kwargs,
            acq_func_kwargs=acq_func_kwargs,
        )
        # check x0: element-wise data type, dimensionality
        assert all(isinstance(p, Iterable) for p in x0)
        if not all(len(p) == optimizer.space.n_dims for p in x0):
            raise RuntimeError(
                "Optimization space (%s) and initial points in x0 "
                "use inconsistent dimensions." % optimizer.space
            )
        # check callback
        callbacks = check_callback(callback)
        if verbose:
            callbacks.append(
                VerboseCallback(
                    n_init=len(x0) if not y0 else 0,
                    n_random=n_initial_points,
                    n_total=n_calls,
                )
            )

        # Record provided points

        # create return object
        result = None
        # evaluate y0 if only x0 is provided
        if x0 and y0 is None:
            y0 = list(map(func, x0))
            n_calls -= len(y0)
        # record through tell function
        if x0:

            # Adaptation Start
            # Randomly draw dimensions considered in acquisition function
            self.dims = self.draw_dimensions()
            # Adaptation End

            if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
                raise ValueError(
                    "`y0` should be an iterable or a scalar, got %s" % type(y0)
                )
            if len(x0) != len(y0):
                raise ValueError("`x0` and `y0` should have the same length")

            # Adaptation Start
            # apply custom tell function to only consider drawn dimensions in aquisition function
            result = self.custom_tell(optimizer, x0, y0)
            # Adaptation End

            result.specs = specs
            if eval_callbacks(callbacks, result):
                return result

        # Optimize
        for n in range(n_calls):

            # adapt optimizer._next_x as this is asked in next function
            # fill up missing values with adapt function
            if (
                self.res_x != None
                and len(self.res_x) >= n_initial_points
                and optimizer._next_x != None
            ):
                optimizer._next_x = self.adapt_next_x(
                    optimizer._next_x, optimizer.space, optimizer.rng
                )

            # get next_x from optimizer (has been calculated in tell())
            next_x = optimizer.ask()
            self.res_x.append(next_x)

            # run model for next_x
            next_y = func(next_x)
            self.res_y.append(next_y)

            # adapt data basis before determining next_x in tell()
            # only needed after n random drawings
            if len(self.res_x) >= n_initial_points:
                self.dims = self.draw_dimensions()

            result = self.custom_tell(optimizer, next_x, next_y)
            # result = optimizer.tell(next_x, next_y)
            result.specs = specs

            # document results and update best combination
            if self.best_y == None or result["func_vals"][-1] < self.best_y:
                self.best_y = result["func_vals"][-1]
                self.best_x = next_x

            if eval_callbacks(callbacks, result):
                break

        return result

    def custom_gp_minimize(
        self,
        func: Callable,
        dimensions: List[Any],
        base_estimator: Any = None,
        n_calls: int = 100,
        n_random_starts: Optional[int] = None,
        n_initial_points: int = 10,
        initial_point_generator: str = "random",
        acq_func: str = "gp_hedge",
        acq_optimizer: str = "auto",
        x0: Optional[List[Any]] = None,
        y0: Union[None, List[Any], float] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
        callback: Union[Callable, List[Callable], None] = None,
        n_points: int = 10000,
        n_restarts_optimizer: int = 5,
        xi: float = 0.01,
        kappa: float = 1.96,
        noise: str = "gaussian",
        n_jobs: int = 1,
        model_queue_size: Optional[int] = None,
    ):
        """Bayesian optimization using Gaussian Processes.

        If every function evaluation is expensive, for instance
        when the parameters are the hyperparameters of a neural network
        and the function evaluation is the mean cross-validation score across
        ten folds, optimizing the hyperparameters by standard optimization
        routines would take for ever!

        The idea is to approximate the function using a Gaussian process.
        In other words the function values are assumed to follow a multivariate
        gaussian. The covariance of the function values are given by a
        GP kernel between the parameters. Then a smart choice to choose the
        next parameter to evaluate can be made by the acquisition function
        over the Gaussian prior which is much quicker to evaluate.

        The total number of evaluations, `n_calls`, are performed like the
        following. If `x0` is provided but not `y0`, then the elements of `x0`
        are first evaluated, followed by `n_initial_points` evaluations.
        Finally, `n_calls - len(x0) - n_initial_points` evaluations are
        made guided by the surrogate model. If `x0` and `y0` are both
        provided then `n_initial_points` evaluations are first made then
        `n_calls - n_initial_points` subsequent evaluations are made
        guided by the surrogate model.

        The first `n_initial_points` are generated by the
        `initial_point_generator`.

        Parameters
        ----------
        func : callable
            Function to minimize. Should take a single list of parameters
            and return the objective value.
        
            If you have a search-space where all dimensions have names,
            then you can use :func:`skopt.utils.use_named_args` as a decorator
            on your objective function, in order to call it directly
            with the named arguments. See `use_named_args` for an example.

        dimensions : [list, shape (n_dims,)
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
            dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
            dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
            `Categorical`).

            .. note:: The upper and lower bounds are inclusive for `Integer`
                dimensions.

        base_estimator : a Gaussian process estimator
            The Gaussian process estimator to use for optimization.
            By default, a Matern kernel is used with the following
            hyperparameters tuned.

            - All the length scales of the Matern kernel.
            - The covariance amplitude that each element is multiplied with.
            - Noise that is added to the matern kernel. The noise is assumed
            to be iid gaussian.

        n_calls : int, default: 100
            Number of calls to `func`.

        n_random_starts : int, default: None
            Number of evaluations of `func` with random points before
            approximating it with `base_estimator`.

            .. deprecated:: 0.8
                use `n_initial_points` instead.

        n_initial_points : int, default: 10
            Number of evaluations of `func` with initialization points
            before approximating it with `base_estimator`. Initial point
            generator can be changed by setting `initial_point_generator`.

        initial_point_generator : str, InitialPointGenerator instance, \
                default: 'random'
            Sets a initial points generator. Can be either

            - `"random"` for uniform random numbers,
            - `"sobol"` for a Sobol' sequence,
            - `"halton"` for a Halton sequence,
            - `"hammersly"` for a Hammersly sequence,
            - `"lhs"` for a latin hypercube sequence,

        acq_func : string, default: `"gp_hedge"`
            Function to minimize over the gaussian prior. Can be either

            - `"LCB"` for lower confidence bound.
            - `"EI"` for negative expected improvement.
            - `"PI"` for negative probability of improvement.
            - `"gp_hedge"` Probabilistically choose one of the above three
            acquisition functions at every iteration. The weightage
            given to these gains can be set by :math:`\\eta` through
            `acq_func_kwargs`.

            - The gains `g_i` are initialized to zero.
            - At every iteration,

                - Each acquisition function is optimised independently to
                propose an candidate point `X_i`.
                - Out of all these candidate points, the next point `X_best` is
                chosen by :math:`softmax(\\eta g_i)`
                - After fitting the surrogate model with `(X_best, y_best)`,
                the gains are updated such that :math:`g_i -= \\mu(X_i)`

            - `"EIps"` for negated expected improvement per second to take into
            account the function compute time. Then, the objective function is
            assumed to return two values, the first being the objective value and
            the second being the time taken in seconds.
            - `"PIps"` for negated probability of improvement per second. The
            return type of the objective function is assumed to be similar to
            that of `"EIps"`

        acq_optimizer : string, `"sampling"` or `"lbfgs"`, default: `"lbfgs"`
            Method to minimize the acquisition function. The fit model
            is updated with the optimal value obtained by optimizing `acq_func`
            with `acq_optimizer`.

            The `acq_func` is computed at `n_points` sampled randomly.

            - If set to `"auto"`, then `acq_optimizer` is configured on the
            basis of the space searched over.
            If the space is Categorical then this is set to be `"sampling"`.
            - If set to `"sampling"`, then the point among these `n_points`
            where the `acq_func` is minimum is the next candidate minimum.
            - If set to `"lbfgs"`, then

            - The `n_restarts_optimizer` no. of points which the acquisition
                function is least are taken as start points.
            - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
            - The optimal of these local minima is used to update the prior.

        x0 : list, list of lists or `None`
            Initial input points.

            - If it is a list of lists, use it as a list of input points.
            - If it is a list, use it as a single initial input point.
            - If it is `None`, no initial input points are used.

        y0 : list, scalar or `None`
            Evaluation of initial input points.

            - If it is a list, then it corresponds to evaluations of the function
            at each element of `x0` : the i-th element of `y0` corresponds
            to the function evaluated at the i-th element of `x0`.
            - If it is a scalar, then it corresponds to the evaluation of the
            function at `x0`.
            - If it is None and `x0` is provided, then the function is evaluated
            at each element of `x0`.

        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        verbose : boolean, default: False
            Control the verbosity. It is advised to set the verbosity to True
            for long optimization runs.

        callback : callable, list of callables, optional
            If callable then `callback(res)` is called after each call to `func`.
            If list of callables, then each callable in the list is called.

        n_points : int, default: 10000
            Number of points to sample to determine the next "best" point.
            Useless if acq_optimizer is set to `"lbfgs"`.

        n_restarts_optimizer : int, default: 5
            The number of restarts of the optimizer when `acq_optimizer`
            is `"lbfgs"`.

        kappa : float, default: 1.96
            Controls how much of the variance in the predicted values should be
            taken into account. If set to be very high, then we are favouring
            exploration over exploitation and vice versa.
            Used when the acquisition is `"LCB"`.

        xi : float, default: 0.01
            Controls how much improvement one wants over the previous best
            values. Used when the acquisition is either `"EI"` or `"PI"`.

        noise : float, default: "gaussian"

            - Use noise="gaussian" if the objective returns noisy observations.
            The noise of each observation is assumed to be iid with
            mean zero and a fixed variance.
            - If the variance is known before-hand, this can be set directly
            to the variance of the noise.
            - Set this to a value close to zero (1e-10) if the function is
            noise-free. Setting to zero might cause stability issues.

        n_jobs : int, default: 1
            Number of cores to run in parallel while running the lbfgs
            optimizations over the acquisition function. Valid only
            when `acq_optimizer` is set to `"lbfgs"`.
            Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
            to number of cores.

        model_queue_size : int or None, default: None
            Keeps list of models only as long as the argument given. In the
            case of None, the list has no capped length.

        Returns
        -------
        res : `OptimizeResult`, scipy object
            The optimization result returned as a OptimizeResult object.
            Important attributes are:

            - `x` [list]: location of the minimum.
            - `fun` [float]: function value at the minimum.
            - `models`: surrogate models used for each iteration.
            - `x_iters` [list of lists]: location of function evaluation for each
            iteration.
            - `func_vals` [array]: function value for each iteration.
            - `space` [Space]: the optimization space.
            - `specs` [dict]`: the call specifications.
            - `rng` [RandomState instance]: State of the random state
            at the end of minimization.

            For more details related to the OptimizeResult object, refer
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html

        .. seealso:: functions :class:`skopt.forest_minimize`,
            :class:`skopt.dummy_minimize`, :class:`skopt.gbrt_minimize`

        """
        # Check params
        rng = check_random_state(random_state)
        space = normalize_dimensions(dimensions)

        # Adaptation Start
        # Always apply GP for BO with dimension dropout
        self.gl_base_estimator_str = "GP"
        self.gl_dimensions = dimensions
        # Adaptation End

        if base_estimator is None:
            base_estimator = self.custom_cook_estimator(
                base_estimator="GP",
                space=space,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                noise=noise,
            )

        return self.custom_base_minimize(
            func=func,
            dimensions=space,
            base_estimator=base_estimator,
            acq_func=acq_func,
            xi=xi,
            kappa=kappa,
            acq_optimizer=acq_optimizer,
            n_calls=n_calls,
            n_points=n_points,
            n_random_starts=n_random_starts,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            n_restarts_optimizer=n_restarts_optimizer,
            x0=x0,
            y0=y0,
            random_state=rng,
            verbose=verbose,
            callback=callback,
            n_jobs=n_jobs,
            model_queue_size=model_queue_size,
        )

    def _custom_tell(
        self, optimizer: Optimizer, x: List[Any], y: Union[float, List[float]], fit=True
    ) -> Any:
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer class

        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        Returns
        ----------
        res : `OptimizeResult`, scipy object
            The optimization result returned as a OptimizeResult object.
            Important attributes are:

            - `x` [list]: location of the minimum.
            - `fun` [float]: function value at the minimum.
            - `models`: surrogate models used for each iteration.
            - `x_iters` [list of lists]: location of function evaluation for each
            iteration.
            - `func_vals` [array]: function value for each iteration.
            - `space` [Space]: the optimization space.
            - `specs` [dict]`: the call specifications.
            - `rng` [RandomState instance]: State of the random state
            at the end of minimization.

            For more details related to the OptimizeResult object, refer
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        """

        # We add new points to Xi
        if "ps" in optimizer.acq_func:
            if is_2Dlistlike(x):
                optimizer.Xi.extend(x)
                optimizer.yi.extend(y)
                optimizer._n_initial_points -= len(y)
            elif is_listlike(x):
                optimizer.Xi.append(x)
                optimizer.yi.append(y)
                optimizer._n_initial_points -= 1
        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            optimizer.Xi.extend(x)
            optimizer.yi.extend(y)
            optimizer._n_initial_points -= len(y)
        elif is_listlike(x):
            optimizer.Xi.append(x)
            optimizer.yi.append(y)
            optimizer._n_initial_points -= 1
        else:
            raise ValueError(
                "Type of arguments `x` (%s) and `y` (%s) "
                "not compatible." % (type(x), type(y))
            )

        # optimizer learned something new - discard cache
        optimizer.cache_ = {}

        # Adaptation Start
        # adapt space and parameter configurations such that only
        # dimensions considered in acquisition function are optimized.
        if optimizer._n_initial_points <= 0:

            self_Xi_copy = optimizer.Xi
            self_space_copy = optimizer.space

            space = normalize_dimensions(
                (np.array(self.gl_dimensions)[self.dims]).tolist()
            )
            optimizer.base_estimator_ = self.custom_cook_estimator(
                base_estimator=self.gl_base_estimator_str,
                space=space,
                random_state=optimizer.base_estimator_.random_state,
                noise=optimizer.base_estimator_.noise,
            )

            if isinstance(x[0], list):
                x = (np.array(x, dtype=object)[:, self.dims]).tolist()
            else:
                x = (np.array(x, dtype=object)[self.dims]).tolist()
            optimizer.Xi = (np.array(optimizer.Xi, dtype=object)[:, self.dims]).tolist()
            optimizer.space = space
        # Adaptation End

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if (
            fit
            and optimizer._n_initial_points <= 0
            and optimizer.base_estimator_ is not None
        ):
            transformed_bounds = np.array(optimizer.space.transformed_bounds)
            est = clone(optimizer.base_estimator_)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(optimizer.space.transform(optimizer.Xi), optimizer.yi)

            if hasattr(optimizer, "next_xs_") and optimizer.acq_func == "gp_hedge":
                optimizer.gains_ -= est.predict(np.vstack(optimizer.next_xs_))

            if optimizer.max_model_queue_size is None:
                optimizer.models.append(est)
            elif len(optimizer.models) < optimizer.max_model_queue_size:
                optimizer.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                optimizer.models.pop(0)
                optimizer.models.append(est)

            # Draw 10,000 samples

            # even with BFGS as optimizer we want to sample a large number
            # of points and then pick the best ones as starting points
            X = optimizer.space.transform(
                optimizer.space.rvs(
                    n_samples=optimizer.n_points, random_state=optimizer.rng
                )
            )

            optimizer.next_xs_ = []
            for cand_acq_func in optimizer.cand_acq_funcs_:
                values = _gaussian_acquisition(
                    X=X,
                    model=est,
                    y_opt=np.min(optimizer.yi),
                    acq_func=cand_acq_func,
                    acq_func_kwargs=optimizer.acq_func_kwargs,
                )
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                if optimizer.acq_optimizer == "sampling":
                    next_x = X[np.argmin(values)]

                # Use BFGS to find the mimimum of the acquisition function, the
                # minimization starts from `n_restarts_optimizer` different
                # points and the best minimum is used
                elif optimizer.acq_optimizer == "lbfgs":
                    x0 = X[np.argsort(values)[: optimizer.n_restarts_optimizer]]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        results = Parallel(n_jobs=optimizer.n_jobs)(
                            delayed(fmin_l_bfgs_b)(
                                gaussian_acquisition_1D,
                                x,
                                args=(
                                    est,
                                    np.min(optimizer.yi),
                                    cand_acq_func,
                                    optimizer.acq_func_kwargs,
                                ),
                                bounds=optimizer.space.transformed_bounds,
                                approx_grad=False,
                                maxiter=20,
                            )
                            for x in x0
                        )

                    cand_xs = np.array([r[0] for r in results])
                    cand_acqs = np.array([r[1] for r in results])
                    next_x = cand_xs[np.argmin(cand_acqs)]

                # lbfgs should handle this but just in case there are
                # precision errors.
                if not optimizer.space.is_categorical:
                    next_x = np.clip(
                        next_x, transformed_bounds[:, 0], transformed_bounds[:, 1]
                    )
                optimizer.next_xs_.append(next_x)

            if optimizer.acq_func == "gp_hedge":
                logits = np.array(optimizer.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(optimizer.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = optimizer.next_xs_[
                    np.argmax(optimizer.rng.multinomial(1, probs))
                ]
            else:
                next_x = optimizer.next_xs_[0]

            # note the need for [0] at the end
            optimizer._next_x = optimizer.space.inverse_transform(
                next_x.reshape((1, -1))
            )[0]

        # Adaptation Start
        # after acquisition function has been optimized, re-adapt space
        # and parameter configurations such that all dimensions are considered again.
        # this is necessary, as in next interation other dimensions are drawn.
        if optimizer._n_initial_points <= 0:
            optimizer.Xi = self_Xi_copy
            optimizer.space = self_space_copy
        # Adaptation End

        # Pack results
        result = create_result(
            optimizer.Xi,
            optimizer.yi,
            optimizer.space,
            optimizer.rng,
            models=optimizer.models,
        )

        result.specs = optimizer.specs
        return result

    def custom_tell(
        self,
        optimizer: Optimizer,
        x: List[Any],
        y: Union[float, List[float]],
        fit: bool = True,
    ):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        optimizer : Optimizer
            Optimizer class

        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.

        Returns
        -------
        res : `OptimizeResult`, scipy object
            The optimization result returned as a OptimizeResult object.
            Important attributes are:

            - `x` [list]: location of the minimum.
            - `fun` [float]: function value at the minimum.
            - `models`: surrogate models used for each iteration.
            - `x_iters` [list of lists]: location of function evaluation for each
            iteration.
            - `func_vals` [array]: function value for each iteration.
            - `space` [Space]: the optimization space.
            - `specs` [dict]`: the call specifications.
            - `rng` [RandomState instance]: State of the random state
            at the end of minimization.

            For more details related to the OptimizeResult object, refer
            http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        """
        check_x_in_space(x, optimizer.space)
        optimizer._check_y_is_valid(x, y)

        # take the logarithm of the computation times
        if "ps" in optimizer.acq_func:
            if is_2Dlistlike(x):
                y = [[val, log(t)] for (val, t) in y]
            elif is_listlike(x):
                y = list(y)
                y[1] = log(y[1])

        # Adaptation Start
        # apply custom tell function
        return self._custom_tell(optimizer, x, y, fit=fit)
        # Adaptation End

    def custom_cook_estimator(
        self, base_estimator: str, space: Optional[Space] = None, **kwargs
    ) -> Any:
        """Cook a default estimator.

        For the special base_estimator called "DUMMY" the return value is None.
        This corresponds to sampling points at random, hence there is no need
        for an estimator.

        Parameters
        ----------
        base_estimator : "GP", "RF", "ET", "GBRT", "DUMMY" or sklearn regressor
            Should inherit from `sklearn.base.RegressorMixin`.
            In addition the `predict` method should have an optional `return_std`
            argument, which returns `std(Y | x)`` along with `E[Y | x]`.
            If base_estimator is one of ["GP", "RF", "ET", "GBRT", "DUMMY"], a
            surrogate model corresponding to the relevant `X_minimize` function
            is created.

        space : Space instance
            Has to be provided if the base_estimator is a gaussian process.
            Ignored otherwise.

        kwargs : dict
            Extra parameters provided to the base_estimator at init time.
        """
        if isinstance(base_estimator, str):
            base_estimator = base_estimator.upper()
            if base_estimator not in ["GP", "ET", "RF", "GBRT", "DUMMY"]:
                raise ValueError(
                    "Valid strings for the base_estimator parameter "
                    " are: 'RF', 'ET', 'GP', 'GBRT' or 'DUMMY' not "
                    "%s." % base_estimator
                )
        elif not is_regressor(base_estimator):
            raise ValueError("base_estimator has to be a regressor.")

        if base_estimator == "GP":
            if space is not None:
                space = Space(space)
                space = Space(normalize_dimensions(space.dimensions))
                n_dims = space.transformed_n_dims
                # Adaptation Start
                # Don't use HammingKernel, as "Not Implemented Error" is thrown
                # when calling "grad = self.kernel_.gradient_x(X[0], self.X_train_)" in
                # "predict"-function in skopt.learning.gaussian_process.grp.py
                is_cat = False
                # Adaptation End
            else:
                raise ValueError("Expected a Space instance, not None.")

            cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
            # only special if *all* dimensions are categorical
            if is_cat:
                other_kernel = HammingKernel(length_scale=np.ones(n_dims))
            else:
                other_kernel = Matern(
                    length_scale=np.ones(n_dims),
                    length_scale_bounds=[(0.01, 100)] * n_dims,
                    nu=2.5,
                )

            base_estimator = GaussianProcessRegressor(
                kernel=cov_amplitude * other_kernel,
                normalize_y=True,
                noise="gaussian",
                n_restarts_optimizer=2,
            )
        elif base_estimator == "RF":
            base_estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
        elif base_estimator == "ET":
            base_estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3)
        elif base_estimator == "GBRT":
            gbrt = GradientBoostingRegressor(n_estimators=30, loss="quantile")
            base_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt)

        elif base_estimator == "DUMMY":
            return None

        if ("n_jobs" in kwargs.keys()) and not hasattr(base_estimator, "n_jobs"):
            del kwargs["n_jobs"]

        base_estimator.set_params(**kwargs)
        return base_estimator
