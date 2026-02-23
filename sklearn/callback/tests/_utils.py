# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.callback import (
    CallbackContext,
    CallbackSupportMixin,
    with_callback_context,
)
from sklearn.utils.parallel import Parallel, delayed


class TestingCallback:
    """A minimal callback used for smoke testing purposes.

    This callback doesn't define `max_estimator_depth` and is therefore not an
    `AutoPropagatedCallback`: it should not be propagated to sub-estimators.
    """

    def on_fit_begin(self, estimator):
        pass

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass

    def on_fit_end(self, estimator, context):
        pass


class TestingAutoPropagatedCallback(TestingCallback):
    """A minimal auto-propagated callback used for smoke testing purposes."""

    max_estimator_depth = None


class NotValidCallback:
    """Invalid callback since it's missing a method from the protocol."""

    def on_fit_begin(self, estimator):
        pass  # pragma: no cover

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass  # pragma: no cover


class MaxIterEstimator(CallbackSupportMixin, BaseEstimator):
    """A class that mimics the behavior of an estimator.

    The iterative part uses a loop with a max number of iterations known in advance.
    """

    _parameter_constraints: dict = {}

    def __init__(self, max_iter=20, computation_intensity=0.001):
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context(max_subtasks=self.max_iter)
        callback_ctx.eval_on_fit_begin(estimator=self)

        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={"X_train": X, "y_train": y},
            ):
                break

        self.n_iter_ = i + 1

        return self


class WhileEstimator(CallbackSupportMixin, BaseEstimator):
    """A class that mimics the behavior of an estimator.

    The iterative part uses a while loop with a number of iterations unknown in
    advance.
    """

    _parameter_constraints: dict = {}

    def __init__(self, computation_intensity=0.001):
        self.computation_intensity = computation_intensity

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context(max_subtasks=None)
        callback_ctx.eval_on_fit_begin(estimator=self)

        i = 0
        while True:
            subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={"X_train": X, "y_train": y},
            ):
                break

            if i == 20:
                break

            i += 1

        return self


class ThirdPartyEstimator(CallbackSupportMixin, BaseEstimator):
    """A class that mimics a third-party estimator with callback support only using
    public API.
    """

    def __init__(self, max_iter=20, computation_intensity=0.001):
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    @with_callback_context
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context(max_subtasks=self.max_iter)
        callback_ctx.eval_on_fit_begin(estimator=self)

        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={"X_train": X, "y_train": y},
            ):
                break

        self.n_iter_ = i + 1

        return self


class ParentFitEstimator(MaxIterEstimator):
    """A class that mimics an estimator using its parent fit method."""

    _parameter_constraints: dict = {}

    def __init__(self, max_iter=20, computation_intensity=0.001):
        super().__init__(max_iter, computation_intensity)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        return super().fit(X, y)


class NoCallbackEstimator(BaseEstimator):
    """A class that mimics an estimator without callback support."""

    def __init__(self, max_iter=20, computation_intensity=0.001):
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    def fit(self, X=None, y=None):
        for i in range(self.max_iter):
            time.sleep(self.computation_intensity)  # Computation intensive task

        return self


class MetaEstimator(CallbackSupportMixin, BaseEstimator):
    """A class that mimics the behavior of a meta-estimator.

    It has two levels of iterations. The outer level uses parallelism and the inner
    level is done in a function that is not a method of the class. That function must
    therefore receive the estimator and the callback context as arguments.
    """

    _parameter_constraints: dict = {}

    def __init__(
        self, estimator, n_outer=4, n_inner=3, n_jobs=None, prefer="processes"
    ):
        self.estimator = estimator
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_jobs = n_jobs
        self.prefer = prefer

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context(max_subtasks=self.n_outer)
        callback_ctx.eval_on_fit_begin(estimator=self)

        Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(_meta_est_func)(
                self,
                self.estimator,
                X,
                y,
                outer_callback_ctx=callback_ctx.subcontext(
                    task_name="outer", task_id=i, max_subtasks=self.n_inner
                ),
            )
            for i in range(self.n_outer)
        )

        return self


def _meta_est_func(meta_estimator, inner_estimator, X, y, *, outer_callback_ctx):
    for i in range(meta_estimator.n_inner):
        est = clone(inner_estimator)

        inner_ctx = outer_callback_ctx.subcontext(
            task_name="inner", task_id=i
        ).propagate_callbacks(sub_estimator=est)

        est.fit(X, y)

        inner_ctx.eval_on_fit_task_end(
            estimator=meta_estimator,
            data={"X_train": X, "y_train": y},
        )

    outer_callback_ctx.eval_on_fit_task_end(
        estimator=meta_estimator,
        data={"X_train": X, "y_train": y},
    )


class NoSubtaskEstimator(CallbackSupportMixin, BaseEstimator):
    """A class mimicking an estimator without subtasks in fit."""

    @with_callback_context
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context().eval_on_fit_begin(estimator=self)

        # No task performed

        return self


def func_with_callbacks(
    estimator, X=None, y=None, n_iter=4, n_jobs=None, prefer="processes", callbacks=None
):
    """A function mimicking a meta-estimator-like function.

    This function fits sub-estimators in a parallel loop, thus behaving like a
    meta-estimator. Such functions (e.g. cross_validate) should be able to handle
    callbacks.
    """
    callback_ctx = CallbackContext._from_function(
        func_with_callbacks,
        task_name="my_func",
        task_id=0,
        max_subtasks=n_iter,
        callbacks=callbacks,
    )
    callback_ctx.eval_on_fit_begin(estimator=func_with_callbacks)

    Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_func_prl_loop)(
            estimator,
            X,
            y,
            callback_ctx.subcontext(
                task_name=f"func_loop_step_{i}", task_id=i, max_subtasks=0
            ),
            calling_func=func_with_callbacks,
        )
        for i in range(n_iter)
    )

    callback_ctx.eval_on_fit_end(func_with_callbacks)


def _func_prl_loop(estimator, X, y, callback_ctx, calling_func):
    cloned_est = clone(estimator)
    callback_ctx.propagate_callbacks(cloned_est)
    cloned_est.fit(X, y)

    callback_ctx.eval_on_fit_task_end(
        estimator=calling_func,
        data={"X_train": X, "y_train": y},
    )


class MetaEstimatorUsingFunc(CallbackSupportMixin, BaseEstimator):
    """A meta-estimator using a callback compatible function."""

    _parameter_constraints: dict = {}

    def __init__(
        self, func, func_kwargs, estimator, n_iter=2, n_jobs=None, prefer="processes"
    ):
        self.func = func
        self.func_kwargs = func_kwargs
        self.estimator = estimator
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.prefer = prefer

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        callback_ctx = self._init_callback_context(max_subtasks=self.n_iter)
        callback_ctx.eval_on_fit_begin(estimator=self)

        Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(_meta_est_using_func_func)(
                self,
                self.estimator,
                self.func,
                self.func_kwargs,
                X,
                y,
                callback_ctx=callback_ctx.subcontext(
                    task_name="outer", task_id=i, max_subtasks=0
                ),
            )
            for i in range(self.n_iter)
        )

        return self


def _meta_est_using_func_func(
    meta_estimator, inner_estimator, func, func_kwargs, X, y, callback_ctx
):
    callback_ctx.propagate_callbacks(func)
    func_kwargs["callbacks"] = func_kwargs.get("callbacs", []) + callback_ctx._callbacks
    func(inner_estimator, X, y, **func_kwargs)
    callback_ctx.eval_on_fit_task_end(
        estimator=meta_estimator,
        data={"X_train": X, "y_train": y},
    )
