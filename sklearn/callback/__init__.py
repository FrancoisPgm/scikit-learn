"""
The :mod:`sklearn.callback` module implements the framework and off the shelf
callbacks for scikit-learn estimators.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.callback._base import AutoPropagatedCallback, Callback, check_callbacks
from sklearn.callback._callback_context import CallbackContext, with_callback_context
from sklearn.callback._callback_support import CallbackSupportMixin
from sklearn.callback._progressbar import ProgressBar

__all__ = [
    "AutoPropagatedCallback",
    "Callback",
    "CallbackContext",
    "CallbackSupportMixin",
    "ProgressBar",
    "check_callbacks",
    "with_callback_context",
]
