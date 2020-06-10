# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/6/10 14:41
# @Last Modified by: wqshen


import numpy as np


class PointStatBase(object):
    """
    Point data statistical metric base class
    """
    def __init__(self, forecast, obs, group=None):
        if forecast is not None and obs is not None:
            self.check_consistent_length(forecast, obs, group)
            _data = np.column_stack([forecast, obs])
            _data = _data[~np.any(np.isnan(_data), axis=1)]
            self._data = _data
        self._group = None if group is None else np.asarray(group)
        self._round_score = 3

    @property
    def round(self):
        """get float decimal round value used in list_score"""
        return self._round_score

    @round.setter
    def round(self, value):
        """set float decimal round in list_score"""
        self._round_score = value

    @property
    def _f(self):
        """forecast"""
        return self._data[:, 0]

    @property
    def _o(self):
        """obs"""
        return self._data[:, 1]

    @staticmethod
    def check_consistent_length(*arrays):
        """Check that all arrays have consistent first dimensions.

        **This method COPY FROM sklearn.metrics**

        Checks whether all objects in arrays have the same shape or length.

        Parameters
        ----------
        *arrays : list or tuple of input objects.
            Objects that will be checked for consistent length.
        """

        lengths = [len(X) for X in arrays if X is not None]
        uniques = np.unique(lengths)
        if len(uniques) > 1:
            raise ValueError("Found input variables with inconsistent numbers of"
                             " samples: %r" % [int(l) for l in lengths])

    @property
    def N(self):
        """Sample size"""
        if hasattr(self, "_error"):
            return len(self._error)
        elif hasattr(self, "_data"):
            return len(self._data[:, 0])
        else:
            raise Exception("sample size can't be calculate.")

    @property
    def error(self):
        """Forecast Error Array"""
        if hasattr(self, "_error"):
            return self._error
        elif hasattr(self, "_data"):
            return self._data[:, 0] - self._data[:, 1]
        else:
            raise Exception("fcsterr not given and can't be calculate.")