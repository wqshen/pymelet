# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/6/10 15:52
# @Last Modified by: wqshen

import numpy as np
from numpy.lib.stride_tricks import as_strided
from skimage.measure import label as sklabel, regionprops
from logzero import logger


class ObjectResolve(object):
    def __init__(self, threshold, radius):
        """Initial ObjectResolve class

        Parameters
        ----------
        threshold (float): the threshold construct mask
        radius (int):  a radius of influence of the circular filter function
        """
        self._radius = radius
        self._threshold = threshold

    @property
    def weights(self):
        """Property filter weight array"""
        H = 1 / (np.pi * np.power(self._radius, 2))
        weight_field = np.zeros((self._radius * 2 + 1, self._radius * 2 + 1))
        x = np.arange(-self._radius, self._radius + 0.1, 1)
        y = x[:, None]
        weight_field[(np.power(x, 2) + np.power(y, 2)) <= np.power(self._radius, 2)] = H
        return weight_field

    @property
    def radius(self):
        """Get filter window radius size"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """Set filter window radius size"""
        self._radius = value

    @property
    def threshold(self):
        """Get the threshold of construct mask"""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """Set the threshold of construct mask"""
        self._threshold = value

    def convolve2d(self, field, stride=1):
        """2D numpy based convolve

        Parameters
        ----------
        field (numpy.array): 2D raw field
        stride (int): stride moving convolve window

        Returns
        -------
        numpy.array, 2D weight convolved filter field
        """
        weight = self.weights
        im_h, im_w = field.shape
        f_h, f_w = weight.shape

        out_shape = (1 + (im_h - f_h) // stride, 1 + (im_w - f_w) // stride, f_h, f_w)
        out_strides = (field.strides[0] * stride, field.strides[1] * stride, field.strides[0], field.strides[1])
        windows = as_strided(field, shape=out_shape, strides=out_strides, writeable=False)
        return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))

    def mask(self, field):
        """Create threshold mask

        Parameters
        ----------
        field (numpy.array): the weight convolved filtered 2D field

        Returns
        -------
        numpy.array, 1 when value greater or equal than self._threshold else 0
        """
        return np.where(field >= self._threshold, 1, 0)

    @staticmethod
    def restore_field(field, mask):
        """Restore raw data field multiply mask

        Parameters
        ----------
        field (numpy.array): the raw data field
        mask (numpy.array): mask array, same size with field

        Returns
        -------
        numpy.array, masked applied data field
        """
        return field * mask

    def label(self, mask, connectivity=2):
        """label objects

        Parameters
        ----------
        mask (numpy.array):  mask array, same size with field

        Returns
        -------
        pass
        """
        label_mask = sklabel(mask, connectivity=connectivity)
        return label_mask

    @staticmethod
    def show_label(field, label):
        """show labeled object

        Parameters
        ----------
        field (numpy.array): restored field
        label (numpy.array): result from skimage.measure.label
        """
        from skimage import color
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        dst = color.label2rgb(label)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.contourf(field, cmap='Blues', extend='both')
        ax2.imshow(dst, interpolation='nearest', origin='lower')
        for i, region in enumerate(regionprops(label)):
            logger.debug("ID={}, C=({:5.1f},{:5.1f}), A={}".format(i, *region.centroid, region.area))
            if region.area <= 20:
                continue
            x, y = region.centroid
            ax1.text(y, x, i, verticalalignment='center', horizontalalignment='center', c='r')
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)

        fig.tight_layout()
        plt.show()

