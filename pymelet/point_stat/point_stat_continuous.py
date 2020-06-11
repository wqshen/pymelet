# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/6/10 14:43
# @Last Modified by: wqshen

import numpy as np
from logzero import logger
from .point_stat_base import PointStatBase


class ContinuousVariableVerification(PointStatBase):

    def __init__(self, forecast=None, obs=None, fcsterr=None, group=None):
        if (forecast is None or obs is None) and fcsterr is None:
            raise Exception("Initialize failed, check forecast and obs and fcsterr values.")
        elif forecast is not None and obs is not None and fcsterr is not None:
            logger.warning("You give forecast, obs and fcsterr, but the fcsterr will be ignored.")
            fcsterr = None
        self._available_score = ['N', 'ME', 'ME2', 'MSE', 'RMSE', 'ESTDEV', 'BCMSE', 'MAE', 'IQR', 'MAD', 'EPCT']
        if fcsterr is not None:
            self._error = fcsterr[~np.isnan(fcsterr)]
            if forecast is None:
                forecast = fcsterr + obs
            if obs is None:
                obs = forecast - fcsterr
        # Not Available, 'BAGSS', 'ANOM_CORR'
        self._available_score += ['FBAR', 'OBAR', 'FSTDEV', 'OSTDEV', 'PR_CORR', 'SP_CORR', 'KT_CORR', 'MBIAS', ]

        super(ContinuousVariableVerification, self).__init__(forecast, obs, group)

    @property
    def FBAR(self):
        """**The sample mean forecast, FBAR**"""
        return self.mean_forecast(self._f)

    @staticmethod
    def mean_forecast(forecast):
        r"""**The sample mean forecast, FBAR**

        the sample mean forecast (FBAR) is defined as,

        .. math::
            \bar{f} = \frac{1}{n}\sum_{i=1}^{n}f_i

        Returns
        ------
        numpy.ndarray, the sample mean forecast (FBAR)
        """
        return np.average(forecast)

    @property
    def OBAR(self):
        """**The sample mean observation, OBAR**"""
        return self.mean_observation(self._o)

    @staticmethod
    def mean_observation(obs):
        r"""**The sample mean observation, OBAR**

        the sample mean observation (OBAR) is defined as,

        .. math::
            \bar{o} = \frac{1}{n}\sum_{i=1}^{n}o_i

        Returns
        -------
        numpy.ndarray, the sample mean observation (OBAR)
        """
        return np.average(obs)

    @property
    def FSTDEV(self):
        """**The forecast standard deviation (FSTDEV)**"""
        return self.forecast_standard_deviation(self._f)

    @staticmethod
    def forecast_standard_deviation(forecast):
        r"""**The forecast standard deviation (FSTDEV)**

        The sample variance of the forecasts is defined as

        .. math::
            s^{2}_{f} = \frac{1}{T-1}\sum_{i=1}^{T}(f_i - \bar{f})^2

        The forecast standard deviation, FSTDEV, is defined as

        .. math::
            s_{f} = \sqrt{s^{2}_{f}}

        Returns
        -------
        numpy.ndarray, the forecast standard deviation (FSTDEV)
        """
        return np.std(forecast)

    @property
    def OSTDEV(self):
        r"""**The observed standard deviation (OSTDEV)**"""
        return self.observation_standard_deviation(self._o)

    @staticmethod
    def observation_standard_deviation(obs):
        r"""**The observed standard deviation (OSTDEV)**

        The sample variance of the observations is defined as

        .. math::
            s^{2}_{o} = \frac{1}{T-1}\sum_{i=1}^{T}(o_i - \bar{o})^2

        The observed standard deviation, OSTDEV, is defined as

        .. math::
            s_{o} = \sqrt{s^{2}_{o}}

        Returns
        -------
        numpy.ndarray, the observed standard deviation (OSTDEV)
        """
        return np.std(obs)

    @property
    def PR_CORR(self):
        r"""**The Pearson correlation coefficient ( :math:`r` , PR_CORR)**"""
        return self.pearson_correlation_coefficient(self._f, self._o)

    @staticmethod
    def pearson_correlation_coefficient(forecast, obs):
        r"""**The Pearson correlation coefficient ( :math:`r` , PR_CORR)**

        The Pearson correlation coefficient, **r**,
        measures the strength of linear association between the forecasts and observations.
        The Pearson correlation coefficient is defined as:

        .. math::
            r = \frac{\sum^{T}_{i=1}(f_i - \bar{f})(o_i - \bar{o})}{\sqrt{\sum{(f_i - \bar{f})^2}}\sqrt{\sum{(o_i - \bar{o})^2}}}

        r can range between -1 and 1;
        a value of 1 indicates perfect correlation and
        a value of -1 indicates perfect negative correlation.
        A value of 0 indicates that the forecasts and observations are not correlated.

        Returns
        -------
        numpy.ndarray, the Pearson correlation coefficient (PR_CORR)
        """
        return np.corrcoef(forecast, obs)[1, 0]

    @property
    def SP_CORR(self):
        r"""**The Spearman rank correlation coefficient ( :math:`\rho_s` , SP_CORR)**"""
        return self.spearman_rank_correlation_cofficient(self._f, self._o)

    @staticmethod
    def spearman_rank_correlation_cofficient(forecast, obs):
        r"""**The Spearman rank correlation coefficient ( :math:`\rho_s` , SP_CORR)**

        The Spearman rank correlation cofficient ( :math:`\rho_s` ) is a robust measure of association
        that is based on the ranks of the forecast and observed values rather than the actual values.
        That is, the forecast and observed samples are ordered from smallest to largest
        and rank values (from 1 to **n**, where **n** is the total number of pairs) are assigned.
        The pairs of forecast-observed ranks are then used to compute a correlation cofficient,
        analogous to the Pearson correlation cofficient, **r**.

        A simpler formulation of the Spearman-rank correlation is based on differences
        between the each of the pairs of ranks (denoted as  ( :math:`d_i` ) ):

        .. math::
            \rho_s = \frac{6}{n(n^2 - 1)}\sum^{n}_{i=1}d^{2}_{i}

        Like **r**, the Spearman rank correlation coecient ranges between -1 and 1;
        a value of 1 indicates perfect correlation and
        a value of -1 indicates perfect negative correlation.
        A value of 0 indicates that the forecasts and observations are not correlated.

        Returns
        -------
        numpy.ndarray, the Spearman correlation coefficient (SP_CORR)
        """
        from scipy.stats import spearmanr
        return spearmanr(forecast, obs)

    @property
    def KT_CORR(self):
        r"""**Kendall's Tau statistic ( :math:`\tau` , KT_CORR)**"""
        return self.kendall_tau_statistic(self._f, self._o)

    @staticmethod
    def kendall_tau_statistic(forecast, obs):
        r"""**Kendall's Tau statistic ( :math:`\tau` , KT_CORR)**

        Kendall's Tau statistic ( :math:`\tau` ) is a robust measure of the level of association
        between the forecast and observation pairs. It is defined as

        .. math::
            \tau = \frac{N_c - N_p}{n(n-1)/2}

        where NC is the number of "concordant" pairs and ND is the number of "discordant" pairs.
        Concordant pairs are identied by comparing each pair with all other pairs in the sample;
        this can be done most easily by ordering all of the ( :math:`f_i, o_i` ) pairs
        according to :math:`f_i`, in which case the :math:`o_i`, values won't necessarily be in order.
        The number of concordant matches of a particular pair with other pairs is computed by
        counting the number of pairs (with larger values)
        for which the value of oi for the current pair is exceeded (that is, pairs for which
        the values of **f** and **o** are both larger than the value for the current pair).
        Once this is done, Nc is computed by summing the counts for all pairs.
        The total number of possible pairs is ; thus, the number of discordant pairs is .

        Like **r** and  :math:`\rho_s` , Kendall's Tau ( :math:`\tau` ) ranges between -1 and 1;
        a value of 1 indicates perfect association (concor-dance) and
        a value of -1 indicates perfect negative association.
        A value of 0 indicates that the forecasts and observations are not associated.

        Returns
        -------
        numpy.ndarray, Kendall's Tau statistic ( :math:`\tau` , KT_CORR)
        """
        from scipy.stats import kendalltau
        return kendalltau(forecast, obs)

    @property
    def ME(self):
        """**The Mean Error (ME)**"""
        return self.mean_error(self.error)

    @staticmethod
    def mean_error(error):
        r"""**The Mean Error (ME)**

        The Mean Error, ME, is a measure of overall bias for continuous variables;
        in particular ME = Bias. It is defined as

        .. math::
            ME = \frac{1}{n}\sum^{n}_{i=1}(f_i - o_i) = \bar{f} - \bar{o}

        A perfect forecast has ME = 0.

        Returns
        -------
        numpy.ndarray, The Mean Error (ME)
        """
        return np.average(error)

    @property
    def ME2(self):
        """**The Mean Error Squared** (ME2)"""
        return self.mean_error_squared(self.error)

    @staticmethod
    def mean_error_squared(error):
        """**The Mean Error Squared** (ME2)

        The Mean Error Squared, ME2, is provided to give a complete breakdown of MSE
        in terms of squared Bias plus estimated variance of the error,
         as detailed below in the section on BCMSE. It is defined as ME2 = ME2.

        A perfect forecast has ME2 = 0.

        Returns
        -------
        numpy.ndarray, The Mean Error (ME)
        """
        return np.square(np.average(error))

    @property
    def MBIAS(self):
        """**Multiplicative bias (MBIAS)**"""
        return self.multiplicative_bias(self._f, self._o)

    @staticmethod
    def multiplicative_bias(forecast, error):
        r"""**Multiplicative bias (MBIAS)**

        Multiplicative bias is simply the ratio of the means of the forecasts and the observations:

        .. math::
            MBIAS = \frac{\bar{f}}{\bar{o}}

        Returns
        -------
        numpy.ndarray, Multiplicative bias (MBIAS)
        """
        return np.average(forecast) / np.average(error)

    @property
    def MSE(self):
        """**Mean-squared error (MSE)**"""
        return self.mean_squared_error(self.error)

    @staticmethod
    def mean_squared_error(error):
        r"""**Mean-squared error (MSE)**

        MSE measures the average squared error of the forecasts. Specifically,

        .. math::
            MSE = \frac{1}{n}\sum{(f_i - o_i)^2}

        Returns
        -------
        numpy.ndarray, Mean-squared error (MSE)
        """
        return np.average(error ** 2)

    @property
    def RMSE(self):
        """**Root-mean-squared error (RMSE)**"""
        return self.root_mean_squared_error(self.error)

    @staticmethod
    def root_mean_squared_error(error):
        """**Root-mean-squared error (RMSE)**

        RMSE is simply the square root of the MSE, :math:`RMSE = \sqrt{MSE}`

        Returns
        -------
        numpy.ndarray, Root-mean-squared error (RMSE)
        """
        return np.sqrt(np.average(error ** 2))

    @property
    def ESTDEV(self):
        """**Standard deviation of the error** (ESTDEV)"""
        return self.standard_deviation_of_error(self.error)

    @staticmethod
    def standard_deviation_of_error(error):
        """**Standard deviation of the error** (ESTDEV)

        Returns
        -------
        numpy.ndaray, Standard deviation of the error
        """
        return np.std(error)

    @property
    def BCMSE(self):
        """**Bias-Corrected MSE (BCMSE)**"""
        return self.bias_corrected_mse(self.error)

    @staticmethod
    def bias_corrected_mse(error):
        r"""**Bias-Corrected MSE (BCMSE)**

        MSE and RMSE are strongly impacted by large errors.
        They also are strongly impacted by large bias (ME) values.
        MSE and RMSE can range from 0 to infinity.
        A perfect forecast would have MSE = RMSE = 0.

        MSE can be re-written as,

        .. math::
            MSE = (\bar{f} - \bar{o})^2 + s^{2}_{f} + s^{2}_{o} -2 s_f s_o r_{fo}

        where :math:`\bar{f} - \bar{o} = ME` and :math:`s^{2}_{f} + s^{2}_{o} -2 s_f s_o r_{fo}` is
        the estimated variance of the error, :math:`s^{2}_{fo}` . Thus, :math:`MSE = ME^2 + s^{2}_{f-o}`
        To understand the behavior of MSE, it is important to examine both of the terms of MSE,
        rather than examining MSE alone. Moreover, MSE can be strongly influenced by ME,
        as shown by this decomposition.

        The standard deviation of the error, :math:`s_{f-o}` , is

        .. math::
            s_{f-o}=\sqrt{s^{2}_{f-o}}=\sqrt{s^{2}_{f} + s^{2}_{o} -2 s_f s_o r_{fo}}

        Note that the square of the standard deviation of the error (ESTDEV2) is
        sometimes called the "Bias-corrected MSE" (BCMSE)
        because it removes the effect of overall bias from the forecast-observation squared differences.

        Returns
        -------
        numpy.ndarray, Bias-Corrected MSE (BCMSE)
        """
        return np.square(np.std(error))

    @property
    def MAE(self):
        """**Mean Absolute Error (MAE)**"""
        return self.mean_absolute_error(self.error)

    @staticmethod
    def mean_absolute_error(error):
        r"""**Mean Absolute Error (MAE)**

        The Mean Absolute Error (MAE) is defined as :math:`MAE = \frac{1}{n}\sum{|f_i - o_i|}`

        MAE is less inuenced by large errors and also does not depend on the mean error.
        A perfect forecast would have MAE = 0.

        Returns
        -------
        numpy.ndarray, Mean Absolute Error (MAE)
        """
        return np.average(np.abs(error))

    @property
    def IQR(self):
        """"**Inter Quartile Range of the Errors (IQR)**"""
        return self.inter_quartile_range_of_errors(self.error)

    @staticmethod
    def inter_quartile_range_of_errors(error):
        r"""**Inter Quartile Range of the Errors (IQR)**

        The Inter Quartile Range of the Errors (IQR) is the difference
        between the 75th and 25th percentiles of the errors. It is dened as

        .. math::
            IQR = p_{75} (f_i - o_i) - p_{25}(f_i - o_i)

        IQR is another estimate of spread, similar to standard error,
        but is less inuenced by large errors and also does not depend on the mean error.

        A perfect forecast would have IQR = 0.

        Returns
        -------
        nupmy.ndarray, Inter Quartile Range of the Errors (IQR)
        """
        return np.percentile(error, 75) - np.percentile(error, 25)

    @property
    def MAD(self):
        """Median Absolute Deviation (MAD)"""
        return self.median_absolute_deviation(self.error)

    @staticmethod
    def median_absolute_deviation(error):
        """Median Absolute Deviation (MAD)

        The Median Absolute Deviation (MAD) is defined as :math:`MAD=median|f_i - o_i|`

        MAD is an estimate of spread, similar to standard error,
        but is less inuenced by large errors and also does not depend on the mean error.

        A perfect forecast would have MAD = 0.

        Returns
        -------
        numpy.ndarray, Median Absolute Deviation (MAD)
        """
        return np.median(np.abs(error))

    @property
    def BAGSS(self):
        """Bias Adjusted Gilbert Skill Score (BAGSS)"""
        return self.bias_adjusted_gilbert_skill_score(self._f, self._o)

    @staticmethod
    def bias_adjusted_gilbert_skill_score(forecast, obs):
        """Bias Adjusted Gilbert Skill Score (BAGSS)

        The Bias Adjusted Gilbert Skill Score (BAGSS) is the Gilbert Skill Score,
        but with the contingency table counts adjusted to eliminate
        as much bias in the forecast as possible.

        For details, see `Brill and Messinger, 2009. <https://www.adv-geosci.net/16/137/2008/>`_

        Returns
        -------
        Not implemented
        numpy.ndarray, Bias Adjusted Gilbert Skill Score (BAGSS)
        """
        return

    @property
    def EPCT(self):
        """Percentiles (0.1, 0.25, 0.5, 0.75, 0.9) of the errors"""
        return self.percentile_errors(self.error)

    @staticmethod
    def percentile_errors(error):
        """Percentiles of the errors

        Percentiles of the errors provide more information about the distribution of errors
        than can be obtained from the mean and standard deviations of the errors.
        Percentiles are computed by ordering the errors from smallest to largest
        and computing the rank location of each percentile in the ordering,
        and matching the rank to the actual value.

        Percentiles can also be used to create box plots of the errors.

        The 0.10th, 0.25th, 0.50th, 0.75th, and 0.90th quantile values of the errors are computed.

        Returns
        -------
        numpy.ndarray, Percentiles of the errors
        """
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        return np.quantile(error, quantiles)

    @property
    def ANOM_CORR(self):
        """The Anomaly correlation coefficient (ANOM_CORR)"""
        return self.anomaly_correlation_coefficient(self._f, self._o, None)

    @staticmethod
    def anomaly_correlation_coefficient(forecast, obs, climate):
        r"""The Anomaly correlation coefficient (ANOM_CORR)

        The Anomaly correlation coecient is equivalent to the Pearson correlation coefficient,
        except that both the forecasts and observations are first adjusted according to a climatology value.
        The anomaly is the difference between the individual forecast or observation and the typical situation,
        as measured by a climatology (**c**) of some variety.
        It measures the strength of linear association between the forecast anomolies and observed anomalies.

        The Anomaly correlation coefficient is defined as:

        .. math::
            Anomoly Correlation = \frac{\sum{(f_i - c)(o_i - c)}} {\sqrt{\sum{(f_i - c)^2}}  \sqrt{\sum{(o_i - c)^2}}}

        Anomaly correlation can range between -1 and 1;

            - a value of 1 indicates perfect correlation and
            - a value of -1 indicates perfect negative correlation.
            - A value of 0 indicates that the forecast and observed anomalies are not correlated.

        Returns
        -------
        Not implemented
        """
        return

    def list_score(self):
        """list all available score"""
        return {k: np.round(getattr(self, k), self.round) for k in self._available_score}
