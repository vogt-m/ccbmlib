#   Copyright 2020 Martin Vogt, Antonio de la Vega de Leon
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial
#  portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
#  WITH  THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from collections import Counter
from math import sqrt, exp, pi
from scipy.stats import norm
import numpy as np
from scipy.sparse import csr_matrix
from itertools import combinations, islice, groupby
from operator import itemgetter
import pickle
import types
import logging
from functools import lru_cache

_use_numpy = True

logger = logging.getLogger(__name__)


def binary_to_list(binary_vector):
    return [pos for pos, bit in enumerate(binary_vector) if bit]


def sort_uniq(sequence):
    return map(
        itemgetter(0),
        groupby(sorted(sequence)))


class MarginalStats:
    """
    Class for the individual feature statistics.
    """
    __slots__ = ("sample_count",
                 "feature_to_index",
                 "frequencies",
                 "avg_no_of_features",
                 "no_of_features_variance",
                 "var_sum")

    def __init__(self, sc, fti, af, anof, fcv):
        """
        :param sc: Number of samples
        :param fti: map from fingerprint features to index of frequency list
        :param af: list of feature frequencies in decreasing order
        :param anof: average number of features per fingerprint
        :param fcv: variance of number of features
        """
        self.sample_count = sc
        self.feature_to_index = fti
        self.frequencies = af
        self.avg_no_of_features = anof
        self.no_of_features_variance = fcv
        self.var_sum = sum(p * (1 - p) for p in self.frequencies)

    def var(self, i):
        """
        :param i:
        :return: variance
        """
        return self.frequencies[i] * (1 - self.frequencies[i])

    def frequency(self, feature):
        """

        :param feature: fingerprint feature
        :return: frequency of feature
        """
        return self.frequencies[self.feature_to_index.get(feature, 0)]

    def average_intersection_union_size(self, fp=None):
        if fp:
            avg_i = sum(self.frequency(f) for f in fp)
            avg_u = len(fp) + self.avg_no_of_features - avg_i
        else:
            avg_i = sum(p * p for p in self.frequencies)
            avg_u = 2 * self.avg_no_of_features - avg_i
        return avg_i, avg_u

    def get_on_off_bits(self, fp, limit=0):
        if limit == 0:
            on_bits = set(map(lambda x: self.feature_to_index.get(x, -1), fp))
            on_bits.discard(-1)
            off_bits = set(range(len(self.frequencies))).difference(on_bits)
            return on_bits, off_bits
        else:
            on_bits = set(filter(lambda x: x < limit, map(lambda x: self.feature_to_index.get(x, limit), fp)))
            off_bits = set(range(min(len(self.frequencies), limit))).difference(on_bits)
            return on_bits, off_bits

    def pickle(self, file_obj):
        pickle.dump((self.sample_count, self.feature_to_index, self.frequencies, self.avg_no_of_features,
                     self.no_of_features_variance),
                    file_obj, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle(file_obj):
        args = pickle.load(file_obj)

        return MarginalStats(*args)

    @staticmethod
    def from_fingerprints(fingerprints, binary=False) -> "MarginalStats":
        """
        Determine marginal feature statistics from a collection of fingerprints.
        :param fingerprints:
        :param binary:
        :return:
        """
        sample_count = 0
        frequency_count = Counter()
        feature_count = 0
        feature_count_squared = 0
        for fp in fingerprints:
            sample_count += 1
            if binary:
                fp = binary_to_list(fp)
            frequency_count.update(fp)
            n = len(fp)
            feature_count += n
            feature_count_squared += n * n
        sorted_frequencies = sorted(frequency_count.items(), key=itemgetter(1), reverse=True)
        key_to_index = {key: idx for idx, (key, value) in enumerate(sorted_frequencies)}
        average_frequency = np.array(list(map(itemgetter(1), sorted_frequencies))) / sample_count
        feature_count_average = feature_count / sample_count
        feature_count_variance = feature_count_squared / sample_count - feature_count_average * feature_count_average
        return MarginalStats(sample_count, key_to_index, average_frequency, feature_count_average,
                             feature_count_variance)


class PairwiseStats:
    """
    Class for pairwise and marginal fingerprint feature statistics
    """
    __slots__ = ("marginal",
                 "cov_array",
                 "rare_frequency",
                 "limit",
                 "cov_dim",
                 "cov_sum")

    def __init__(self, m, ca, rare_frequency, limit):
        """
        :param m: MarginalStats
        :param ca: covariance array
        :param rare_frequency: frequency of rare features
        :param limit: feature index limit for covariance information
        """
        self.marginal: MarginalStats = m
        self.cov_array = ca
        self.rare_frequency = rare_frequency
        self.limit = limit
        self.cov_dim = int((1 + sqrt(1 + 8 * len(ca))) // 2)
        self.cov_sum = sum(self.cov_array[:limit * (limit - 1) // 2])

    def __enter__(self):
        logger.debug("{}.__enter__()".format(PairwiseStats.__name__))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug("{}.__exit__()".format(PairwiseStats.__name__))
        PairwiseStats._get_cached_tc_distribution.cache_clear()

    def cov(self, i, j):
        if i >= self.limit or j >= self.limit:
            if i >= self.limit and j >= self.limit:
                return self.marginal.var(i)
            else:
                return self.cov_array[self.limit * (self.limit - 1) // 2 + min(i, j)]
        if i > j:
            return self.cov_array[i * (i - 1) // 2 + j]
        elif j > i:
            return self.cov_array[j * (j - 1) // 2 + i]
        else:
            return self.marginal.var(i)

    def get_cached_tc_distribution(self, fp=None):
        key = frozenset(fp) if fp else None
        return self._get_cached_tc_distribution(key)

    @lru_cache(1024)
    def _get_cached_tc_distribution(self, fp):
        return self.get_tc_distribution(fp)

    def get_tc_distribution(self, fp=None):
        """
        Determine the unconditional or conditional correlated Bernoulli model
        :param fp: if given, a conditional distribution is returned
        :return: the distribution model
        """
        # average intersection & union cardinality
        avg_i, avg_u = self.marginal.average_intersection_union_size(fp)
        if fp is None:
            var_i = 0.0
            cijpi = 0.0
            marginal_freq = self.marginal.frequencies[:self.limit]
            cov_limit = min(len(self.cov_array), self.limit * (self.limit - 1) // 2)
            for i, pi in enumerate(marginal_freq):
                for j, pj in enumerate(marginal_freq):
                    cij = self.cov(i, j)
                    var_i += cij * (cij + 2 * pi * pj)
                    cijpi += cij * pi
            if self.rare_frequency > 0 and False:
                for i, pi in enumerate(marginal_freq):
                    cij = self.cov(i, self.limit)
                    var_i += cij * (cij + 2 * pi * self.rare_frequency)
                    cijpi += cij * pi
                cij = self.rare_frequency * (1 - self.rare_frequency)
                var_i += cij * (cij + 2 * self.rare_frequency * self.rare_frequency)
                cijpi += cij * pi
            # covariance of intersection and union cardinalities
            cov_iu = 2 * cijpi - var_i
            # variance of union cardinality
            var_u = var_i - 4 * cijpi + 2 * (
                    2 * sum(self.cov_array[:cov_limit]) + sum(p * (1 - p) for p in self.marginal.frequencies))
        else:
            on_bits, off_bits = self.marginal.get_on_off_bits(fp, self.limit)
            cov_iu = sum(self.cov(i, j) for i in on_bits for j in off_bits)
            var_on = sum(map(self.marginal.var, on_bits))
            cov_on = sum(self.cov(i, j) for i, j in combinations(on_bits, 2))
            cov_off = self.cov_sum - cov_iu - cov_on
            var_i = var_on + 2 * cov_on
            var_u = (self.marginal.var_sum - var_on) + 2 * cov_off
        return CorrelatedNormalDistributions(avg_i, var_i, avg_u, var_u, cov_iu)

    def pickle(self, file_obj):
        self.marginal.pickle(file_obj)
        pickle.dump((self.cov_array, self.rare_frequency, self.limit), file_obj, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickle(file_obj):
        marginal = MarginalStats.unpickle(file_obj)
        args = pickle.load(file_obj)
        return PairwiseStats(marginal, *args)

    @staticmethod
    def from_fingerprints(get_fp_suppl, marginal=None, limit=2048, binary=False) -> "PairwiseStats":
        """
        Obtain pairwise (and marginal statistics) from a collection of fingerprints
        :param get_fp_suppl:
        :param marginal:
        :param limit:
        :param binary:
        :return:
        """
        if marginal is None:
            fps = get_fp_suppl() if isinstance(get_fp_suppl, types.FunctionType) else get_fp_suppl
            marginal = MarginalStats.from_fingerprints(fps, binary)
        n = min(len(marginal.feature_to_index), limit)
        if n < len(marginal.feature_to_index):
            N = n + 1
        else:
            N = n
        cov = np.zeros((N * (N - 1) // 2,))
        cov_full = np.zeros((N, N)) if _use_numpy else None
        rare_count = 0
        fps = get_fp_suppl() if isinstance(get_fp_suppl, types.FunctionType) else get_fp_suppl
        for (ct, fp) in enumerate(fps):
            if ct % 10000 == 0:
                logger.debug("Fingerprints processed: {}".format(ct))
            if binary:
                fp = binary_to_list(fp)
            rare_feature = False
            if _use_numpy:
                indices = list(sort_uniq(map(lambda x: min(x, limit), map(marginal.feature_to_index.get, fp))))
                rare_feature = indices[-1] == limit if indices else False
                d = len(indices)
                data = np.ones((d,))
                row_idx = np.zeros((d,))
                a = csr_matrix((data, (row_idx, indices)), (1, N))
                b = a.transpose()
                cov_full += b.dot(a)
                if rare_feature:
                    rare_count += 1
            else:
                fp_indices = sorted(map(marginal.feature_to_index.get, fp))
                for pos, i in enumerate(fp_indices):
                    if i >= n:
                        rare_feature = True
                    else:
                        for j in islice(fp_indices, pos):
                            if j < n:
                                cov[i * (i - 1) // 2 + j] += 1
                if rare_feature:
                    rare_count += 1
                    for i in fp_indices:
                        if i < n:
                            cov[n * (n - 1) // 2 + i] += 1
        if _use_numpy:
            for i in range(1, n):
                row = i * (i - 1) // 2
                for j in range(i):
                    cov[row + j] = cov_full[i, j] / marginal.sample_count - marginal.frequencies[i] * \
                                   marginal.frequencies[j]
            if N > n:
                rare_avg = rare_count / marginal.sample_count
                row = n * (n - 1) // 2
                for j in range(n):
                    cov[row + j] = cov_full[n, j] / marginal.sample_count - marginal.frequencies[j] * rare_avg
            else:
                rare_avg = 0.0
        else:
            for i in range(1, n):
                row = i * (i - 1) // 2
                for j in range(i):
                    idx = row + j
                    cov[idx] = cov[idx] / marginal.sample_count - marginal.frequencies[i] * marginal.frequencies[j]
            if N > n:
                rare_avg = rare_count / marginal.sample_count
                row = n * (n - 1) // 2
                for j in range(n):
                    idx = row + j
                    cov[idx] = cov[idx] / marginal.sample_count - marginal.frequencies[j] * rare_avg
            else:
                rare_avg = 0.0
        return PairwiseStats(marginal, cov, rare_avg, limit)


class CorrelatedNormalDistributions:
    """
    Distribution model for the ratio of two correlated normal distributions.
    """

    def __init__(self, muX, varX, muY, varY, cov):
        """
        :param muX: mean of the numerator normal distribution
        :param varX: variance of the numerator normal distribution
        :param muY: mean of the denominator normal distribution
        :param varY: variance of the denominator normal distribution
        :param cov: covariance between the two distributions
        """
        self.muX = muX
        self.varX = varX
        self.muY = muY
        self.varY = varY
        try:
            self.sigX = sqrt(varX)
            self.sigY = sqrt(varY)
        except ValueError:
            print("muX", muX)
            print("varX", varX)
            print("muY", muY)
            print("varY", varY)
            print("cov", cov)
            self.sigX = 0
            self.sigY = 0
        self.rho = cov / (self.sigX * self.sigY) if self.sigX > 0 and self.sigY > 0 else 0.0

    @staticmethod
    def almostZero(x):
        return abs(x) < 1e-6

    def pdf(p, t):
        """
        :param t:
        :return: probability distribution function at t
        """
        ohr = 1 - p.rho * p.rho
        Rnum = (p.varY * p.muX - p.rho * p.sigX * p.sigY * p.muY) * t - p.rho * p.sigX * p.sigY * p.muX + p.varX * p.muY
        Rdenom = p.sigX * p.sigY * sqrt(
            ohr * (p.varY * t * t - 2 * p.rho * p.sigX * p.sigY * t + p.varX))
        R = Rnum / Rdenom
        supR2 = (p.varY * p.muX * p.muX - 2 * p.rho * p.sigX * p.sigY * p.muX * p.muY + p.varX * p.muY * p.muY) / (
                ohr * p.varX * p.varY)
        supR2mR2denom = p.varY * t * t - 2 * p.rho * p.sigX * p.sigY * t + p.varX
        v = p.muX - p.muY * t
        supR2mR2 = v * v / supR2mR2denom
        c = p.sigX * p.sigY * sqrt(ohr) / (pi * supR2mR2denom)
        phiR = norm.cdf(R) - .5
        return c * (exp(-0.5 * supR2) + sqrt(2 * pi) * R * phiR * exp(-0.5 * (supR2mR2)))

    def cdf_arg(p, t):
        if p.almostZero(p.varX) or p.almostZero(p.varY):
            # This usually means muX==0 so the pdf is concentrated at 0
            return np.inf if p.muX == 0 and t > 0 or t > p.muX / p.muY else -np.inf
        else:
            a = sqrt(t * t / p.varX + 1 / p.varY - 2 * p.rho * t / (p.sigX * p.sigY))
            arg = (p.muY * t - p.muX) / (p.sigX * p.sigY * a)
            return arg

    def cdf(p, t):
        """
        :param t:
        :return: cumulative distribution function at t
        """
        return norm.cdf(p.cdf_arg(t))

    def icdf(p, s):
        tl = -5
        th = 5
        sl = p.cdf(tl)
        sh = p.cdf(th)
        if s <= sl:
            return tl
        if s >= sh:
            return th
        while th - tl > 1e-6:
            tm = (th+tl)/2
            sm = p.cdf(tm)
            if sm < s:
                tl = tm
                sl = sm
            elif sm > s:
                th = tm
                sh = sm
            else:
                return tm
        alpha = (s-sl)/(sh-sl)
        return (1-alpha)*tl + alpha*th
