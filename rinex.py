import _rinex
import numpy as np
from scipy.interpolate import BarycentricInterpolator

Generator = _rinex.Generator
Record = _rinex.Record


class Ephemeris:
    def __init__(self, *files, order=None):
        # if mode not in [None, 'precise', 'fast', 'adapt']:
        #     raise ValueError("mode must be either 'precise' or 'fast' (None)")
        if order and int(order) % 2 != 1:
            raise ValueError('order must be an odd integer')

        # self.mode = mode or 'precise'
        self.order = order or 15
        self.eph = _rinex.Ephemeris(*files)

        intervals = np.diff(self.eph.epoches)

        if np.any(intervals != intervals[0]):
            raise ValueError('SP3 files with variable intervals are not supported yet')

        self.interval = intervals[0] / np.timedelta64(1, 's')

        steps = np.arange(-int(self.order/2), int(self.order/2)+1) * self.interval

        self.interp = BarycentricInterpolator(steps)

    def position_at(self, sat: str, time: np.datetime64):

        closest_epoch_idx = np.argmin(np.abs(self.eph.epoches - time))

        sat_idx = np.where(self.eph.sats == bytes(sat, 'ascii'))

        idx_start = max(closest_epoch_idx - int(self.order/2), 0)
        idx_end = min(closest_epoch_idx + int(self.order/2) + 1, self.eph.records.shape[0])

        yi = self.eph.records[idx_start:idx_end, sat_idx, :]

        if yi.shape[0] != self.interp.xi.shape[0]:
            raise ValueError("Ephemeris boundary is too close, add more data or switch mode")

        self.interp.set_yi(yi)

        x = (self.eph.epoches[closest_epoch_idx] - time) / np.timedelta64(1, 's')

        return self.interp(x)
