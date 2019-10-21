import unittest
import logging
import datetime

log = logging.getLogger()

import rinex


class Tests(unittest.TestCase):
    def test_header_essential(self):
        rinex_file = rinex.load("test.rnx")
        self.assertEqual((-4236472.1312, 4559859.8294, -1388765.5574), rinex_file.position)
        self.assertDictEqual({
            'J': ['C1C', 'L1C', 'D1C', 'S1C', 'C2L', 'L2L', 'D2L', 'S2L', 'C5Q', 'L5Q', 'D5Q', 'S5Q'],
            'G': ['C1C', 'L1C', 'D1C', 'S1C', 'C1W', 'S1W', 'C2W', 'L2W', 'D2W', 'S2W', 'C2L', 'L2L', 'D2L', 'S2L',
                  'C5Q', 'L5Q', 'D5Q', 'S5Q'],
            'C': ['C1I', 'L1I', 'D1I', 'S1I', 'C7I', 'L7I', 'D7I', 'S7I'],
            'S': ['C1C', 'L1C', 'D1C', 'S1C', 'C5I', 'L5I', 'D5I', 'S5I'],
            'R': ['C1C', 'L1C', 'D1C', 'S1C', 'C2P', 'L2P', 'D2P', 'S2P', 'C2C', 'L2C', 'D2C', 'S2C', 'C3Q', 'L3Q',
                  'D3Q', 'S3Q'],
            'E': ['C1C', 'L1C', 'D1C', 'S1C', 'C5Q', 'L5Q', 'D5Q', 'S5Q', 'C7Q', 'L7Q', 'D7Q', 'S7Q', 'C8Q', 'L8Q',
                  'D8Q', 'S8Q']},
            rinex_file.obs_types)

    def test_bad_file(self):
        with self.assertRaises(Exception):
            rinex.load("test.crx")


    def test_missing_values(self):
        rinex_file = rinex.load("test.rnx", svs=[("G", 8)])
        first_obs = next(rinex_file)
        self.assertEqual(None, first_obs.obs("G", 8, 0))
        self.assertEqual(None, first_obs.obs("G", 8, 1))


    def test_obs_filtering(self):
        rinex_file = rinex.load("test.rnx", svs=[("G", 11)], meas=["GC1C", "GL1C"])
        first_obs = next(rinex_file)
        C1C_idx = rinex_file.obs_types['G'].index('C1C')
        L1C_idx = rinex_file.obs_types['G'].index('L1C')
        self.assertEqual((22186411.450, -1, 7), first_obs.obs("G", 11, C1C_idx))
        self.assertEqual((116590366.396, 0, 7), first_obs.obs("G", 11, L1C_idx))
        self.assertEqual(None, first_obs.obs("G", 10, 0))
        self.assertEqual(None, first_obs.obs("G", 11, 3))

    def test_getObs(self):
        rinex_file = rinex.load("test.rnx")
        C1C_idx = rinex_file.obs_types['G'].index('C1C')
        L1C_idx = rinex_file.obs_types['G'].index('L1C')
        first_obs = next(rinex_file)
        self.assertEqual((22186411.450, -1, 7), first_obs.obs("G", 11, C1C_idx))
        self.assertEqual((116590366.396, 0, 7), first_obs.obs("G", 11, L1C_idx))

    def test_times(self):
        rinex_file = rinex.load("test.rnx")
        first_obs = next(rinex_file)
        self.assertEqual(datetime.datetime(2017, 9, 1, 0, 0, 0), first_obs.time)
        last_obs = None
        for el in rinex_file:
            last_obs = el
        self.assertEqual(datetime.datetime(2017, 9, 1, 23, 59, 30), last_obs.time)
