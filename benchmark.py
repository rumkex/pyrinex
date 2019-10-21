import numpy as np
import matplotlib.pyplot as plt
import time

import rinex

# file = 'test.rnx'
#
# start = time.time()
# rinex_file = rinex.load(file, ["GL1C"], [("G", 8)])
# L1C_idx = rinex_file.obs_types["G"].index("L1C")
# arr = np.array([(record.obs("G", 8, L1C_idx) or (np.NaN, -1, -1))[0] for record in rinex_file])
# end = time.time()
#
# print(end-start, 's')
#
# plt.plot(arr)
# plt.show()

e = rinex.Ephemeris('test.sp3')
print(e.position_at('G01', np.datetime64('2018-03-11 03:00:00', 's')))
print(e.position_at('G01', np.datetime64('2018-03-11 00:00:00', 's')))
