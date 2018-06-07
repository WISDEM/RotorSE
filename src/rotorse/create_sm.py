import os
import numpy as np

# this script calls another script multiple times to calculate FAST outputs
# using certain training points, defined in the other script.

filename = "test_FAST_DEMs.py"

# === single variable surrogate model === #
# sm_range = [1, 2, 3, 4, 5]
# sm_range = [1]
# for i in sm_range:
#     os.system("python " + filename + " " + str(i) )


# === multi-variable surrogate model === #

# sm_range = [1, 2, 3]
# sm_range = [1]
# for i in sm_range:
#     for j in sm_range:
#         # for k in sm_range:
#             # print(i,j)
#             # os.system("python " + filename + " " + str(i) + " " + str(j) +  " " + str(k) )
#         os.system("python " + filename + " " + str(i) + " " + str(j))

# === lhs surrogate model === #
total_pts = 1
for i in range(0, total_pts):
    os.system("python " + filename + " " + str(i))
