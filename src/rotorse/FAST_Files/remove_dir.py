# This file removes directories that were created when training points were used in surrogate model creation, but
# aren't needed when the surrogate model is used. Necessary because it significantly cuts down the amount
# of file space is pushed/pulled/saved.

import os, shutil

sm_name = 'test_ranges'
num_pts = 100

for i in range(num_pts):

    dir_name = 'Opt_Files/' + sm_name + '/sm_' + str(i)
    shutil.rmtree(dir_name)