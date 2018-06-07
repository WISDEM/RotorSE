# This file removes directories that were created when training points were used in surrogate model creation, but
# aren't needed when the surrogate model is used. Necessary because it significantly cuts down the amount
# of file space is pushed/pulled/saved.

import os, shutil

sm_name = ['test_1000', 'test_2000']
num_pts = [1000, 2000]

for j in range(len(sm_name)):
	for i in range(num_pts[j]):
		dir_name = 'Opt_Files/' + sm_name[j] + '/sm_' + str(i)
		try:
			shutil.rmtree(dir_name)
		except:
			pass
