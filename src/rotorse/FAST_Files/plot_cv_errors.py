import numpy as np
import matplotlib.pyplot as plt

file_dir_name = "sc_test_1_var"
# file_dir_name = "sc_test_all_var"

file_dir = "Opt_Files/" + file_dir_name

num_pts = 100

file_template = "error_"
fit_types = ["second_order_poly", "least_squares", "kriging", "KPLS", "KPLSK"]
# fit_types = ["second_order_poly", "least_squares", "KPLS", "KPLSK"]

data = np.zeros([len(fit_types), 2])

for i in range(len(fit_types)):

    file_name = file_template + fit_types[i] + "_" + str(num_pts) + ".txt"

    f = open(file_dir + "/" + file_name)
    lines = f.readlines()

    for j in range(len(lines)):
        data[i,j] = float(lines[j])

# print(data)

plt.figure()

for i in range(len(fit_types)):
    plt.plot(i, ((data[i,0]**2.0+data[i,1]**2.0)/2.0)**0.5*100.0, 'o', label=fit_types[i])

plt.ylabel('RMS (%)')
plt.title('Overall RMS for test: ' + file_dir_name)

plt.legend()
# plt.savefig('/Users/bingersoll/Desktop' + file_dir_name + '.png')
plt.savefig(file_dir_name + '.png')
plt.show()
plt.close()