import os, re

def combine_results(src_dirs, dest_dir, turb, turb_class, af):

    sm_var_dir = 'sm_var_dir_' + turb + '_' + turb_class + '_' + af

    # make new destination directory
    if not os.path.isdir('Opt_Files/' + dest_dir):
        os.mkdir('Opt_Files/' + dest_dir)
    if not os.path.isdir('Opt_Files/' + dest_dir + '/' + sm_var_dir):
        os.mkdir('Opt_Files/' + dest_dir + '/' + sm_var_dir)

    # files
    ft = ['def', 'DEM', 'load', 'var']

    for k in range(len(ft)):
        for i in range(len(src_dirs)):

            f_src = open('Opt_Files/' + src_dirs[i] + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "r")
            var_lines = f_src.readlines()
            f_src.close()

            if i == 0:
                f_dest = open('Opt_Files/' + dest_dir + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "w+")
                f_dest.write(var_lines[0])
            else:
                f_dest = open('Opt_Files/' + dest_dir + '/' + sm_var_dir + '/sm_master_' + ft[k] + '.txt', "a")

            for j in range(1, len(var_lines)):
                f_dest.write(var_lines[j])

            f_dest.close()

if __name__ == "__main__":

    opt_file_srcs = ['test_3MW_1', 'test_3MW_2']

    opt_file_dest = 'test_3MW'

    turbulence = 'B'
    turbine_class = 'I'
    airfoils = 'af1'

    combine_results(opt_file_srcs, opt_file_dest, turbulence, turbine_class, airfoils)

