# FAST_util.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

# ========================================================================================================= #

def setupFAST_checks(FASTinfo):

    # === check splines / results === #
    # if any are set as true, optimization will stop
    FASTinfo['check_results'] = False  # Opt. stops if set as True
    FASTinfo['check_sgp_spline'] = False  # Opt. stops if set as True
    FASTinfo['check_stif_spline'] = False  # Opt. stops if set as True
    FASTinfo['check_peaks'] = False  # Opt. stops if set as True
    FASTinfo['check_rainflow'] = False  # Opt. stops if set as True
    FASTinfo['check_rm_time'] = False  # Opt. stops if set as True

    FASTinfo['check_damage'] = False  # Opt. stops if set as True
    FASTinfo['check_nom_DEM_damage'] = False # only works when check_damage is set as True

    FASTinfo['check_fit'] = False  # Opt. stops if set as True
    FASTinfo['check_opt_DEMs'] = False # only called when opt_with_fixed_DEMs is True

    FASTinfo['do_cv_DEM'] = False  # cross validation of surrogate model for DEMs
    FASTinfo['do_cv_Load'] = False  # cross validation of surrogate model for extreme loads
    FASTinfo['do_cv_def'] = False  # cross validation of surrogate model for tip deflection

    FASTinfo['check_point_dist'] = False  # plot distribution of points (works best in 2D)
    FASTinfo['check_cv'] = False # works best in 2D

    FASTinfo['check_var_domains'] = False # plots

    FASTinfo['print_sm'] = False

    return FASTinfo

# ========================================================================================================= #

def setupFAST(rotor, FASTinfo, description):

    # === set up FAST top level options === #
    FASTinfo = setup_top_level_options(FASTinfo)

    # === set up FAST check options === #
    FASTinfo = setupFAST_checks(FASTinfo)

    # === constraint groups === #
    FASTinfo['use_fatigue_cons'] = True
    FASTinfo['use_struc_cons'] = True
    FASTinfo['use_tip_def_cons'] = True

    #=== ===#

    FASTinfo['description'] = description

    # === Platform (Local or SC) - unique to each user === #

    # path to RotorSE_FAST upper directory
    # FASTinfo['path'] = '/fslhome/ingerbry/GradPrograms/'
    FASTinfo['path'] = '/Users/bingersoll/Dropbox/GradPrograms/'

    # === dir_saved_plots === #
    # FASTinfo['dir_saved_plots'] = '/fslhome/ingerbry/GradPrograms/opt_plots'
    FASTinfo['dir_saved_plots'] = '/Users/bingersoll/Desktop'

    # === Optimization and Template Directories === #
    FASTinfo['opt_dir'] = ''.join((FASTinfo['path'], 'RotorSE_FAST/' \
        'RotorSE/src/rotorse/FAST_Files/Opt_Files/', FASTinfo['description']))

    if os.path.isdir(FASTinfo['opt_dir']):
        # placeholder
        print('optimization directory already created')
    else:
        os.mkdir(FASTinfo['opt_dir'])

    FASTinfo['template_dir'] = ''.join((FASTinfo['path'], 'RotorSE_FAST/' \
        'RotorSE/src/rotorse/FAST_Files/FAST_File_templates/'))

    # === options if previous optimizations have been performed === #

    if FASTinfo['seq_run']:
        FASTinfo['prev_description'] = 'tst_path'

        # for running multiple times
        FASTinfo['prev_opt_dir'] = ''.join((FASTinfo['path'], 'RotorSE_FAST/' \
            'RotorSE/src/rotorse/FAST_Files/Opt_Files/', FASTinfo['prev_description']))

    # === Surrogate Model Options === #

    if FASTinfo['train_sm'] or FASTinfo['Use_FAST_sm']:
        FASTinfo = create_surr_model_params(FASTinfo)

    if FASTinfo['train_sm']:

        if FASTinfo['training_point_dist'] == 'linear':
            FASTinfo, rotor = create_surr_model_linear_options(FASTinfo, rotor)
        elif FASTinfo['training_point_dist'] == 'lhs':
            FASTinfo, rotor = create_surr_model_lhs_options(FASTinfo, rotor)
        else:
            raise Exception('Training point distribution not specified correctly.')
    # === ===#

    # === Add FAST outputs === #
    FASTinfo = add_outputs(FASTinfo)

    # === FAST Run Time === #
    FASTinfo['Tmax_turb'] = 100.0 # 640.0
    FASTinfo['Tmax_nonturb'] = 100.0 # 100.0
    FASTinfo['dT'] = 0.0125

    # remove artificially noisy data
    # obviously, must be greater than Tmax_turb, Tmax_nonturb
    FASTinfo['rm_time'] = 40.0 # 40.0

    FASTinfo['turb_sf'] = 1.0

    # option for cross validation
    if FASTinfo['do_cv_DEM'] or FASTinfo['do_cv_Load'] or FASTinfo['do_cv_def']:

        FASTinfo = kfold_params(FASTinfo)

        # FASTinfo['cv_description'] = FASTinfo['description'] + '_cv'
        #
        # FASTinfo['cv_dir'] = ''.join((FASTinfo['path'], 'RotorSE_FAST/' \
        # 'RotorSE/src/rotorse/FAST_files/Opt_Files/', FASTinfo['cv_description']))

    # === strain gage placement === #
    # FASTinfo['sgp'] = [1,2,3]
    FASTinfo['sgp'] = [4]

    #for each position
    FASTinfo['NBlGages'] = []
    FASTinfo['BldGagNd'] = []
    FASTinfo['BldGagNd_config'] = []

    if 1 in FASTinfo['sgp']:
        FASTinfo['NBlGages'].append(7)  # number of strain gages (max is 7)
        FASTinfo['BldGagNd'].append([1, 2, 3, 4, 5, 6, 7])  # strain gage positions
        FASTinfo['BldGagNd_config'].append([1, 2, 3, 4, 5, 6, 7])  # strain gage positions
    if 2 in FASTinfo['sgp']:
        FASTinfo['NBlGages'].append(7) # number of strain gages (max is 7)
        FASTinfo['BldGagNd'].append([8,9,10,11,12,13,14]) # strain gage positions
        FASTinfo['BldGagNd_config'].append([8,9,10,11,12,13,14]) # strain gage positions
    if 3 in FASTinfo['sgp']:
        FASTinfo['NBlGages'].append(3) # number of strain gages (max is 7)
        FASTinfo['BldGagNd'].append([15,16,17]) # strain gage positions
        FASTinfo['BldGagNd_config'].append([15,16,17]) # strain gage positions
    if 4 in FASTinfo['sgp']:
        # over entire range
        FASTinfo['NBlGages'].append(7) # number of strain gages (max is 7)
        FASTinfo['BldGagNd'].append([1,3,5,7,9,12,17]) # strain gage positions
        FASTinfo['BldGagNd_config'].append([1,3,5,7,9,12,17]) # strain gage positions

    # FASTinfo['spec_sgp_dir'] = FASTinfo['opt_dir'] + '/' + 'sgp' + str(FASTinfo['sgp'])

    # === specify which DLCs will be included === #

    # === options if active DLC list has been created === #
    FASTinfo['use_DLC_list'] = False
    if FASTinfo['use_DLC_list']:
        FASTinfo['DLC_list_loc'] = FASTinfo['opt_dir'] + '/' + 'active_wnd.txt'

    if not FASTinfo['use_DLC_list']:

        # === for optimization === #
        # DLC_List = ['DLC_1_2','DLC_1_3', 'DLC_1_4','DLC_1_5','DLC_6_1','DLC_6_3']

        # === for testing === #

        # nominal wind file
        # DLC_List = ['DLC_0_0']

        #non turbulent DLCs
        # DLC_List = ['DLC_1_4','DLC_1_5','DLC_6_1','DLC_6_3']

        #non turbulent extreme events
        # DLC_List = ['DLC_6_1','DLC_6_3']
        # DLC_List = ['DLC_6_1']

        #turbulent DLCs
        # DLC_List = ['DLC_1_2','DLC_1_3']
        DLC_List=['DLC_1_3']

    else:
        DLC_List_File = open(FASTinfo['DLC_list_loc'], 'r')

        DLC_lines = DLC_List_File.read().split('\n')
        DLC_List = []
        for i in range(0, len(DLC_lines) - 1):
            DLC_List.append(DLC_lines[i])
    FASTinfo['DLC_List'] = DLC_List

    # === turbulent wind file parameters === #
    #  random seeds (np.linspace(1,6,6) is pre-calculated)
    FASTinfo['rand_seeds'] = np.linspace(1, 1, 1)
    # FASTinfo['rand_seeds'] = np.linspace(1, 6, 6)

    #  mean wind speeds (np.linspace(5,23,10) is pre-calculated)
    FASTinfo['mws'] = np.linspace(11, 11, 1)
    # FASTinfo['mws'] = np.linspace(5, 23, 10)

    # === create list of .wnd files === #
    # .wnd files list
    FASTinfo['wnd_list'] = []

    # wnd type list
    FASTinfo['wnd_type_list'] = []

    # list of whether turbine is parked or not
    FASTinfo['parked'] = []

    for i in range(0, len(FASTinfo['DLC_List'])+0):
        # call DLC function
        FASTinfo['wnd_list'], FASTinfo['wnd_type_list'] \
            = DLC_call(FASTinfo['DLC_List'][i], FASTinfo['wnd_list'], FASTinfo['wnd_type_list'],
                       FASTinfo['rand_seeds'], FASTinfo['mws'], len(FASTinfo['sgp']), FASTinfo['parked'])


    # fatigue options
    FASTinfo['m_value'] = 10.0

    # turbulent, nonturbulent directories
    FASTinfo['turb_wnd_dir'] = 'RotorSE_FAST/WND_Files/turb_wnd_dir/'
    FASTinfo['nonturb_wnd_dir'] = 'RotorSE_FAST/WND_Files/nonturb_wnd_dir/'

    return FASTinfo


# ========================================================================================================= #

def kfold_params(FASTinfo):

    # number of folds
    FASTinfo['num_folds'] = 10

    # check that num_pts/num_folds doesn't have a remainder (is divisible)
    if (FASTinfo['num_pts'] % FASTinfo['num_folds']) > 0:
        raise Exception('Number of folds (k) should be a factor of num_pts.')
    if FASTinfo['num_folds'] == FASTinfo['num_pts']:
        raise Exception('Number of folds (k) needs to be less than num_pts (for current setup).')
    num_pts_in_grp = FASTinfo['num_pts'] / FASTinfo['num_folds']

    # randomly divide pts into num_folds groups
    num_pts_list = np.linspace(0, FASTinfo['num_pts']-1, FASTinfo['num_pts']) # -1 so it's zero-based

    # === shuffled list === #
    shuffled_list_file = FASTinfo['opt_dir'] + '/shuffled_list.txt'

    use_preset_shuffled_list = True

    if use_preset_shuffled_list:

        if not os.path.isfile(shuffled_list_file):

            # create shuffled list
            shuffled_list = random.sample(num_pts_list, len(num_pts_list))

            f = open(shuffled_list_file, "w+")

            for i in range(0, FASTinfo['num_pts']):
                    f.write(str(shuffled_list[i]) + '\n')

            f.close()

        else:

            shuffled_list_lines = open(shuffled_list_file, "r+").readlines()
            shuffled_list = []
            for i in range(FASTinfo['num_pts']):
                shuffled_list.append(float(shuffled_list_lines[i]))

    else:
        shuffled_list = random.sample(num_pts_list, len(num_pts_list))

    FASTinfo['kfolds'] = []
    for i in range(FASTinfo['num_folds']):
        FASTinfo['kfolds'].append(shuffled_list[i*num_pts_in_grp:(i+1)*num_pts_in_grp])

    # check
    # print(num_pts_list)
    # print(random.sample(num_pts_list, len(num_pts_list)))

    return FASTinfo

# ========================================================================================================= #

def setup_top_level_options(FASTinfo):

    if FASTinfo['opt_with_FAST_in_loop']:
        FASTinfo['use_FAST'] = True
        FASTinfo['Run_Once'] = False
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = False
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = False

    elif FASTinfo['opt_without_FAST']:
        FASTinfo['use_FAST'] = False
        FASTinfo['Run_Once'] = False
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = False
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = False

    elif FASTinfo['calc_fixed_DEMs']:
        FASTinfo['use_FAST'] = True
        FASTinfo['Run_Once'] = True
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = False
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = False

    elif FASTinfo['opt_with_fixed_DEMs']:
        FASTinfo['use_FAST'] = False
        FASTinfo['Run_Once'] = False
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = True
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = False

    elif FASTinfo['opt_with_fixed_DEMs_seq']:
        FASTinfo['use_FAST'] = False
        FASTinfo['Run_Once'] = False
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = True
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = True

    elif FASTinfo['calc_surr_model']:
        FASTinfo['use_FAST'] = True
        FASTinfo['Run_Once'] = True
        FASTinfo['train_sm'] = True
        FASTinfo['Use_FAST_Fixed_DEMs'] = False
        FASTinfo['Use_FAST_sm'] = False
        FASTinfo['seq_run'] = False

    elif FASTinfo['opt_with_surr_model']:
        FASTinfo['use_FAST'] = False
        FASTinfo['Run_Once'] = False
        FASTinfo['train_sm'] = False
        FASTinfo['Use_FAST_Fixed_DEMs'] = False
        FASTinfo['Use_FAST_sm'] = True
        FASTinfo['seq_run'] = False

    else:
        raise Exception('Must choose a FAST option.')

    return FASTinfo

# ========================================================================================================= #

def setup_FAST_seq_run_des_var(rotor, FASTinfo):

    rotor_desvar0, rotor_desvar1, rotor_desvar2, rotor_desvar3, rotor_desvar4 = [], [], [], [], []

    file_name = FASTinfo['prev_opt_dir'] + '/' + 'opt_results.txt'

    fp = open(file_name)
    line = fp.readlines()

    for i in range(0, 5):
        globals()['desvar%s' % i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", line[i])

        globals()['rotor_desvar%s' % i] = np.zeros(len(globals()['desvar%s' % i]))
        for j in range(0, len(globals()['desvar%s' % i])):
            globals()['rotor_desvar%s' % i][j] = float(globals()['desvar%s' % i][j])

    rotor['r_max_chord'] = rotor_desvar0
    rotor['chord_sub'] = rotor_desvar1
    rotor['theta_sub'] = rotor_desvar2
    rotor['sparT'] = rotor_desvar3
    rotor['teT'] = rotor_desvar4

    return rotor

# ========================================================================================================= #

def create_surr_model_params(FASTinfo):

    # total number of points (lhs)
    FASTinfo['num_pts'] = 100

    # approximation model
    # implemented options - second_order_poly, least_squares, kriging, KPLS, KPLSK
    FASTinfo['approximation_model'] = 'second_order_poly'
    # FASTinfo['approximation_model'] = 'least_squares'
    # FASTinfo['approximation_model'] = 'kriging'
    # FASTinfo['approximation_model'] = 'KPLS'
    # FASTinfo['approximation_model'] = 'KPLSK'

    # initial hyperparameter value (kriging, KPLS, KPLSK only use)
    FASTinfo['theta0_val'] = [1e-2]

    # training point distribution
    FASTinfo['training_point_dist'] = 'lhs'
    # FASTinfo['training_point_dist'] = 'linear'

    if FASTinfo['training_point_dist'] == 'lhs':

        FASTinfo['sm_var_out_dir'] = 'sm_var_dir'

        FASTinfo['sm_var_file_template'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_var_'
        FASTinfo['sm_DEM_file_template'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_DEM_'
        FASTinfo['sm_load_file_template'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_load_'
        FASTinfo['sm_def_file_template'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_def_'
    else:
        FASTinfo['sm_var_file'] = 'sm_var.txt'
        FASTinfo['sm_DEM_file'] = 'sm_DEM.txt'

    # list of variables that we are varying
    # FASTinfo['sm_var_names'] = ['r_max_chord']
    # FASTinfo['sm_var_names'] = ['chord_sub']
    FASTinfo['sm_var_names'] = ['r_max_chord', 'chord_sub', 'theta_sub']

    # indices of which variables are used
    # FASTinfo['sm_var_index'] = [[1]] # second point in distribution
    # FASTinfo['sm_var_index'] = [[1, 2]] # second/third point in distribution
    # FASTinfo['sm_var_index'] = [[0], [1, 2]] # r_max_chord & second/third point in distribution
    # FASTinfo['sm_var_index'] = [[0], [1]] # r_max_chord & second in chord distribution
    # FASTinfo['sm_var_index'] = [[0,1,2,3]] # chord_sub distribution
    # FASTinfo['sm_var_index'] = [[0]] # r_max_chord
    FASTinfo['sm_var_index'] = [[0], [0,1,2,3], [0,1,2,3]]


    # total num of variables used, variable index
    var_index = []
    num_var = 0

    for i in range(0, len(FASTinfo['sm_var_names'])):
        for j in range(0, len(FASTinfo['sm_var_index'][i])):
            num_var += 1
            var_index.append(i)

    FASTinfo['var_index'] = var_index
    FASTinfo['num_var'] = num_var

    # use smaller domain to sample points
    FASTinfo['restrict_lhs_domain'] = True

    return FASTinfo

# ========================================================================================================= #

# full factorial data choice
def create_surr_model_linear_options(FASTinfo, rotor):

    # how many different points will be used (linear)
    # FASTinfo['sm_var_max'] = [[4], [2,2,2,2]]
    FASTinfo['sm_var_max'] = [[10]]
    # FASTinfo['sm_var_max'] = [[4, 4]]
    # FASTinfo['sm_var_max'] = [[3], [3, 3]]
    # FASTinfo['sm_var_max'] = [[3], [3]]
    # FASTinfo['sm_var_max'] = [[3,3,3,3]]

    var_index = []
    for i in range(0, len(FASTinfo['sm_var_max'])):
        for j in range(0, len(FASTinfo['sm_var_max'][i])):
            var_index.append(i)

    FASTinfo['var_index'] = var_index

    # which specific points will be used for this run
    # probably argv variables from command line
    # FASTinfo['sm_var_spec'] = [[4], [2,2,2,2]]
    # FASTinfo['sm_var_spec'] = [[2], [1,1,1,1]]
    # FASTinfo['sm_var_spec'] = [[2]]

    # create options for which points are being used for dist. variables
    # how many total for each set of design variables

    try:
        FASTinfo['sm_var_spec'] = []
        for i in range(0, len(FASTinfo['sm_var_names'])):
            FASTinfo['sm_var_spec'].append([])

        for i in range(1, int(len(sys.argv))):
            FASTinfo['sm_var_spec'][var_index[i-1]].append(int(sys.argv[i]))
    except:
        raise Exception('A system argument is needed to calculate training points for the surrogate model.')
        # FASTinfo['sm_var_spec'] = [[3]]

    # print(FASTinfo['sm_var_spec'])
    # quit()

    # min, max values of design variables
    FASTinfo['sm_var_range'] = [[0.1, 0.5], [1.3, 5.3], [-10.0, 30.0], [0.005, 0.2], [0.005, 0.2]]

    # === initialize design variable values === #
    FASTinfo = initialize_dv(FASTinfo)

    FASTinfo['var_range'] = []
    # initialize rotor design variables
    for i in range(0, len(FASTinfo['sm_var_names'])):

        # create var_range
        if FASTinfo['sm_var_names'][i] == 'r_max_chord':
            var_range = FASTinfo['sm_var_range'][0]
        elif FASTinfo['sm_var_names'][i] == 'chord_sub':
            var_range = FASTinfo['sm_var_range'][1]
        elif FASTinfo['sm_var_names'][i] == 'theta_sub':
            var_range = FASTinfo['sm_var_range'][2]
        elif FASTinfo['sm_var_names'][i] == 'sparT':
            var_range = FASTinfo['sm_var_range'][3]
        elif FASTinfo['sm_var_names'][i] == 'teT':
            var_range = FASTinfo['sm_var_range'][4]
        else:
            Exception('A surrogate model variable was listed that is not a design variable.')


        for j in range(0, len(FASTinfo['sm_var_max'][i])):

            index = FASTinfo['sm_var_index'][i][j]

            sm_range = np.linspace(var_range[0], var_range[1], FASTinfo['sm_var_max'][i][j])

            FASTinfo['var_range'].append(sm_range)

            if hasattr(FASTinfo[FASTinfo['sm_var_names'][i]+'_init'], '__len__'):

                FASTinfo[FASTinfo['sm_var_names'][i]+'_init'][index] = sm_range[FASTinfo['sm_var_spec'][i][j]-1]

            else:

                FASTinfo[FASTinfo['sm_var_names'][i] + '_init'] = sm_range[FASTinfo['sm_var_spec'][i][j]-1]

    # set design variables in rotor dictionary

    if FASTinfo['check_point_dist']:

        plt.figure()

        plt.title('Linear Sampling Example')
        plt.xlabel(FASTinfo['sm_var_names'][0])
        plt.ylabel(FASTinfo['sm_var_names'][1])


        # plt.xlim([0.9*min(FASTinfo['var_range'][0]), 1.1*max(FASTinfo['var_range'][0])])
        # plt.ylim([0.9*min(FASTinfo['var_range'][1]), 1.1*max(FASTinfo['var_range'][1])])

        for i in range(0, FASTinfo['sm_var_max'][0][0]):
            for j in range(0, FASTinfo['sm_var_max'][1][0]):
                plt.plot( FASTinfo['var_range'][0][i], FASTinfo['var_range'][1][j], 'ob', label='training points')

        # plt.legend()
        plt.savefig(FASTinfo['dir_saved_plots'] + '/linear_' + FASTinfo['description'] + '.png')
        plt.show()

        quit()

    return FASTinfo, rotor

# ========================================================================================================= #

# Latin Hypercube-Sampling data choice
def create_surr_model_lhs_options(FASTinfo, rotor):

    # determine initial values
    FASTinfo = initialize_dv(FASTinfo)

    try:
        FASTinfo['sm_var_spec'] = int(sys.argv[1])
    except:
        # raise Exception('Need to have system input when latin-hypercube sampling used.')
        FASTinfo['sm_var_spec'] = 0


    # add FASTinfo['sm_dir'] so simultaneous runs don't clobber each other
    FASTinfo['sm_dir'] = FASTinfo['opt_dir'] + '/sm_' + str(FASTinfo['sm_var_spec'])

    # create directory
    if not os.path.isdir(FASTinfo['opt_dir']):
        os.mkdir(FASTinfo['opt_dir'])

    # name of sm .txt files (will be in description folder)

    if not os.path.isdir(FASTinfo['opt_dir'] + '/' + FASTinfo['sm_var_out_dir']):
        os.mkdir(FASTinfo['opt_dir'] + '/' + FASTinfo['sm_var_out_dir'])

    FASTinfo['sm_var_file'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_var_' + str(FASTinfo['sm_var_spec']) + '.txt'
    FASTinfo['sm_DEM_file'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_DEM_' + str(FASTinfo['sm_var_spec']) + '.txt'
    FASTinfo['sm_load_file'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_load_' + str(FASTinfo['sm_var_spec']) + '.txt'
    FASTinfo['sm_def_file'] = FASTinfo['sm_var_out_dir'] + '/' + 'sm_def_' + str(FASTinfo['sm_var_spec']) + '.txt'

    var_index = FASTinfo['var_index']
    num_var = FASTinfo['num_var']

    # ranges of said variables
    # min, max values of design variables
    FASTinfo['sm_var_range'] = [[0.1, 0.5], [1.3, 5.3], [-10.0, 30.0], [0.005, 0.2], [0.005, 0.2]]

    # do linear hypercube spacing
    from pyDOE import lhs

    point_file = FASTinfo['opt_dir'] + '/pointfile.txt'

    if not os.path.isfile(point_file):

        print('Creating training point file...')

        points = lhs(num_var, samples=FASTinfo['num_pts'], criterion='center')

        f = open(point_file,"w+")

        for i in range(0, len(points)):
            for j in range(0, len(points[i])):
                f.write(str(points[i,j]))
                f.write(' ')

            f.write('\n')

        f.close()

    lines = open(point_file,"r+").readlines()

    points = np.zeros([FASTinfo['num_pts'], num_var])
    for i in range(0, len(lines)):
        spec_line = lines[i].strip('\n').split()
        for j in range(0, len(spec_line)):
            points[i,j] = float(spec_line[j])

    new_var_list = []
    old_var_list = []
    FASTinfo['var_range'] = []

    for i in range(0, num_var):

        spec_var_name = FASTinfo['sm_var_names'][var_index[i]]

        # create var_range
        if spec_var_name == 'r_max_chord':
            var_range = FASTinfo['sm_var_range'][0]
        elif spec_var_name == 'chord_sub':
            var_range = FASTinfo['sm_var_range'][1]
        elif spec_var_name == 'theta_sub':
            var_range = FASTinfo['sm_var_range'][2]
        elif spec_var_name == 'sparT':
            var_range = FASTinfo['sm_var_range'][3]
        elif spec_var_name == 'teT':
            var_range = FASTinfo['sm_var_range'][4]
        else:
            Exception('A surrogate model variable was listed that is not a design variable.')

        # decrease var_range
        # doing this will increase surrogate model accuracy,
        # since the points sampled are going to clustered closer together

        if FASTinfo['restrict_lhs_domain']:

            index = -1
            # determine which index in spec_var_name we're at

            for j in range(len(var_index[0:i+1])):
                if var_index[i] == var_index[j]:
                    index += 1

            # get specific variable initial value
            if spec_var_name == 'r_max_chord':
                spec_val = FASTinfo[spec_var_name + '_init']
            else:
                spec_val = FASTinfo[spec_var_name + '_init'][index]

            # create new range
            FASTinfo['range_frac'] = 0.35  # 0.15, 0.25, 0.35
            range_frac =  FASTinfo['range_frac']
            range_len = var_range[1]-var_range[0]
            new_var_range = [spec_val-range_frac*range_len, spec_val+range_frac*range_len]

            # if new range is smaller/larger than old range, adjust
            if new_var_range[0] < var_range[0]:
                new_var_range[0] = var_range[0]
            if new_var_range[1] > var_range[1]:
                new_var_range[1] = var_range[1]

            # append lists
            new_var_list.append(new_var_range)
            old_var_list.append(var_range)


            # print('test')
            # print(spec_val)
            # print(new_var_range)
            # print(var_range)
            # print('--- ---')

            # set var_range to new_var_range
            var_range = new_var_range

        points[:,i] = points[:,i]*(var_range[1]-var_range[0]) + var_range[0]

        FASTinfo['var_range'].append(var_range)

    if FASTinfo['check_var_domains']:

        # chord_sub domain plot
        plt.figure()
        j = 1

        for i in range(len(FASTinfo['var_index'])):

            if FASTinfo['var_index'][i] == 1: # chord_sub
                plt.plot([j,j], old_var_list[i], 'bx')
                plt.plot([j,j], new_var_list[i], '--rx')
                j += 1

        plt.xticks(np.linspace(1,j-1,j-1))
        plt.xlabel('chord variable index')
        plt.ylabel('Var Domain (m)')
        plt.title('chord domain, restriction: ' + str(FASTinfo['range_frac']*100.0) + '%')


        plt.savefig(FASTinfo['dir_saved_plots'] + '/' + 'chord_sub_domain' + str(int(FASTinfo['range_frac']*100.0)) + '.png')
        plt.show()

        plt.close()

        # theta_sub domain plot
        plt.figure()
        j = 1

        for i in range(len(FASTinfo['var_index'])):

            if FASTinfo['var_index'][i] == 2: # theta_sub
                plt.plot([j,j], old_var_list[i], 'bx')
                plt.plot([j,j], new_var_list[i], '--rx')
                j += 1

        plt.xticks(np.linspace(1,j-1,j-1))
        plt.xlabel('twist variable index')
        plt.ylabel('Var Domain (m)')
        plt.title('twist domain, restriction: ' + str(FASTinfo['range_frac']*100.0) + '%')


        plt.savefig(FASTinfo['dir_saved_plots'] + '/' + 'theta_sub_domain_' + str(int(FASTinfo['range_frac']*100.0)) + '.png')
        plt.show()

        plt.close()

        quit()

    # quit()
    # === plot checks === #

    if FASTinfo['check_cv']:

        cv_point_file = FASTinfo['opt_dir'] + '_cv' + '/pointfile.txt'

        cv_lines = open(cv_point_file, "r+").readlines()

        cv_points = np.zeros([len(cv_lines), num_var])
        for i in range(0, len(lines)):
            cv_spec_line = cv_lines[i].strip('\n').split()
            for j in range(0, len(cv_spec_line)):
                cv_points[i, j] = float(cv_spec_line[j])

        for i in range(0, num_var):

            spec_var_name = FASTinfo['sm_var_names'][var_index[i]]

            # create var_range
            if spec_var_name == 'r_max_chord':
                var_range = FASTinfo['sm_var_range'][0]
            elif spec_var_name == 'chord_sub':
                var_range = FASTinfo['sm_var_range'][1]
            elif spec_var_name == 'theta_sub':
                var_range = FASTinfo['sm_var_range'][2]
            elif spec_var_name == 'sparT':
                var_range = FASTinfo['sm_var_range'][3]
            elif spec_var_name == 'teT':
                var_range = FASTinfo['sm_var_range'][4]
            else:
                Exception('A surrogate model variable was listed that is not a design variable.')

            #
            cv_points[:, i] = cv_points[:, i] * (var_range[1] - var_range[0]) + var_range[0]

        plt.figure()

        plt.title('Latin Hypercube Cross Validation Example')
        plt.xlabel(FASTinfo['sm_var_names'][0])
        plt.ylabel(FASTinfo['sm_var_names'][1])

        # plt.xlim(FASTinfo['var_range'][0])
        # plt.ylim(FASTinfo['var_range'][1])

        plt.plot(points[:,0], points[:,1], 'o', label='training points')
        plt.plot(cv_points[:,0], cv_points[:,1], 'o', label='cross-validation points')
        plt.legend()

        plt.savefig(FASTinfo['dir_saved_plots'] + '/lhs_cv_' + FASTinfo['description'] + '.png')
        plt.show()

        quit()

    if FASTinfo['check_point_dist']:

        plt.figure()

        plt.title('Latin Hypercube Sampling Example')
        plt.xlabel(FASTinfo['sm_var_names'][0])
        plt.ylabel(FASTinfo['sm_var_names'][1])

        # plt.xlim(FASTinfo['var_range'][0])
        # plt.ylim(FASTinfo['var_range'][1])

        plt.plot(points[:,0], points[:,1], 'o', label='training points')
        plt.legend()

        plt.savefig(FASTinfo['dir_saved_plots'] + '/lhs_' + FASTinfo['description'] + '.png')
        plt.show()

        quit()

    # === assign values === #

    cur_var = 0
    for i in range(0, len(FASTinfo['sm_var_names'])):

        # spec_var_name = FASTinfo['sm_var_names'][var_index[i]]
        spec_var_name = FASTinfo['sm_var_names'][i]

        for j in range(0, len(FASTinfo['sm_var_index'][i])):

            # print('-- i j pair ---')
            # print(i)
            # print(j)

            if hasattr( FASTinfo[spec_var_name + '_init'], '__len__'):
                FASTinfo[spec_var_name + '_init'][FASTinfo['sm_var_index'][i][j]] = points[FASTinfo['sm_var_spec'],cur_var]
            else:
                FASTinfo[spec_var_name + '_init'] = points[FASTinfo['sm_var_spec'],cur_var]

            cur_var += 1

    return FASTinfo, rotor

# ========================================================================================================= #

# initialize design variables
def initialize_dv(FASTinfo):

    FASTinfo['r_max_chord_init'] = 0.23577  # (Float): location of max chord on unit radius
    FASTinfo['chord_sub_init'] = np.array([3.2612, 4.5709, 3.3178,
                                   1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    FASTinfo['theta_sub_init'] = np.array([13.2783, 7.46036, 2.89317,
                                   -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    FASTinfo['sparT_init'] = np.array(
        [0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    FASTinfo['teT_init'] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters

    return FASTinfo


# ========================================================================================================= #


def DLC_call(dlc, wnd_list, wnd_list_type, rand_seeds, mws, num_sgp, parked_list):

    # for each possible dlc case
    # add .wnd files for each specified DLC

    # nominal .wnd file (not a DLC)
    if dlc == 'DLC_0_0':
        wnd_list.append('nom.wnd')

        for i in range(0, num_sgp):
            wnd_list_type.append('nonturb')
            parked_list.append('no')

    # turbulent DLCs
    if dlc == 'DLC_1_2':
        for i in range(1, len(rand_seeds) + 1):
            for j in range(0, len(mws)):

                inp_file_hh = 'dlc_{0}_seed{1}_mws{2}.hh'.format('NTM', i, int(mws[j]))
                wnd_list.append(inp_file_hh)

                for k in range(0, num_sgp):
                    wnd_list_type.append('turb')
                    parked_list.append('no')

    if dlc == 'DLC_1_3':
        for i in range(1, len(rand_seeds) + 1):
            for j in range(0, len(mws)):
                inp_file_hh = 'dlc_{0}_seed{1}_mws{2}.hh'.format('1ETM', i, int(mws[j]))
                wnd_list.append(inp_file_hh)

                for k in range(0, num_sgp):
                    wnd_list_type.append('turb')
                    parked_list.append('no')

    # nonturbulent DLCs
    if dlc == 'DLC_1_4':
        wnd_list.append('ECD+R+2.0.wnd')
        wnd_list.append('ECD+R-2.0.wnd')
        wnd_list.append('ECD-R+2.0.wnd')
        wnd_list.append('ECD-R-2.0.wnd')
        wnd_list.append('ECD+R.wnd')
        wnd_list.append('ECD-R.wnd')

        for i in range(0, num_sgp):
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')

            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')

    if dlc == 'DLC_1_5':
        wnd_list.append('EWSH+12.0.wnd')
        wnd_list.append('EWSH-12.0.wnd')
        wnd_list.append('EWSV+12.0.wnd')
        wnd_list.append('EWSV-12.0.wnd')

        for i in range(0, num_sgp):
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')
            wnd_list_type.append('nonturb')

            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')
            parked_list.append('no')

    if dlc == 'DLC_6_1':
        wnd_list.append('EWM50.wnd')

        for i in range(0, num_sgp):
            wnd_list_type.append('nonturb')
            parked_list.append('yes')

    if dlc == 'DLC_6_3':
        wnd_list.append('EWM01.wnd')

        for i in range(0, num_sgp):
            wnd_list_type.append('nonturb')
            parked_list.append('yes')


    return wnd_list, wnd_list_type

# ========================================================================================================= #


def Use_FAST_DEMs(FASTinfo, rotor, checkplots):

    # from FASTinfo, get number of wind files
    caseids = []
    for i in range(0, len(FASTinfo['wnd_list'])):
        caseids.append("WNDfile{0}".format(i+1))

    # create array to hold all DEMs (for each wnd_file)
    DEMx_master_array = np.zeros([len(FASTinfo['wnd_list']), 18])
    DEMy_master_array = np.zeros([len(FASTinfo['wnd_list']), 18])

    for i in range(0, len(FASTinfo['wnd_list'])):

        DEMrange = [0,1,8,15]
        sgp_range = [1,1,2,3]

        lines_x = []
        lines_y = []

        for j in DEMrange:

            if j == DEMrange[0]:
                sgp = sgp_range[0]
            elif j == DEMrange[1]:
                sgp = sgp_range[1]
            elif j == DEMrange[2]:
                sgp = sgp_range[2]
            elif j == DEMrange[3]:
                sgp = sgp_range[3]

            spec_wnd_dir = FASTinfo['description'] + '/' + 'sgp' + str(sgp) + '/' + caseids[i - 1] + '_sgp' + str(sgp)

            FAST_wnd_directory = ''.join((FASTinfo['path'], 'RotorSE_FAST/' \
                    'RotorSE/src/rotorse/FAST_files/Opt_Files/', spec_wnd_dir))

            # xDEM / yDEM files

            if j == 0:
                xDEM_file = FAST_wnd_directory + '/' + 'xRoot.txt'
                yDEM_file = FAST_wnd_directory + '/' + 'yRoot.txt'
            else:
                xDEM_file = FAST_wnd_directory + '/' + 'xDEM_' + str(j) + '.txt'
                yDEM_file = FAST_wnd_directory + '/' + 'yDEM_' + str(j) + '.txt'

            lines_x.append([line.rstrip('\n') for line in open(xDEM_file)])
            lines_y.append([line.rstrip('\n') for line in open(yDEM_file)])

        xDEM = []
        yDEM = []

        for j in range(0,4):
            for k in range(0, len(lines_x[j])):
                xDEM.append(float(lines_x[j][k]))

        for j in range(0,4):
            for k in range(0, len(lines_y[j])):
                yDEM.append(float(lines_y[j][k]))


        xDEM = np.array(xDEM)
        yDEM = np.array(yDEM)

        DEMx_master_array[i][0:18] = xDEM
        DEMy_master_array[i][0:18] = yDEM

        # TODO: plot check for different wnd files

    # create DEM plots using DEMx_master_array, DEMy_master_array
    FASTinfo['createDEMplot'] = False

    if FASTinfo['createDEMplot']:

        plt.figure()
        plt.xlabel('strain gage position')
        plt.ylabel('DEM (kN*m)')
        plt.title('DEMx for different .wnd files')  #: Bending Moment at Spanwise Station #1, Blade #1')
        for i in range(0, len(FASTinfo['wnd_list'])):
            plt.plot(DEMx_master_array[i][0:18], label = FASTinfo['wnd_list'][i])

        plt.legend()
        plt.show()

        quit()

    # set rotor parameters
    rotor['rstar_damage'] = np.insert(rotor['r_aero'],0,0.0)

    # From DEMx_master_array, DEMy_master_array, determine DEMx_max, DEMy_max
    DEMx_max = np.zeros([1+len(rotor['r_aero']), 1])
    DEMy_max = np.zeros([1+len(rotor['r_aero']), 1])

    # DEMx_max_wnd_list = np.zeros([len(DEMx_max),1])
    DEMx_max_wnd_list = []
    DEMy_max_wnd_list = []

    for i in range(0,len(DEMx_max)):
        for j in range(0, len(FASTinfo['wnd_list'])):

            # determine max DEMx
            if DEMx_max[i] < DEMx_master_array[j][i] :

                # add name of .wnd file
                DEMx_max_wnd_list.append(FASTinfo['wnd_list'][j])

                # set max DEM
                DEMx_max[i] = DEMx_master_array[j][i]

            # determine max DEMy
            if DEMy_max[i] < DEMy_master_array[j][i] :

                # add name of .wnd file
                # DEMy_max_wnd_list[i] = FASTinfo['wnd_list'][j]
                DEMy_max_wnd_list.append(FASTinfo['wnd_list'][j])

                # set max DEM
                DEMy_max[i] = DEMy_master_array[j][i]

    # create list of active .wnd files at current design
    DEM_master_wnd_list = []
    for i in range(0, len(DEMx_max)):
        if not (DEMx_max_wnd_list[i] in DEM_master_wnd_list):
            DEM_master_wnd_list.append(DEMx_max_wnd_list[i])
    for i in range(0, len(DEMy_max)):
        if not (DEMy_max_wnd_list[i] in DEM_master_wnd_list):
            DEM_master_wnd_list.append(DEMy_max_wnd_list[i])

    active_wnd_file = FASTinfo['opt_dir'] + '/' + 'active_wnd.txt'
    file_wnd = open(active_wnd_file, "w")
    for j in range(0, len(DEM_master_wnd_list)):
        # write to xDEM file
        file_wnd.write(str(DEM_master_wnd_list[j]) + '\n')
    file_wnd.close()

    rotor['Mxb_damage'] = DEMx_max*10.0**3.0 # kN*m to N*m
    rotor['Myb_damage'] = DEMy_max*10.0**3.0 # kN*m to N*m

    # xDEM/yDEM files
    xDEM_file = FASTinfo['opt_dir'] + '/' + 'xDEM_max.txt'
    file_x = open(xDEM_file, "w")
    for j in range(0, len(rotor['Mxb_damage'])):
        # write to xDEM file
        file_x.write(str(rotor['Mxb_damage'][j]) + '\n')
    file_x.close()

    yDEM_file = FASTinfo['opt_dir'] + '/' + 'yDEM_max.txt'
    file_y = open(yDEM_file, "w")
    for j in range(0, len(rotor['Myb_damage'])):
        # write to xDEM file
        file_y.write(str(rotor['Myb_damage'][j]) + '\n')
    file_y.close()

    if checkplots:
        plot_DEMs(rotor)
        quit()

    return rotor

# ========================================================================================================= #

def plot_DEMs(rotor):
    rstar_damage_init = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
                                      0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933,
                                      0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    Mxb_damage_init = 1e3 * np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
                                          1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002,
                                          1.6339E+002,
                                          1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000,
                                          4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    Myb_damage_init = 1e3 * np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
                                          1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002,
                                          4.5229E+002,
                                          3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001,
                                          1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction

    plt.figure()
    plt.plot(rstar_damage_init,Mxb_damage_init,label='Nominal Values in run.py script')
    plt.plot(rotor['rstar_damage'],rotor['Mxb_damage'],'--x',label='FAST Calculated')
    plt.xlabel('Blade Fraction')
    plt.ylabel('Mxb (N*m)')
    plt.title('Mxb Comparison')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(rstar_damage_init,Myb_damage_init,label='Nominal Values in run.py script')
    plt.plot(rotor['rstar_damage'],rotor['Myb_damage'],'--x',label='FAST Calculated')
    plt.xlabel('Blade Fraction')
    plt.ylabel('Myb (N*m)')
    plt.title('Myb Comparison')
    plt.legend()
    plt.show()

# ========================================================================================================= #

def add_outputs(FASTinfo):

    FASTinfo['output_list'] = []

    # OutputList = open("FAST_Files/FASTOutputList_full.txt", 'r')
    OutputList = open("FAST_Files/FASTOutputList.txt", 'r')

    lines = OutputList.read().split('\n')

    for i in range(0, len(lines)):  # in OutputList:
        FASTinfo['output_list'].append(lines[i])

    return FASTinfo

# ========================================================================================================= #

def extract_results(rotor,FASTinfo):

    # results file name
    opt_dir = FASTinfo['opt_dir']

    file_name = opt_dir + '/' + 'opt_results.txt'

    resultsfile = open(file_name, 'w')

    # design variables
    resultsfile.write(str(rotor['r_max_chord']) + '\n' )
    resultsfile.write(str(rotor['chord_sub']) + '\n' )
    resultsfile.write(str(rotor['theta_sub']) + '\n' )
    resultsfile.write(str(rotor['sparT']) + '\n' )
    resultsfile.write(str(rotor['teT']) + '\n' )

    # TODO : add additional parameters to record

    resultsfile.close()

# ========================================================================================================= #

