import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# with open('omdaoCase1.out') as f:
#     results = f.readlines()[9:]
#
# results = results.replace("\n", "")
# results = np.loadtxt('results.txt')

plot_constraints = 0
plotcurves = 0
plot_instance = 0
plot_lin_reg = 0
plot_chord = 1
plot_span_moments = 0
plot_ltd = 0
plot_theta = 0

if plot_ltd == 1:
    r_max_chord = np.linspace(0.1, 0.5, 20)

    ltd = np.loadtxt('life_time_damage.txt')
    ltd_nc = np.loadtxt('life_time_damage_no_cor.txt')


    plt.figure()
    plt.title('Fatigue at Span Location 4')
    plt.plot(r_max_chord,ltd/20, label = 'Using Goodman Correction')
    plt.plot(r_max_chord,ltd_nc/20, label = 'Without Goodman Correction')
    plt.xlabel('Maximum Chord Location (m)')
    plt.ylabel('Fatigue')
    # plt.ylabel('Generated Power (kW)')
    #plt.xlim([3,25])
    #plt.xlim([11,15])
    plt.ylim([0,1.0])
    #plt.ylim([0,6000])
    plt.legend(loc=2)
    plt.show()
    quit()

if plot_span_moments == 1:

    tl = 100
    time = np.linspace(0,tl,tl/.0125+1)

    span_x = np.loadtxt('Spn2Mlxb1.txt')

    plt.figure()
    plt.title('DLC 1.4 (Extreme Coherent Gust w/Direction Change) ')
    plt.plot(time,span_x)
    plt.xlabel('Time (s)')
    plt.ylabel('Local Moment @ Span Station 2(kN*m)')
    #plt.xlim([3,25])
    #plt.ylim([0,6000])
    plt.show()

if plot_constraints == 1:

    r_max_chord = np.linspace(0.1,0.5,20)

    buckling = np.loadtxt('buckling.txt')
    crit_def = np.loadtxt('critical_deflection.txt')
    fs = np.loadtxt('flapwise_strain.txt')
    es = np.loadtxt('edgewise_strain.txt')

    fs_stress = fs
    es_stress = es

    plt.figure()
    plt.title('DLC 1.4 (Extreme Coherent Gust w/Direction Change) ')
    plt.plot(r_max_chord,crit_def)
    plt.xlabel('Max Chord Location Fraction')
    plt.ylabel('Critical Deflection (m)')
    #plt.xlim([3,25])
    #plt.ylim([0,6000])
    plt.show()

    plt.figure()
    plt.title('DLC 1.4 (Extreme Coherent Gust w/Direction Change) ')
    plt.plot(r_max_chord,fs_stress)
    plt.xlabel('Max Chord Location Fraction')
    plt.ylabel('Flapwise Maximum Stress (MPa)')
    #plt.xlim([3,25])
    #plt.ylim([0,6000])
    plt.show()

    plt.figure()
    plt.title('DLC 1.4 (Extreme Coherent Gust w/Direction Change) ')
    plt.plot(r_max_chord,es_stress)
    plt.xlabel('Max Chord Location Fraction')
    plt.ylabel('Edgewise Maximum Stress (MPa)')
    #plt.xlim([3,25])
    #plt.ylim([0,6000])
    plt.show()

if plot_chord == 1:
    chord0 = [3.48606523,  3.9036488,   4.24039191,  4.50990016 , 4.5558932,  4.44615193,
     4.27100081,  4.04524342,  3.78368336,  3.5011242,   3.21194817,  2.92042342,
     2.62230309,  2.31332848,  2.04448308,  1.82019896,  1.5863591]
    r = [2.8667,   5.6,      8.3333,  11.75,    15.85,    19.95,    24.05,    28.15,
     32.25,    36.35,    40.45,   44.55,    48.65,    52.75,    56.1667,  58.9,
     61.6333]
    chord_opt = [ 3.2182206,   3.69902692,  4.1312615,   4.58993873,  4.99808184,  5.22315577,
  5.2356485,   5.00602254,  4.49234017, 3.78968888,  3.06783205,  2.49653305,
  2.13420411,  1.81104443,  1.58270451,  1.43764649,  1.33433803]


    plt.figure()
    plt.title('Optimize Chord Distribution')
    #plt.plot(r,chord0,'-*', label = 'original design')
    plt.plot(r,chord_opt,'-*', label = 'optimized design using RotorSE')
    plt.plot(r,chord_opt,'-*', label='optimized design w/FAST constraints')
    plt.xlabel('Blade Location (m)')
    plt.ylabel('Chord (m)')
    # plt.ylabel('Generated Power (kW)')
    #plt.xlim([3,25])
    #plt.xlim([11,15])
    #plt.ylim([0,50])
    #plt.ylim([0,6000])
    plt.legend(loc=3)
    plt.show()
    quit()

if plot_theta == 1:
    theta0 = [ 12.00829859,  12.00829859,  12.00829859,  12.00829859,  11.37397181,
   9.99299004,   8.67663126,   7.42041502,   6.22929502,   5.11206596,
   4.07146573,   3.11023224,   2.23640415,   1.46081228 ,  0.88587252,
   0.47028779,   0.09230388]
    theta_opt = [ 12.08138836 , 12.08138836,  12.08138836,  12.08138836 , 10.31557938,
   8.68131875,   7.16479699  , 5.7522359    ,4.46370386,   3.32471881,
   2.33528076  , 1.4953897  ,  0.82687696 ,  0.37134189 ,  0.14267032,
   0.05072835  , 0.03365287]
    theta_fast = 1.2*(np.array(theta_opt))
    r = [2.8667,   5.6,      8.3333,  11.75,    15.85,    19.95,    24.05,    28.15,
     32.25,    36.35,    40.45,   44.55,    48.65,    52.75,    56.1667,  58.9,
     61.6333]


    plt.figure()
    plt.title('Optimize Twist Distribution')
    #plt.plot(r,theta0,'-*', label = 'original design')
    plt.plot(r,theta_opt,'-*', label = 'optimized design using RotorSE')
    plt.plot(r,theta_opt,'-*', label='optimized design w/FAST constraints')
    plt.xlabel('Blade Location (m)')
    plt.ylabel('Twist (deg)')
    # plt.ylabel('Generated Power (kW)')
    #plt.xlim([3,25])
    #plt.xlim([11,15])
    #plt.ylim([0,50])
    #plt.ylim([0,6000])
    plt.legend(loc=1)
    plt.show()
    #quit()

if plotcurves == 1:

    #results = np.loadtxt('GenTq_GenPwr_3_25_100.txt')
    results = np.loadtxt('GenTq_GenPwr.txt')

    windspeed = results[:,2]
    gen_tq = results[:,0]
    gen_pwr = results[:,1]

    plt.figure()
    plt.plot(windspeed,gen_tq)
    plt.xlabel('Incoming Wind Speed (m/s)')
    plt.ylabel('Generated Torque (kN*m)')
    # plt.ylabel('Generated Power (kW)')
    plt.xlim([3,25])
    #plt.xlim([11,15])
    plt.ylim([0,50])
    #plt.ylim([0,6000])
    plt.show()

    plt.figure()
    plt.plot(windspeed,gen_pwr)
    plt.xlabel('Incoming Wind Speed (m/s)')
    #plt.ylabel('Generated Torque (kN*m)')
    plt.ylabel('Generated Power (kW)')
    plt.xlim([3,25])
    #plt.ylim([0,50])
    plt.ylim([0,6000])
    plt.show()

if plot_instance == 1:

    tl = 100
    time = np.linspace(0,tl,tl/.0125+1)

    #results = np.loadtxt('OoPDefl1.txt')

    results = np.loadtxt('GenPwr.txt')
    results = np.array(results)


    #plot Generated Power
    plt.figure()
    plt.plot(time,results)
    plt.xlabel('Time (s)')
    plt.ylabel('Generated Power (kW)')
    plt.title('Wind Speed = 11.4 m/s')
    plt.xlim([0,tl])
    plt.show()

    results = np.loadtxt('GenTq.txt')
    results = np.array(results)

    #Plot Generated Torque
    plt.figure()
    plt.plot(time, results)
    plt.xlabel('Time (s)')
    # plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.ylabel('Generated Torque (kN*m)')
    plt.title('Wind Speed = 11.4 m/s')
    plt.xlim([0, tl])
    plt.show()

    results = np.loadtxt('OoPDefl1.txt')
    results = np.array(results)

    #Plot Out of Plane Deflection
    plt.figure()
    plt.plot(time, results)
    plt.xlabel('Time (s)')
    plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.title('Wind Speed = 11.4 m/s')
    plt.xlim([0, tl])
    plt.show()

if plot_lin_reg == 1:

    tl = 50
    time = np.linspace(0,tl,tl/.0125+1)

    results = np.loadtxt('GenTq.txt')
    results = np.array(results)

    points = 120
    numVal = len(results)/points
    slope = np.zeros([points,1])
    avg_y = np.zeros([points,1])

    for i in range(0, points):
        y = results[1+i*numVal:1+(i+1)*numVal]
        x = time[1+i*numVal:1+(i+1)*numVal]

        # linear regression
        slope[i], intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # batch mean
        avg_y[i] = np.sum(y)/len(y)

    plt.figure()
    plt.plot(np.linspace(0,tl,len(slope)), slope)
    plt.xlabel('Time (s)')
    # plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.ylabel('Slope of Linear Regression Line of Generated Torque (kN*m/s)')
    plt.title('Linear Regression Slope')
    plt.xlim([0, tl])
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0,tl,len(avg_y)), avg_y)
    plt.xlabel('Time (s)')
    # plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.ylabel('Batch Mean of Generated Torque (kN*m/s)')
    plt.title('Batch Mean')
    plt.xlim([0, tl])
    plt.show()




    results = np.loadtxt('GenPwr.txt')
    results = np.array(results)

    points = 250
    numVal = len(results) / points
    slope = np.zeros([points, 1])
    avg_y = np.zeros([points, 1])

    for i in range(0, points):
        y = results[1 + i * numVal:1 + (i + 1) * numVal]
        x = time[1 + i * numVal:1 + (i + 1) * numVal]

        slope[i], intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # batch mean
        avg_y[i] = np.sum(y)/len(y)

    plt.figure()
    plt.plot(np.linspace(0, tl, len(slope)), slope)
    plt.xlabel('Time (s)')
    # plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.ylabel('Slope of Linear Regression Line of Generated Power (kW/s)')
    plt.title('Linear Regression Slope')
    plt.xlim([0, tl])
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, tl, len(avg_y)), avg_y)
    plt.xlabel('Time (s)')
    # plt.ylabel('Out-Of-Plane Blade Tip Deflection (m)')
    plt.ylabel('Batch Mean of Generated Power (kW/s)')
    plt.title('Batch Mean')
    plt.xlim([0, tl])
    plt.ylim([0,6000])
    plt.show()




# speed = [3,4,5,6,7,8,9,10,11,12,13,14,15,16]
# d = [0.08,0.1666,0.2644,0.3722,0.5044,0.6475,0.8186,0.9935,1.17,1.337,1.504,1.686,1.858,2.029]
#
# plt.figure()
# plt.plot(speed,d,'--*')
# plt.xlabel('Incoming Wind Speed (m/s)')
# plt.ylabel('Max Deflection (m)')
# plt.plot([14,15,16],[1.686,1.858,2.029],'or', label='Small Angle Assumption Violated')
# plt.legend(loc=2)
# plt.title('FAST Check')
# plt.xlim([3,16])
# plt.show()
