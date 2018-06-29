import numpy as np
import os
from StringIO import StringIO
import csv
import commonse

from openmdao.api import Component, Group, IndepVarComp

from akima import Akima, akima_interp_with_derivs
from ccblade.ccblade_component import CCBladeGeometry
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp

NINPUT = 5
TURBULENCE_CLASS = commonse.enum.Enum('A B C')
TURBINE_CLASS = commonse.enum.Enum('I II III')
DRIVETRAIN_TYPE = commonse.enum.Enum('geared single_stage multi_drive pm_direct_drive')

class ReferenceBlade(object):
    def __init__(self):
        self.name           = None
        self.rating         = None
        self.turbine_class  = None
        self.drivetrainType = None
        self.downwind       = None
        self.nBlades        = None
        
        self.bladeLength   = None
        self.hubFraction   = None
        self.precone       = None
        self.tilt          = None
        
        self.r         = None
        self.r_in      = None
        self.npts      = None
        self.chord     = None
        self.chord_ref = None
        self.theta     = None
        self.precurve  = None
        self.precurveT = None
        self.presweep  = None
        self.airfoils  = None

        self.r_cylinder     = None
        self.r_max_chord    = None
        self.spar_thickness = None
        self.te_thickness   = None
        self.le_location    = None
        
        self.web1 = None
        self.web2 = None
        self.web3 = None
        self.web4 = None
        
        self.sector_idx_strain_spar = None
        self.sector_idx_strain_te   = None

        self.control_Vin  = None
        self.control_Vout = None
        self.control_tsr  = None
        self.control_minOmega = None
        self.control_maxOmega = None
        
    def setRin(self):
        self.r_in = np.r_[0.0, self.r_cylinder, np.linspace(self.r_max_chord, 1.0, NINPUT-2)]
        
    def getAeroPath(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), self.name+'_AFFiles')
    
    def getStructPath(self):
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), self.name+'_PreCompFiles')

    def getAirfoilCoordinates(self):
        data = []
        for a in self.airfoils:
            coord = np.loadtxt(a.replace('.dat','.pfl'), skiprows=2)
            data.append(coord)
        return data
        
class NREL5MW(ReferenceBlade):
    def __init__(self):
        super(NREL5MW, self).__init__()

        # Raw data from https://www.nrel.gov/docs/fy09osti/38060.pdf
        #Node,RNodes,AeroTwst,DRNodes,Chord,Airfoil,Table
        #(-),(m),(deg),(m),(m),(-)
        raw = StringIO(
        """1,2.8667,13.308,2.7333,3.542,Cylinder1.dat
        2,5.6000,13.308,2.7333,3.854,Cylinder1.dat
        3,8.3333,13.308,2.7333,4.167,Cylinder2.dat
        4,11.7500,13.308,4.1000,4.557,DU40_A17.dat
        5,15.8500,11.480,4.1000,4.652,DU35_A17.dat
        6,19.9500,10.162,4.1000,4.458,DU35_A17.dat
        7,24.0500,9.011,4.1000,4.249,DU30_A17.dat
        8,28.1500,7.795,4.1000,4.007,DU25_A17.dat
        9,32.2500,6.544,4.1000,3.748,DU25_A17.dat
        10,36.3500,5.361,4.1000,3.502,DU21_A17.dat
        11,40.4500,4.188,4.1000,3.256,DU21_A17.dat
        12,44.5500,3.125,4.1000,3.010,NACA64_A17.dat
        13,48.6500,2.319,4.1000,2.764,NACA64_A17.dat
        14,52.7500,1.526,4.1000,2.518,NACA64_A17.dat
        15,56.1667,0.863,2.7333,2.313,NACA64_A17.dat
        16,58.9000,0.370,2.7333,2.086,NACA64_A17.dat
        17,61.6333,0.106,2.7333,1.419,NACA64_A17.dat""")
        
        # Name to recover / lookup this info
        self.name     = '5MW'
        self.rating   = 5e6
        self.nBlades  = 3
        self.downwind = False
        self.turbine_class = TURBINE_CLASS['I']
        self.drivetrain    = DRIVETRAIN_TYPE['GEARED']

        self.hub_height  = 90.0
        self.hubFraction = 0.025 
        self.bladeLength = 61.5
        self.precone     = 2.5
        self.tilt        = 5.0
        
        # Analysis grid (old r_str)
        eps = 1e-4
        self.r = np.array([eps, 0.00492790457512, 0.00652942887106, 0.00813095316699,
                           0.00983257273154, 0.0114340970275, 0.0130356213234, 0.02222276,
                           0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
                           0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333,
                           0.276686558545, 0.3, 0.333640766319, 0.36666667, 0.400404310407,
                           0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696,
                           0.63333333, 0.667358391486, 0.683573824984, 0.7, 0.73242031601,
                           0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0-eps])
        self.npts = self.r.size
        
        # Blade aero geometry
        raw    = list([row for row in csv.reader(raw)])
        raw_r  = np.array([float(m[1]) for m in raw]) / float(raw[-1][1])
        raw_tw = np.array([float(m[2]) for m in raw])
        raw_c  = np.array([float(m[4]) for m in raw])
        raw_af = [m[-1] for m in raw]

        idx_cylinder = np.argmin(np.abs(raw_r - 11.75/61.6333))
        self.r_cylinder  = raw_r[idx_cylinder]
        self.r_max_chord = raw_r[np.argmax(raw_c)]
        self.setRin()
        
        myspline = Akima(raw_r, raw_tw)
        self.theta, _, _, _ = myspline.interp(self.r_in)
        
        myspline = Akima(raw_r, raw_c)
        self.chord, _, _, _     = myspline.interp(self.r_in)
        self.chord_ref, _, _, _ = myspline.interp(self.r)
        #np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
        #3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
        #                               4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
        #                               4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
        #                               3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
        #                               2.05553464181, 1.82577817774, 1.5860853279, 1.4621])
        # TODO: what's the difference?
        #print np.c_[self.chord_ref, chord]


        self.precurve  = np.zeros(self.chord.shape)
        self.precurveT = 0.0
        self.presweep  = np.zeros(self.chord.shape)

        # Spar cap thickness- linear taper from end of cylinder to tip
        spar_str_orig = np.array([0.05, 0.04974449, 0.04973077, 0.04971704, 0.04970245, 0.04968871,
                                  0.04967496, 0.04959602, 0.04957689, 0.04956310, 0.04921172, 0.04901266,
                                  0.04882344, 0.04851176, 0.04833251, 0.04807698, 0.04773559, 0.04749433,
                                  0.04743920, 0.04738432, 0.04728946, 0.04707145, 0.04666403, 0.04495385,
                                  0.04418236, 0.04219110, 0.04038254, 0.03861577, 0.03649927, 0.03542479,
                                  0.03429437, 0.03194315, 0.02928793, 0.02357558, 0.01826438, 0.01365496,
                                  0.00872504, 0.0061398])
        myspline = Akima(self.r, spar_str_orig)
        self.spar_thickness, _, _, _ = myspline.interp(self.r_in)
        
        te_str_orig = np.array([0.1, 0.10082163, 0.10085572, 0.10088880, 0.10092286, 0.10095388,
                                0.10098389, 0.10113668, 0.10116870, 0.10119056, 0.10140956, 0.10124916,
                                0.10090965, 0.09996015, 0.09919790, 0.09784357, 0.09554915, 0.09204126,
                                0.08974946, 0.08603226, 0.08200398, 0.07760656, 0.07314138, 0.06393682,
                                0.06083395, 0.05341039, 0.04736243, 0.04205857, 0.03645787, 0.03391951,
                                0.03146610, 0.02706400, 0.02308701, 0.01642426, 0.01190607, 0.00896844,
                                0.00663243, 0.00569])
        myspline = Akima(self.r, te_str_orig)
        self.te_thickness, _, _, _ = myspline.interp(self.r_in)
        
        self.le_location = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
                                     0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                                     0.4, 0.4, 0.4, 0.4])

        afpath = self.getAeroPath()
        self.airfoils = ['']*self.npts
        for k in range(self.npts):
            idx = np.argmin( np.abs(raw_r - self.r[k]) )
            self.airfoils[k] = os.path.join(afpath, raw_af[idx])
            if (self.r[k] <= self.r_cylinder) and raw_af[idx].find('Cylinder') < 0:
                self.airfoils[k] = os.path.join(afpath, 'Cylinder2.dat')
         
        # Layup info
        self.sector_idx_strain_spar = np.array([2]*self.npts)
        self.sector_idx_strain_te = np.array([3]*self.npts)
        
        self.web1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4114, 0.4102, 0.4094, 0.3876,
                              0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104,
                              0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731,
                              0.2664, 0.2607, 0.2562, 0.1886, np.nan])
        self.web2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.5886, 0.5868, 0.5854, 0.5508,
                              0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896,
                              0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269,
                              0.5336, 0.5393, 0.5438, 0.6114, np.nan])
        self.web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        self.web4 = np.nan * np.ones(self.web3.shape)

        
        # Control
        self.control_Vin      = 3.0
        self.control_Vout     = 25.0
        self.control_minOmega = 6.9
        self.control_maxOmega = 12.1
        self.control_tsr      = 80.0 / 11.4
        self.control_pitch    = 0.0
        
        
class DTU10MW(ReferenceBlade):
    def __init__(self):
        super(DTU10MW, self).__init__()

        self.name   = '10MW'
        self.rating = 10e6
        self.nBlades  = 3
        self.downwind = False
        self.turbine_class = TURBINE_CLASS['I']
        self.drivetrain    = DRIVETRAIN_TYPE['GEARED']

        self.hub_height  = 119.0
        self.bladeLength = 0.5 * (198.0 - 4.6)
        self.hubFraction = 0.5*4.6 / self.bladeLength
        self.precone     = -4.0
        self.tilt        = 6.0

        # DTU 10MW BLADE PROPS
        #Eta,Chord,Twist,Rel.Thick.,Abs.Thick.,PreBend,Sweep,PitchAx.
        #[-],[mm],[deg],[%],[mm],[mm],[mm],[%]
        raw = StringIO("""0.0000,4600.000,14.500,100.000,4.600,-0.000,0.000,0.500
        0.0176,4601.556,14.037,99.739,4.590,-7.669,0.000,0.500
        0.0313,4601.651,13.737,98.192,4.518,-14.693,0.000,0.500
        0.0420,4609.049,13.545,95.682,4.410,-20.900,0.000,0.498
        0.0507,4636.155,13.413,92.513,4.289,-26.516,0.000,0.489
        0.0579,4667.782,13.312,89.240,4.166,-31.484,0.000,0.480
        0.0642,4698.278,13.223,86.130,4.047,-35.920,0.000,0.473
        0.0700,4729.304,13.139,83.061,3.928,-40.113,0.000,0.465
        0.0758,4763.632,13.056,79.871,3.805,-44.343,0.000,0.459
        0.0821,4807.427,12.969,76.338,3.670,-48.998,0.000,0.453
        0.0893,4869.795,12.874,72.208,3.516,-54.497,0.000,0.445
        0.0980,4953.764,12.752,67.393,3.338,-61.250,0.000,0.435
        0.1087,5063.527,12.565,61.870,3.133,-69.747,0.000,0.423
        0.1224,5218.259,12.223,55.573,2.900,-80.713,0.000,0.408
        0.1400,5444.001,11.600,48.640,2.648,-95.183,0.000,0.389
        0.1610,5688.237,10.674,42.379,2.411,-112.440,0.000,0.369
        0.1841,5937.831,9.329,37.456,2.224,-131.381,0.000,0.351
        0.2093,5987.655,7.980,34.759,2.081,-152.316,0.000,0.339
        0.2366,5990.629,6.724,33.223,1.990,-175.303,0.000,0.331
        0.2660,5896.920,5.879,32.682,1.927,-200.154,0.000,0.331
        0.2973,5704.650,5.222,32.580,1.859,-227.286,0.000,0.332
        0.3305,5463.845,4.668,32.606,1.782,-257.214,0.000,0.341
        0.3653,5176.618,4.131,32.547,1.685,-292.010,0.000,0.353
        0.4015,4857.565,3.614,32.367,1.572,-334.013,0.000,0.368
        0.4387,4520.899,3.085,32.063,1.450,-385.250,0.000,0.387
        0.4766,4179.799,2.496,31.474,1.316,-447.911,0.000,0.407
        0.5148,3842.977,1.843,30.700,1.180,-526.833,0.000,0.427
        0.5529,3519.719,1.161,29.781,1.048,-625.007,0.000,0.447
        0.5905,3217.849,0.472,28.768,0.926,-744.099,0.000,0.465
        0.6273,2942.296,-0.197,27.680,0.814,-887.576,0.000,0.481
        0.6628,2695.475,-0.821,26.556,0.716,-1057.588,0.000,0.495
        0.6969,2477.568,-1.382,25.426,0.630,-1253.153,0.000,0.506
        0.7293,2287.113,-1.870,24.340,0.557,-1474.395,0.000,0.515
        0.7597,2122.247,-2.277,23.397,0.497,-1720.593,0.000,0.522
        0.7882,1980.577,-2.603,22.828,0.452,-1987.436,0.000,0.525
        0.8145,1859.108,-2.850,22.755,0.423,-2273.422,0.000,0.526
        0.8387,1754.739,-3.024,22.673,0.398,-2573.827,0.000,0.526
        0.8608,1665.081,-3.125,22.533,0.375,-2884.584,0.000,0.523
        0.8809,1586.370,-3.160,22.351,0.355,-3202.247,0.000,0.519
        0.8990,1511.990,-3.130,22.145,0.335,-3522.496,0.000,0.513
        0.9153,1434.483,-3.038,21.930,0.315,-3841.371,0.000,0.507
        0.9299,1350.941,-2.883,21.722,0.293,-4154.921,0.000,0.499
        0.9429,1260.917,-2.669,21.533,0.272,-4460.302,0.000,0.492
        0.9544,1166.025,-2.399,21.372,0.249,-4755.279,0.000,0.485
        0.9646,1062.491,-2.085,21.241,0.226,-5037.057,0.000,0.477
        0.9736,957.120,-1.727,21.143,0.202,-5304.189,0.000,0.470
        0.9816,862.535,-1.340,21.073,0.182,-5555.265,0.000,0.464
        0.9885,759.755,-0.927,21.030,0.160,-5789.509,0.000,0.457
        0.9946,540.567,-0.483,21.007,0.114,-6006.941,0.000,0.452
        1.0000,96.200,-0.037,21.000,0.020,-6206.217,0.000,0.446""")

        eps = 1e-4
        self.r = np.array([eps, 0.0204081632653, 0.0408163265306, 0.0612244897959, 0.0816326530612, 0.102040816327,
                           0.122448979592, 0.142857142857, 0.163265306122, 0.183673469388, 0.204081632653, 0.224489795918,
                           0.244897959184, 0.265306122449, 0.285714285714, 0.30612244898, 0.326530612245, 0.34693877551,
                           0.367346938776, 0.387755102041, 0.408163265306, 0.428571428571, 0.448979591837, 0.469387755102,
                           0.489795918367, 0.510204081633, 0.530612244898, 0.551020408163, 0.571428571429, 0.591836734694,
                           0.612244897959, 0.632653061224, 0.65306122449, 0.673469387755, 0.69387755102, 0.714285714286,
                           0.734693877551, 0.755102040816, 0.775510204082, 0.795918367347, 0.816326530612, 0.836734693878,
                           0.857142857143, 0.877551020408, 0.897959183673, 0.918367346939, 0.938775510204, 0.959183673469,
                           0.979591836735, 1.0-eps])
        self.npts = self.r.size
        
        raw     = np.loadtxt(raw, delimiter=',')
        raw_r   = raw[:,0]
        raw_c   = raw[:,1] * 1e-3
        raw_tw  = raw[:,2]
        raw_th  = raw[:,3]
        raw_pre = raw[:,5] * 1e-3
        raw_sw  = raw[:,6] * 1e-3

        idx_cylinder = 10
        self.r_cylinder  = raw_r[idx_cylinder]
        self.r_max_chord = raw_r[np.argmax(raw_c)]
        self.setRin()
        
        myspline = Akima(raw_r, raw_tw)
        self.theta, _, _, _ = myspline.interp(self.r_in)
        
        myspline = Akima(raw_r, raw_c)
        self.chord, _, _, _     = myspline.interp(self.r_in)
        self.chord_ref, _, _, _ = myspline.interp(self.r)
        #self.chord_ref = np.array([5.38, 5.3800643553, 5.38031711143, 5.38780280252, 5.40677951126, 5.48505840079, 5.59326574185,
        #                               5.73141566075, 5.86843135503, 5.99999190341, 6.09904231251, 6.17116486928, 6.19400935481,
        #                               6.20302411962, 6.18309136227, 6.14171800022, 6.07759639166, 5.99796748755, 5.90179584286,
        #                               5.79486385463, 5.67757358533, 5.55267710905, 5.42166648998, 5.28499407124, 5.14373698212,
        #                               4.99871795233, 4.85053210974, 4.70010140244, 4.54830998864, 4.39588552607, 4.24360835244,
        #                               4.09216987991, 3.94187021346, 3.79298624168, 3.64578973915, 3.50055778142, 3.3575837574,
        #                               3.21717725827, 3.07962549378, 2.94515242311, 2.81389912887, 2.68570603823, 2.56087639104,
        #                               2.43927605964, 2.31610551275, 2.18016451487, 2.01720583646, 1.81389861412, 1.50584435653, 0.6])

        myspline = Akima(raw_r, raw_pre)
        self.precurve, _, _, _ = myspline.interp(self.r_in)
        self.precurveT = 0.0

        myspline = Akima(raw_r, raw_sw)
        self.presweep, _, _, _ = myspline.interp(self.r_in)

        # Spar cap thickness- linear taper from end of cylinder to tip
        self.spar_thickness = np.linspace(1.44, 0.53, NINPUT)
        self.te_thickness   = np.linspace(0.8, 0.2, NINPUT)
        
        afpath = self.getAeroPath()
        myspline = Akima(raw_r, raw_th)
        thickness, _, _, _ = myspline.interp(self.r)
        thickness = np.minimum(100.0, thickness)
        #af_thicknesses  = np.array([21.1, 24.1, 27.0, 30.1, 33.0, 36.0, 48.0, 60.0, 72.0, 100.0])
        af_thicknesses  = np.array([24.1, 30.1, 36.0, 48.0, 60.0, 100.0])
        self.airfoils = ['']*self.npts
        for k in range(self.npts):
            idx_thick       = np.where(thickness[k] <= af_thicknesses)[0]
            if idx_thick.size > 0 and idx_thick[0] < af_thicknesses.size-1:
                prefix   = 'FFA_W3_'
                thickStr = str(np.int(10*af_thicknesses[idx_thick[0]]))
            else:
                prefix   = 'Cylinder'
                thickStr = ''
            self.airfoils[k] = os.path.join(afpath, prefix + thickStr + '.dat')

        # Structural analysis inputs
        self.le_location = np.array([0.5, 0.499998945239, 0.499990630963, 0.499384561429, 0.497733369567, 0.489487054775,
                                     0.476975219349, 0.458484322766, 0.440125810719, 0.422714559863, 0.407975209714,
                                     0.395449769723, 0.385287280879, 0.376924554763, 0.370088311651, 0.364592902698,
                                     0.3602205136, 0.356780489919, 0.354039530035, 0.351590005932, 0.350233815248, 0.350012355763,
                                     0.349988281626, 0.350000251201, 0.350002561185, 0.350001421895, 0.349997012891, 0.350001029096,
                                     0.350000632518, 0.349999297634, 0.350000264157, 0.350000005654, 0.349999978357, 0.349999995158,
                                     0.350000006591, 0.349999999186, 0.349999998202, 0.350000000551, 0.350000000029, 0.349999999931,
                                     0.35000000004, 0.350000000001, 0.35, 0.350000000001, 0.349999999999, 0.35, 0.35, 0.35, 0.35, 0.35])

        # TODO: These are just guesses for now
        self.sector_idx_strain_spar = np.array([4]*self.npts)
        self.sector_idx_strain_te = np.array([6]*self.npts)

        self.web1 = np.array([0.446529203227, 0.446642686219, 0.447230977047, 0.449423527671, 0.451384667298, 0.45166085909,
                              0.445821859041, 0.433601957075, 0.414203341702, 0.391111637325, 0.367038887871, 0.344148340044,
                              0.32264263023, 0.303040717673, 0.285780556269, 0.271339581072, 0.261077569528, 0.254987877709,
                              0.250499030835, 0.246801903789, 0.243793928448, 0.242362866767, 0.241169996298, 0.240114471242,
                              0.239138338743, 0.238211240433, 0.237380060299, 0.236625908889, 0.235947619537, 0.235375269498,
                              0.234910524166, 0.234573714458, 0.23437656803, 0.234323591937, 0.234429396513, 0.23469408391,
                              0.235090916602, 0.235639910948, 0.236359205424, 0.237292044985, 0.238468772012, 0.239912928964,
                              0.241676539436, 0.24378663077, 0.246041897214, 0.247824545238, 0.248212620456, 0.247666927859,
                              0.246627910571, 0.154148714864])
        self.web2 = np.array([0.579105947595, 0.579342815032, 0.580624719333, 0.585617777398, 0.5905335998, 0.592757384044,
                              0.587897774807, 0.576668436742, 0.557200875669, 0.532380978251, 0.505531782719, 0.479744314701,
                              0.456216340946, 0.43494475968, 0.416533496674, 0.40143144474, 0.390766314857, 0.384449528527,
                              0.379891695643, 0.376232568599, 0.373403485013, 0.372223966271, 0.371379269973, 0.370759395854,
                              0.370295535294, 0.36996224587, 0.369789327231, 0.36975863937, 0.369855492121, 0.370090120509,
                              0.370456638362, 0.370955636744, 0.371593427472, 0.37236755247, 0.373282787035, 0.374330521266,
                              0.37548259513, 0.376740794894, 0.378108493317, 0.379600043497, 0.381225881051, 0.382999965574,
                              0.384937055128, 0.387054062586, 0.389406784803, 0.392151134201, 0.395577451592, 0.398541075929,
                              0.396140574221, 0.377290889301])
        self.web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.883963623418, 0.882804861283,
                              0.881764724203, 0.880854972132, 0.880063593261, 0.879378185003, 0.878797670205, 0.878308574021,
                              0.87788500707, 0.877514541845, 0.877179551402, 0.876879055, 0.876610936666, 0.876374235852,
                              0.876166483426, 0.875986765133, 0.875832594685, 0.875702298474, 0.875595140019, 0.875508315318,
                              0.8754409845, 0.875392657588, 0.875361122776, 0.875345560917, 0.875342086256, 0.875341086148,
                              0.875340761092, 0.875339510238, 0.875338477803, 0.875337374186, 0.875336204651, 0.87533496247,
                              0.875333692251, 0.875332490358, 0.875331189455, 0.875329175403, 0.875324968372, 0.875312750097,
                              0.875329970752])
        self.web4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        
        # Control
        self.control_Vin      = 4.0
        self.control_Vout     = 25.0
        self.control_minOmega = 6.0
        self.control_maxOmega = 90.0 / self.bladeLength * (60.0/(2.0*np.pi))
        self.control_tsr      = 10.58
        self.control_pitch    = 0.0
        

class BladeGeometry(Component):
    
    def __init__(self, RefBlade):
        super(BladeGeometry, self).__init__()

        assert isinstance(RefBlade, ReferenceBlade), 'Must pass in either NREL5MW or DTU10MW Reference Blade instance'
        self.refBlade = RefBlade
        npts = self.refBlade.npts
        
        # variables
        self.add_param('bladeLength', val=0.0, units='m', desc='blade length (if not precurved or swept) otherwise length of blade before curvature')
        self.add_param('r_max_chord', val=0.0, desc='location of max chord on unit radius')
        self.add_param('chord_in', val=np.zeros(NINPUT), units='m', desc='chord at control points')  # defined at hub, then at linearly spaced locations from r_max_chord to tip
        self.add_param('theta_in', val=np.zeros(NINPUT), units='deg', desc='twist at control points')  # defined at linearly spaced locations from r[idx_cylinder] to tip
        self.add_param('precurve_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_param('presweep_in', val=np.zeros(NINPUT), units='m', desc='precurve at control points')  # defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        self.add_param('sparT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of spar cap that linearly vary from non-cylinder position to tip')
        self.add_param('teT_in', val=np.zeros(NINPUT), units='m', desc='thickness values of trailing edge panels that linearly vary from non-cylinder position to tip')

        # parameters
        self.add_param('hubFraction', val=0.0, desc='hub location as fraction of radius')

        # Blade geometry outputs
        self.add_output('Rhub', val=0.0, units='m', desc='dimensional radius of hub')
        self.add_output('Rtip', val=0.0, units='m', desc='dimensional radius of tip')
        self.add_output('r_pts', val=np.zeros(npts), units='m', desc='dimensional aerodynamic grid')
        self.add_output('r_in', val=np.zeros(NINPUT), units='m', desc='Spline control points for inputs')
        self.add_output('max_chord', val=0.0, units='m', desc='maximum chord length')
        self.add_output('chord', val=np.zeros(npts), units='m', desc='chord at airfoil locations')
        self.add_output('theta', val=np.zeros(npts), units='deg', desc='twist at airfoil locations')
        self.add_output('precurve', val=np.zeros(npts), units='m', desc='precurve at airfoil locations')
        self.add_output('presweep', val=np.zeros(npts), units='m', desc='presweep at structural locations')
        self.add_output('sparT', val=np.zeros(npts), units='m', desc='dimensional spar cap thickness distribution')
        self.add_output('teT', val=np.zeros(npts), units='m', desc='dimensional trailing-edge panel thickness distribution')

        self.add_output('hub_diameter', val=0.0, units='m')
        
        self.add_output('airfoil_files', val=[], desc='Spanwise coordinates for aerodynamic analysis', pass_by_obj=True)
        self.add_output('le_location', val=np.zeros(npts), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_output('chord_ref', val=np.zeros(npts), desc='Chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c)')

        # Blade layup outputs
        self.add_output('materials', val=np.zeros(npts), desc='material properties of composite materials', pass_by_obj=True)
        
        self.add_output('upperCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_output('lowerCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_output('websCS', val=np.zeros(npts), desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)
        self.add_output('profile', val=np.zeros(npts), desc='list of CompositeSection profiles', pass_by_obj=True)
        
        self.add_output('sector_idx_strain_spar', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for spar (PreComp definition of sector)', pass_by_obj=True)
        self.add_output('sector_idx_strain_te', val=np.zeros(npts, dtype=np.int_), desc='Index of sector for trailing edge (PreComp definition of sector)', pass_by_obj=True)

        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):

        Rhub = params['hubFraction'] * params['bladeLength']
        Rtip = Rhub + params['bladeLength']

        # make dimensional and evaluate splines
        unknowns['Rhub']     = Rhub
        unknowns['Rtip']     = Rtip
        unknowns['hub_diameter'] = 2.0*Rhub
        unknowns['r_pts']    = Rhub + (Rtip-Rhub)*self.refBlade.r
        unknowns['r_in']     = Rhub + (Rtip-Rhub)*np.r_[0.0, self.refBlade.r_cylinder, np.linspace(params['r_max_chord'], 1.0, NINPUT-2)]

        # Although the inputs get mirrored to outputs, this is still necessary so that the user can designate the inputs as design variables
        myspline = Akima(unknowns['r_in'], params['chord_in'])
        unknowns['max_chord'], _, _, _ = myspline.interp(Rhub + (Rtip-Rhub)*params['r_max_chord'])
        unknowns['chord'], _, _, _ = myspline.interp(unknowns['r_pts'])

        myspline = Akima(unknowns['r_in'], params['theta_in'])
        unknowns['theta'], _, _, _ = myspline.interp(unknowns['r_pts'])

        myspline = Akima(unknowns['r_in'], params['precurve_in'])
        unknowns['precurve'], _, _, _ = myspline.interp(unknowns['r_pts'])

        myspline = Akima(unknowns['r_in'], params['presweep_in'])
        unknowns['presweep'], _, _, _ = myspline.interp(unknowns['r_pts'])

        myspline = Akima(unknowns['r_in'], params['sparT_in'])
        unknowns['sparT'], _, _, _ = myspline.interp(unknowns['r_pts'])
        
        myspline = Akima(unknowns['r_in'], params['teT_in'])
        unknowns['teT'], _, _, _ = myspline.interp(unknowns['r_pts'])
        
        # Setup paths
        strucpath = self.refBlade.getStructPath()
        materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(strucpath, 'materials.inp'))

        # Now compute some layup properties, independent of which turbine it is
        npts = self.refBlade.npts
        upperCS = [0]*npts
        lowerCS = [0]*npts
        websCS  = [0]*npts
        profile = [0]*npts

        for i in range(npts):
            webLoc = []
            if not np.isnan(self.refBlade.web1[i]): webLoc.append(self.refBlade.web1[i])
            if not np.isnan(self.refBlade.web2[i]): webLoc.append(self.refBlade.web2[i])
            if not np.isnan(self.refBlade.web3[i]): webLoc.append(self.refBlade.web3[i])
            if not np.isnan(self.refBlade.web4[i]): webLoc.append(self.refBlade.web4[i])

            istr = str(i+1) if self.refBlade.name == '5MW' else str(i)
            upperCS[i], lowerCS[i], websCS[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(strucpath, 'layup_' + istr + '.inp'), webLoc, materials)
            profile[i] = Profile.initFromPreCompFile(os.path.join(strucpath, 'shape_' + istr + '.inp'))

        # Assign outputs
        unknowns['airfoil_files']          = self.refBlade.airfoils
        unknowns['le_location']            = self.refBlade.le_location
        unknowns['upperCS']                = upperCS
        unknowns['lowerCS']                = lowerCS
        unknowns['websCS']                 = websCS
        unknowns['profile']                = profile
        unknowns['chord_ref']              = self.refBlade.chord_ref
        unknowns['sector_idx_strain_spar'] = self.refBlade.sector_idx_strain_spar
        unknowns['sector_idx_strain_te']   = self.refBlade.sector_idx_strain_te
        unknowns['materials']              = materials
        
        
class Location(Component):
    def __init__(self):
        super(Location, self).__init__()
        self.add_param('hub_height', val=0.0, units='m', desc='Tower top hub height')
        self.add_output('wind_zvec', val=np.zeros(1), units='m', desc='Tower top hub height as vector')
        self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['wind_zvec'] = np.array([ params['hub_height'] ])


        
class TurbineClass(Component):
    def __init__(self):
        super(TurbineClass, self).__init__()
        # parameters
        self.add_param('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class', pass_by_obj=True)

        # outputs should be constant
        self.add_output('V_mean', shape=1, units='m/s', desc='IEC mean wind speed for Rayleigh distribution')
        self.add_output('V_extreme', shape=1, units='m/s', desc='IEC extreme wind speed at hub height')
        self.add_output('V_extreme_full', shape=2, units='m/s', desc='IEC extreme wind speed at hub height')
        
	self.deriv_options['type'] = 'fd'
        self.deriv_options['step_calc'] = 'relative'

    def solve_nonlinear(self, params, unknowns, resids):

        self.turbine_class = params['turbine_class']

        if self.turbine_class == TURBINE_CLASS['I']:
            Vref = 50.0
        elif self.turbine_class == TURBINE_CLASS['II']:
            Vref = 42.5
        elif self.turbine_class == TURBINE_CLASS['III']:
            Vref = 37.5
        elif self.turbine_class == TURBINE_CLASS['IV']:
            Vref = 30.0

        unknowns['V_mean'] = 0.2*Vref
        unknowns['V_extreme'] = 1.4*Vref
        unknowns['V_extreme_full'][0] = 1.4*Vref # for extreme cases TODO: check if other way to do
        unknowns['V_extreme_full'][1] = 1.4*Vref

        

class RotorGeometry(Group):
    def __init__(self, RefBlade):
        super(RotorGeometry, self).__init__()
        """rotor model"""
        assert isinstance(RefBlade, ReferenceBlade), 'Must pass in either NREL5MW or DTU10MW Reference Blade instance'

        self.add('bladeLength', IndepVarComp('bladeLength', 0.0, units='m'), promotes=['*'])
        self.add('hubFraction', IndepVarComp('hubFraction', 0.0), promotes=['*'])
        self.add('r_max_chord', IndepVarComp('r_max_chord', 0.0), promotes=['*'])
        self.add('chord_in', IndepVarComp('chord_in', np.zeros(NINPUT),units='m'), promotes=['*'])
        self.add('theta_in', IndepVarComp('theta_in', np.zeros(NINPUT), units='deg'), promotes=['*'])
        self.add('precurve_in', IndepVarComp('precurve_in', np.zeros(NINPUT), units='m'), promotes=['*'])
        self.add('precurve_tip', IndepVarComp('precurve_tip', 0.0, units='m'), promotes=['*'])
        self.add('presweep_in', IndepVarComp('presweep_in', np.zeros(NINPUT), units='m'), promotes=['*'])
        self.add('presweep_tip', IndepVarComp('presweep_tip', 0.0, units='m'), promotes=['*'])
        self.add('precone', IndepVarComp('precone', 0.0, units='deg'), promotes=['*'])
        self.add('tilt', IndepVarComp('tilt', 0.0, units='deg'), promotes=['*'])
        self.add('yaw', IndepVarComp('yaw', 0.0, units='deg'), promotes=['*'])
        self.add('nBlades', IndepVarComp('nBlades', 3, pass_by_obj=True), promotes=['*'])
        self.add('turbine_class', IndepVarComp('turbine_class', val=TURBINE_CLASS['I'], desc='IEC turbine class', pass_by_obj=True), promotes=['*'])
        
        # --- composite sections ---
        self.add('sparT_in', IndepVarComp('sparT_in', val=np.zeros(NINPUT), units='m', desc='spar cap thickness parameters'), promotes=['*'])
        self.add('teT_in', IndepVarComp('teT_in', val=np.zeros(NINPUT), units='m', desc='trailing-edge thickness parameters'), promotes=['*'])
        
        # --- Rotor Definition ---
        self.add('loc', Location(), promotes=['*'])
        self.add('turbineclass', TurbineClass())
        #self.add('spline0', BladeGeometry(RefBlade))
        self.add('spline', BladeGeometry(RefBlade), promotes=['*'])
        self.add('geom', CCBladeGeometry())

        # connections to turbineclass
        self.connect('turbine_class', 'turbineclass.turbine_class')

        # connections to spline0
        #self.connect('r_max_chord', 'spline0.r_max_chord')
        #self.connect('chord_in', 'spline0.chord_in')
        #self.connect('theta_in', 'spline0.theta_in')
        #self.connect('precurve_in', 'spline0.precurve_in')
        #self.connect('presweep_in', 'spline0.presweep_in')
        #self.connect('bladeLength', 'spline0.bladeLength')
        #self.connect('hubFraction', 'spline0.hubFraction')
        #self.connect('sparT_in', 'spline0.sparT_in')
        #self.connect('teT_in', 'spline0.teT_in')

        # connections to spline
        #self.connect('r_max_chord', 'spline.r_max_chord')
        #self.connect('chord_in', 'spline.chord_in')
        #self.connect('theta_in', 'spline.theta_in')
        #self.connect('precurve_in', 'spline.precurve_in')
        #self.connect('presweep_in', 'spline.presweep_in')
        #self.connect('bladeLength', 'spline.bladeLength')
        #self.connect('hubFraction', 'spline.hubFraction')
        #self.connect('sparT_in', 'spline.sparT_in')
        #self.connect('teT_in', 'spline.teT_in')

        # connections to geom
        self.connect('Rtip', 'geom.Rtip')
        self.connect('precone', 'geom.precone')
        self.connect('precurve_tip', 'geom.precurveTip')
