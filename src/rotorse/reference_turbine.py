import os
import numpy as np
from openmdao.api import Component

from rotorse import REFERENCE_TURBINE
from precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp

NSTR = 38 #50
NAERO = 17 #48

class ReferenceTurbine(Component):
    
    def __init__(self):
        super(ReferenceTurbine, self).__init__()

        self.add_param('reference_turbine', val=REFERENCE_TURBINE['5MW'], desc='NREL 5MW or DTU 10MW reference turbine', pass_by_obj=True)

        self.add_output('r_aero', val=np.zeros(NAERO), desc='Spanwise coordinates for aerodynamic analysis')
        self.add_output('r_str', val=np.zeros(NSTR), desc='Spanwise coordinates for aerodynamic analysis')
        
        self.add_output('airfoil_files', val=[], desc='Spanwise coordinates for aerodynamic analysis', pass_by_obj=True)
        self.add_output('le_location', val=np.zeros(NSTR), desc='Leading-edge positions from a reference blade axis (usually blade pitch axis). Locations are normalized by the local chord length. Positive in -x direction for airfoil-aligned coordinate system')
        self.add_output('chord_str_ref', val=np.zeros(NSTR), desc='Chord distribution for reference section, thickness of structural layup scaled with reference thickness (fixed t/c)')
        self.add_output('materials', shape=NSTR, desc='material properties of composite materials', pass_by_obj=True)
        
        self.add_output('upperCS', shape=NSTR, desc='list of CompositeSection objections defining the properties for upper surface', pass_by_obj=True)
        self.add_output('lowerCS', shape=NSTR, desc='list of CompositeSection objections defining the properties for lower surface', pass_by_obj=True)
        self.add_output('websCS', shape=NSTR, desc='list of CompositeSection objections defining the properties for shear webs', pass_by_obj=True)
        self.add_output('profile', shape=NSTR, desc='list of CompositeSection profiles', pass_by_obj=True)
        
        self.add_output('sector_idx_strain_spar', val=np.zeros(NSTR, dtype=np.int_), desc='Index of sector for spar (PreComp definition of sector)', pass_by_obj=True)
        self.add_output('sector_idx_strain_te', val=np.zeros(NSTR, dtype=np.int_), desc='Index of sector for trailing edge (PreComp definition of sector)', pass_by_obj=True)
        
    def solve_nonlinear(self, params, unknowns, resids):
        flag5 = self.params['reference_turbine'] == REFERENCE_TURBINE['5MW']
        refStr = REFERENCE_TURBINE[ self.params['reference_turbine'] ]

        # Setup paths
        aeropath  = os.path.join(os.path.dirname(os.path.realpath(__file__)), refStr+'_AFFiles')
        strucpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), refStr+'_PreCompFiles')
        materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(strucpath, 'materials.inp'))

        # Store variables for the different reference turbines
        if flag5:
            #------- NREL 5MW DEFNITION -----------#
            # Aero analysis inputs
            r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
	                       0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
	                       0.97777724])  # (Array): new aerodynamic grid on unit radius

            airfoil_types = [0]*8
            airfoil_types[0] = os.path.join(aeropath, 'Cylinder1.dat')
            airfoil_types[1] = os.path.join(aeropath, 'Cylinder2.dat')
            airfoil_types[2] = os.path.join(aeropath, 'DU40_A17.dat')
            airfoil_types[3] = os.path.join(aeropath, 'DU35_A17.dat')
            airfoil_types[4] = os.path.join(aeropath, 'DU30_A17.dat')
            airfoil_types[5] = os.path.join(aeropath, 'DU25_A17.dat')
            airfoil_types[6] = os.path.join(aeropath, 'DU21_A17.dat')
            airfoil_types[7] = os.path.join(aeropath, 'NACA64_A17.dat')
            af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]
            airfoil_files = [airfoil_types[m] for m in af_idx]

            # Structural analysis inputs
            r_str = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699,
                              0.00983257273154, 0.0114340970275, 0.0130356213234, 0.02222276,
                              0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
                              0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333,
                              0.276686558545, 0.3, 0.333640766319, 0.36666667, 0.400404310407,
                              0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696,
                              0.63333333, 0.667358391486, 0.683573824984, 0.7, 0.73242031601,
                              0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0])

            le_location = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
                              0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
                              0.4, 0.4, 0.4, 0.4])
            sector_idx_strain_spar = np.array([2]*NSTR)
            sector_idx_strain_te = np.array([3]*NSTR)
            chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
                                      3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
                                      4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
                                      4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
                                      3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
                                      2.05553464181, 1.82577817774, 1.5860853279, 1.4621])

            web1 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.4114, 0.4102, 0.4094, 0.3876,
                             0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104,
                             0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731,
                             0.2664, 0.2607, 0.2562, 0.1886, np.nan])
            web2 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.5886, 0.5868, 0.5854, 0.5508,
                             0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896,
                             0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269,
                             0.5336, 0.5393, 0.5438, 0.6114, np.nan])
            web3 = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                             np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            web4 = np.nan * np.ones(web3.shape)

        else:
            #------- DTU 10MW DEFNITION -----------#
            # Aero analysis inputs
            r_aero = np.array([0.0388344625662, 0.0775639928626, 0.116084789688, 0.154295288168, 0.192097214403, 0.229396566649,
                               0.266104502965, 0.302138118728, 0.337421101321, 0.37188425356, 0.405465881644, 0.438112047586,
                               0.469776689866, 0.500421619416, 0.530016400857, 0.558538131066, 0.585971128678, 0.612306549063,
                               0.637541939584, 0.661680749783, 0.684731810463, 0.706708794667, 0.727629672281, 0.747516168597,
                               0.766393235642, 0.784288543547, 0.801231997762, 0.817255286497, 0.832391461456, 0.846674553822,
                               0.860139226394, 0.872820461935, 0.884753287105, 0.895972530746, 0.906512614886, 0.916407376498,
                               0.925689917824, 0.93439248297, 0.9425463584, 0.950181794981, 0.957327949275, 0.964012841862,
                               0.970263330598, 0.976105096854, 0.981562642901, 0.986659298795, 0.991417237237, 0.995857495039])

            # Structural analysis inputs
            r_str = np.array([0.0, 0.0204081632653, 0.0408163265306, 0.0612244897959, 0.0816326530612, 0.102040816327,
                              0.122448979592, 0.142857142857, 0.163265306122, 0.183673469388, 0.204081632653, 0.224489795918,
                              0.244897959184, 0.265306122449, 0.285714285714, 0.30612244898, 0.326530612245, 0.34693877551,
                              0.367346938776, 0.387755102041, 0.408163265306, 0.428571428571, 0.448979591837, 0.469387755102,
                              0.489795918367, 0.510204081633, 0.530612244898, 0.551020408163, 0.571428571429, 0.591836734694,
                              0.612244897959, 0.632653061224, 0.65306122449, 0.673469387755, 0.69387755102, 0.714285714286,
                              0.734693877551, 0.755102040816, 0.775510204082, 0.795918367347, 0.816326530612, 0.836734693878,
                              0.857142857143, 0.877551020408, 0.897959183673, 0.918367346939, 0.938775510204, 0.959183673469,
                              0.979591836735, 1.0])
            
            le_location = np.array([0.5, 0.499998945239, 0.499990630963, 0.499384561429, 0.497733369567, 0.489487054775,
                              0.476975219349, 0.458484322766, 0.440125810719, 0.422714559863, 0.407975209714,
                              0.395449769723, 0.385287280879, 0.376924554763, 0.370088311651, 0.364592902698,
                              0.3602205136, 0.356780489919, 0.354039530035, 0.351590005932, 0.350233815248, 0.350012355763,
                              0.349988281626, 0.350000251201, 0.350002561185, 0.350001421895, 0.349997012891, 0.350001029096,
                              0.350000632518, 0.349999297634, 0.350000264157, 0.350000005654, 0.349999978357, 0.349999995158,
                              0.350000006591, 0.349999999186, 0.349999998202, 0.350000000551, 0.350000000029, 0.349999999931,
                              0.35000000004, 0.350000000001, 0.35, 0.350000000001, 0.349999999999, 0.35, 0.35, 0.35, 0.35, 0.35])
            # TODO!!!
            sector_idx_strain_spar = np.array([2]*nstr)
            sector_idx_strain_te = np.array([3]*nstr)
            chord_str_ref = np.array([0.0622930319802, 0.0622937771264, 0.0622967036963, 0.0623833777473, 0.0626031020455,
                                      0.0635094643817, 0.064762357199, 0.0663619440608, 0.0679483981547, 0.0694716891301,
                                      0.0706185572159, 0.07145363765, 0.0717181455065, 0.071822524137, 0.0715917301053,
                                      0.0711126832344, 0.0703702428231, 0.0694482491669, 0.0683347132304, 0.0670965872523,
                                      0.0657385265652, 0.0642923964181, 0.0627754728711, 0.0611929934377, 0.0595574298002,
                                      0.0578783080417, 0.056162518928, 0.0544207373554, 0.0526632006651, 0.050898334137,
                                      0.049135173013, 0.0473817228992, 0.0456414586001, 0.0439175861066, 0.0422132521959,
                                      0.0405316650235, 0.0388762216312, 0.0372505066608, 0.035657845608, 0.0341008316133,
                                      0.03258109822, 0.0310967977934, 0.0296514414358, 0.0282434761323, 0.0268173298838,
                                      0.0252433193023, 0.0233564809816, 0.0210024617803, 0.017435615364, 0.00694717828775])
            
            web1 = np.array([0.446529203227, 0.446642686219, 0.447230977047, 0.449423527671, 0.451384667298, 0.45166085909,
                             0.445821859041, 0.433601957075, 0.414203341702, 0.391111637325, 0.367038887871, 0.344148340044,
                             0.32264263023, 0.303040717673, 0.285780556269, 0.271339581072, 0.261077569528, 0.254987877709,
                             0.250499030835, 0.246801903789, 0.243793928448, 0.242362866767, 0.241169996298, 0.240114471242,
                             0.239138338743, 0.238211240433, 0.237380060299, 0.236625908889, 0.235947619537, 0.235375269498,
                             0.234910524166, 0.234573714458, 0.23437656803, 0.234323591937, 0.234429396513, 0.23469408391,
                             0.235090916602, 0.235639910948, 0.236359205424, 0.237292044985, 0.238468772012, 0.239912928964,
                             0.241676539436, 0.24378663077, 0.246041897214, 0.247824545238, 0.248212620456, 0.247666927859,
                             0.246627910571, 0.154148714864])
            web2 = np.array([0.579105947595, 0.579342815032, 0.580624719333, 0.585617777398, 0.5905335998, 0.592757384044,
                             0.587897774807, 0.576668436742, 0.557200875669, 0.532380978251, 0.505531782719, 0.479744314701,
                             0.456216340946, 0.43494475968, 0.416533496674, 0.40143144474, 0.390766314857, 0.384449528527,
                             0.379891695643, 0.376232568599, 0.373403485013, 0.372223966271, 0.371379269973, 0.370759395854,
                             0.370295535294, 0.36996224587, 0.369789327231, 0.36975863937, 0.369855492121, 0.370090120509,
                             0.370456638362, 0.370955636744, 0.371593427472, 0.37236755247, 0.373282787035, 0.374330521266,
                             0.37548259513, 0.376740794894, 0.378108493317, 0.379600043497, 0.381225881051, 0.382999965574,
                             0.384937055128, 0.387054062586, 0.389406784803, 0.392151134201, 0.395577451592, 0.398541075929,
                             0.396140574221, 0.377290889301])
            web3 = np.array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.883963623418, 0.882804861283,
                             0.881764724203, 0.880854972132, 0.880063593261, 0.879378185003, 0.878797670205, 0.878308574021,
                             0.87788500707, 0.877514541845, 0.877179551402, 0.876879055, 0.876610936666, 0.876374235852,
                             0.876166483426, 0.875986765133, 0.875832594685, 0.875702298474, 0.875595140019, 0.875508315318,
                             0.8754409845, 0.875392657588, 0.875361122776, 0.875345560917, 0.875342086256, 0.875341086148,
                             0.875340761092, 0.875339510238, 0.875338477803, 0.875337374186, 0.875336204651, 0.87533496247,
                             0.875333692251, 0.875332490358, 0.875331189455, 0.875329175403, 0.875324968372, 0.875312750097,
                             0.875329970752])
            web4 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])



        # Now compute some layup properties, independent of which turbine it is
        upperCS = [0]*NSTR
        lowerCS = [0]*NSTR
        websCS  = [0]*NSTR
        profile = [0]*NSTR

        for i in range(NSTR):
            webLoc = []
            if not np.isnan(web1[i]): webLoc.append(web1[i])
            if not np.isnan(web2[i]): webLoc.append(web2[i])
            if not np.isnan(web3[i]): webLoc.append(web3[i])
            if not np.isnan(web4[i]): webLoc.append(web4[i])

            upperCS[i], lowerCS[i], websCS[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(strucpath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
            profile[i] = Profile.initFromPreCompFile(os.path.join(strucpath, 'shape_' + str(i+1) + '.inp'))


        # Assign outputs
        unknowns['r_aero']                 = r_aero
        unknowns['r_str' ]                 = r_str
        unknowns['airfoil_files']          = airfoil_files
        unknowns['le_location']            = le_location
        unknowns['upperCS']                = upperCS
        unknowns['lowerCS']                = lowerCS
        unknowns['websCS']                 = websCS
        unknowns['profile']                = profile
        unknowns['chord_str_ref']          = chord_str_ref
        unknowns['sector_idx_strain_spar'] = sector_idx_strain_spar
        unknowns['sector_idx_strain_te']   = sector_idx_strain_te
        unknowns['materials']              = materials

        
    def linearize(self, params, unknowns, resids):
        J = {}
        J['r_aero','reference_turbine'] = 0.0
        J['r_str','reference_turbine'] = 0.0
        J['aifoil_files','reference_turbine'] = 0.0
        J['upperCS','reference_turbine'] = 0.0
        J['lowerCS','reference_turbine'] = 0.0
        J['websCS','reference_turbine'] = 0.0
        J['chord_str_ref','reference_turbine'] = 0.0
        J['le_location','reference_turbine'] = 0.0
        J['profile','reference_turbine'] = 0.0
        J['sector_idx_strain_spar','reference_turbine'] = 0.0
        J['sector_idx_strain_te','reference_turbine'] = 0.0
        return J

        
