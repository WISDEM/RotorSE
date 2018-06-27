import re
from akima import Akima
import numpy as np

# aerodyn_file_name = 'FAST_file_templates/WP_0.75MW/WP_0.75MW_AD.ipt'
# aerodyn_file_name = 'FAST_file_templates/WP_1.5MW/WP_1.5MW_AD.ipt'
# aerodyn_file_name = 'FAST_file_templates/WP_3.0MW/WP_3.0MW_AD.ipt'
aerodyn_file_name = 'FAST_file_templates/WP_5.0MW/WP_5.0MW_AD.ipt'

f = open(aerodyn_file_name, "r")

lines = f.readlines()
lines = lines[24:39]

DR_nodes = []
AeroTwst = []
DRNodes = []
Chord = []

for i in range(len(lines)):
    lines[i] = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lines[i].strip('\n'))
    DR_nodes.append(float(lines[i][0]))
    AeroTwst.append(float(lines[i][1]))
    DRNodes.append(float(lines[i][2]))
    Chord.append(float(lines[i][3]))

BldNodes = 17.0
tip_length = 64.0  # 35.0  # 25.0, 35.0 49.5, 64.0
hub_length = 3.2  # 1.75  # 1.25, 1.75, 2.475, 3.2

new_drnode_size = (tip_length-hub_length)/BldNodes

new_drnode = []
new_drnode.append(hub_length+new_drnode_size/2.0)
for i in range(1, int(BldNodes)):
    new_drnode.append(new_drnode[i-1]+new_drnode_size)


chord_spline = Akima(DR_nodes, Chord)
new_chord = chord_spline.interp(new_drnode)[0]

twist_spline = Akima(DR_nodes, AeroTwst)
new_twist = twist_spline.interp(new_drnode)[0]

new_airfoil = [1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4]

# # RNodes    AeroTwst  DRNodes  Chord  NFoil  PrnElm
for i in range(int(BldNodes)):
    print new_drnode[i], new_twist[i], new_drnode_size, new_chord[i], new_airfoil[i], "NOPRINT"
