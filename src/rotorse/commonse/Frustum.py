#-------------------------------------------------------------------------
# Name:        SegIntersect.py
# Purpose: It Calculates Intersection of 2 segments in space. Also contains a simple 2-point distance calculator.
#
# Author:      RRD
#
# Created:     24/10/2012
# Copyright:   (c) rdamiani 2012
# Licence:     <your licence>
#-------------------------------------------------------------------------


import numpy as np

def frustum(Db, Dt, H):
    """This function returns a frustum's volume and center of mass, CM

    INPUT:
    Parameters
    ----------
    Db : float,        base diameter
    Dt : float,        top diameter
    H : float,         height

    OUTPUTs:
    -------
    vol : float,        volume
    cm : float,        geometric centroid relative to bottom (center of mass if uniform density)

    """

    vol = np.pi/12*H * (Db**2 + Dt**2 + Db * Dt)
    cm = H/4 * (Db**2 + 3*Dt**2 + 2*Db*Dt) / (Db**2 + Dt**2 + Db*Dt)

    return vol, cm



if __name__ == '__main__':
    Db=6.5
    Dt=4.
    H=120.

    print ('From commonse.Frustum: Sample Volume and CM of FRUSTUM='+4*'{:8.4f}, ').format(*frustum(Db,Dt,H)[0].flatten())

def main():
    pass