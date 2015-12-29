#-------------------------------------------------------------------------------
# Name:        Tube.py
# Purpose: This module contains the tube class, which calculates structural
#          properties of a hollow, uniform, cylindrical beam
#
# Author:      rdamiani
#
# Created:     04/11/2013
# Copyright:   (c) rdamiani 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import math
import numpy as np
from Material import Material
pi = math.pi

def main():
    pass

class Tube:
    """The Tube Class contains functions to calculate properties of tubular circular cross-sections
    for structural analyses. It assumes 1 material specification, even though I may pass
    more than 1 element (D's,L's,Kbuck's, and t's) to the class."""
    def __init__(self,D,t,Lgth=np.NaN, Kbuck=1., mat=Material(name='ASTM992 steel')):
        self.D=D
        self.t=t
        self.L=Lgth*np.ones(np.size(D)) #this makes sure we exapnd Lght if D,t, arrays
        self.Kbuck=Kbuck*np.ones(np.size(D)) #this makes sure we exapnd Kbuck if D,t, arrays
        self.mat=mat
        if np.size(D)>1 and type(mat) != np.ndarray: #in this case I need to make sure we have a list of materials
            import copy
            self.mat=np.array([copy.copy(mat) for i in xrange(np.size(D))])

    @property
    def Area(self): #Cross sectional area of tube
        return (self.D**2-(self.D-2*self.t)**2)* pi/4

    @property
    def Amid(self): #mid-thickness inscribed area of tube (thin wall torsion calculation)
        return (self.D-self.t)**2* pi/4

    @property
    def Jxx(self): #2nd area moment of inertia w.r.t. x-x axis (Jxx=Jyy for tube)
        return (self.D**4-(self.D-2*self.t)**4)* pi/64

    @property
    def Jyy(self): #2nd area moment of inertia w.r.t. x-x axis (Jxx=Jyy for tube)
        return self.Jxx

    @property
    def J0(self):  #polar moment of inertia w.r.t. z-z axis (torsional)
        return (self.D**4-(self.D-2*self.t)**4)* pi/32

    @property
    def Asy(self): #Shear Area for tubular cross-section
        Ri=self.D/2-self.t
        Ro=self.D/2
        return self.Area / ( 1.124235 + 0.055610*(Ri/Ro) + 1.097134*(Ri/Ro)**2 - 0.630057*(Ri/Ro)**3 )

    @property
    def Asx(self): #Shear Area for tubular cross-section
        return self.Asy

    @property
    def BdgMxx(self):  #Bending modulus for tubular cross-section
        return self.Jxx / (self.D/2)

    @property
    def BdgMyy(self):  #Bending modulus for tubular cross-section =BdgMxx
        return self.Jyy / (self.D/2)


    @property
    def TorsConst(self):  #Torsion shear constant for tubular cross-section
        return self.J0 / (self.D/2)

    @property
    def Rgyr(self): #Radius of Gyration for circular tube
        return np.sqrt(self.Jxx/self.Area)

    @property
    def Klr(self): #Klr buckling parameter
        return self.Kbuck*self.L/self.Rgyr


if __name__ == '__main__':
    main()
