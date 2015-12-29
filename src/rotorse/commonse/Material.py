#-------------------------------------------------------------------------------
# Name:        Material.py
# Purpose: This module contains the material class definition
#
# Author:      rdamiani
#
# Created:     04/11/2013
# Copyright:   (c) rdamiani 2013
# Licence:     APache (2014)
#-------------------------------------------------------------------------------

def main():
    """Material Class"""
    #The main fct is written to test the class
    mat=Material(E=3.5e5,rho=8500.)
    mat1=Material(matname='grout')
    print 'Example Returning mat and mat1 as 2 objects'
    return mat,mat1


class Material: #isotropic for the time being
    def __init__(self,**kwargs):

        prms={'matname':'ASTM992_steel','E':2.1e11,'nu':0.33,'G':7.895e10, 'rho':8502., 'fy':345.e6, 'fyc':345.e6} #SI Units
        prms.update(kwargs) #update in case user put some new params in
        for key in prms:  #Initialize material object with these parameters, possibly updated by user' stuff
            setattr(self,key,prms[key])

        #Fill in the rest of the fields user may have skipped

        #Predefined materials
        steel={'matname':'ASTM992_steel',          'E':2.1e11,'nu':0.33,'G':7.895e10,'rho':7805., 'fy':345.e6, 'fyc':345.e6,} #SI Units
        heavysteel={'matname':'ASTM992_steelheavy','E':2.1e11,'nu':0.33,'G':7.895e10,'rho':8741., 'fy':345.e6, 'fyc':345.e6} #SI Units
        grout={'matname':'Grout',                  'E':3.9e10,'nu':0.33,'G':1.466e10,'rho':2500., 'fy':20.68e6,'fyc':20.68e6} #SI Units  TO REVISE FOR GROUT

        if ((prms['matname'].lower() == 'steel') or (prms['matname'].lower() == 'astm992_steel')  or (prms['matname'].lower() == '')):

            for key in prms:
                if not(key in kwargs) or (prms[key]==[]):  #I need to operate on parameters not set by user
                    setattr(self,key,steel[key])

        elif ((prms['matname'].lower() == 'heavysteel') or (prms['matname'].lower() == 'astm992_steelheavy') or (prms['matname'].lower() == 'heavysteel')):

            for key in prms:
                if not(key in kwargs) or (prms[key]==[]):  #I need to operate on parameters not set by user
                    setattr(self,key,heavysteel[key])

        elif ((prms['matname'].lower() == 'concrete') or (prms['matname'].lower() == 'grout')):

            for key in prms:
                if not(key in kwargs) or (prms[key]==[]):
                    setattr(self,key,grout[key])

        if not(hasattr(self,'G')) or not(self.G):
            self.G=self.E/(2.*(1.+self.nu))  #isotropic
        else: #if G is given then recalculate nu
            self.nu=self.E/(2.*self.G)-1.

if __name__ == '__main__':
    mat,mat1=main()
