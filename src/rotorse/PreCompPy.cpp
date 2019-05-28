//   PreComp v1.0.0a

//   Changes:
//   v2.0    07/26/2013  S. Andrew Ning      complete reorganization to allow calling from Python
//   v2.1    05/28/2019  G. Barter           Ported to C++

//
//   This code computes structural properties of a composite blade.

//   Given blade airfoil geometry, twist distribution, composite plies layup,
//   and material properties, PreComp computes cross-sectional stiffness
//   (flap bending, lag bending, torsion, axial, coupled bending-twist,
//   axial-twist, axial-bending), mass per unit length , flap inertia,
//   lag inertia, elastic-axis offset, tension center offset, and c.m. offset)
//   at user-specified span locations.

//     Developed at NWTC/NREL (by Gunjit Bir, 1617, Cole Blvd., Golden, Colorado.
//   phone: 303-384-6953, fax: 303-384-6901)
//
//   NOTE: This code is the property of NREL.  The code, or any part of it, may
//   not be transferred, altered or copied, directly or indirecly, without approval
//   from NWTC/NREL.

//   This code is still in a developmental stage and, therefore, no guarantee
//   is given on its accuracy and proper working. Also, the code is not fully
//   commented yet.
//.................................................................................



#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;

double pi = 3.14159265358979323846;
double r2d = 57.29577951308232;
double eps = 1.0e-10;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//HELPER FUNCTIONS++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// Getting sort indexes (argsort)
// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tuple<double, int> embed(double x, vector<double> &xnode, vector<double> &ynode) {
  //   purpose: embed a node in the upper-surface airfoil section nodes
  //   NOTE: nodal x coordinates must be in ascending order

  int inew  = -999;  // number of the embedded node, an output
  int isave = -999;
  int nnode = xnode.size();
  double y; // the output

  if (x < xnode[0] || x > xnode[nnode-1])
    throw invalid_argument(" ERROR** x position not within bounds");
  
  for (size_t i=0; i<nnode-1; i++) {
    double xl = xnode[i];
    double xr = xnode[i+1];
    double yl = ynode[i];

    if (fabs(x-xl) <= eps) {
      inew  = -1;
      isave = i;
      y     = yl;
      break;
    } else if (x < (xr-eps)) {
      double yr = ynode[i+1];
      // This is linear, could use a nicer Akima spline?
      y    = yl + (yr-yl)*(x-xl)/(xr-xl);
      inew = i+1;
      break;
    }
  }

  if (inew == -999) {
    if( fabs(x - xnode[nnode-1]) < eps) {
      inew = -1;
      isave = nnode-1;
      y =  ynode[nnode-1];
    } else {
      throw invalid_argument(" ERROR unknown, consult NWTC");
    }
  }

  if(inew >= 0) {
    xnode.insert(xnode.begin()+inew, x);
    ynode.insert(ynode.begin()+inew, y);
  } else
    inew = isave;

  return make_tuple(y, inew);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void seg_info(const double ch,  // chord length
	      const double rle, // loc of l.e. (non-d wrt chord)
	      const int nseg, // total number of segs
	      const int nseg_u, // no of segs on the upper surface
	      const int nseg_p, // no of segs for both upper and lower surfaces
	      const vector<double> &xnode_u, // x,y nodes on upper/lower
	      const vector<double> &ynode_u,
	      const vector<double> &xnode_l,
	      const vector<double> &ynode_l,
	      const int ndl1, // 1st seg lhs node number lower/upper surface
	      const int ndu1,
	      const vector<double> loc_web, // x coord of web, y coord of web upper/lower
	      const vector<double> weby_u,
	      const vector<double> weby_l,
	      const vector<int> n_scts, // no of sectors on 'is' surf
	      const int nsecnode,
	      const vector<vector<double>> xsec_node, // x coord of sect-i lhs on 's' surf
	      vector<int> &isur, // surf id
	      vector<int> &idsect, // associated sect or web number
	      vector<double> &yseg, // y-ref of mid-seg point
	      vector<double> &zseg, // z-ref of mid-seg point
	      vector<double> &wseg, // seg width
	      vector<double> &sthseg, // sin(th_seg)
	      vector<double> &cthseg, // cos(th_seg)
	      vector<double> &s2thseg, // sin(2*th_seg)
	      vector<double> &c2thseg) // cos(2*th_seg)
{
  //   NOTE: coord transformation from xaf-yaf to yre-zref and seg info
  
  // local
  //integer :: iseg, is, i, icheck, iweb, nd_a
  //real(dbp) :: xa, ya, xb, yb, xba, yba, thseg
  
  for (size_t iseg=0; iseg<nseg; iseg++) {
    int is = -999;
    int nd_a, iweb;
    bool icheck = false;
    double xa, xb, ya, yb;
    
    if(iseg <= nseg_u) {  // upper surface segs
      
      nd_a = ndu1 + iseg - 1;
      xa = xnode_u[nd_a];
      ya = ynode_u[nd_a];
      xb = xnode_u[nd_a+1];
      yb = ynode_u[nd_a+1];
      is = 0;
    } else {
      if(iseg <= nseg_p) {   // lower surface segs
	nd_a = ndl1 + iseg - nseg_u - 1;
	xa = xnode_l[nd_a];        //xref of node toward le (in a/f ref frame)
	ya = ynode_l[nd_a];        //yref of node toward le (in new ref frame)
	xb = xnode_l[nd_a+1];      //xref of node toward te (in a/f ref frame)
	yb = ynode_l[nd_a+1];      //yref of node toward te (in new ref frame)
	is = 1;
      }
      
      if(iseg > nseg_p ) {  // web segs
	iweb = iseg - nseg_p;
	xa = loc_web[iweb];
	xb = xa;
	ya = weby_u[iweb];
	yb = weby_l[iweb];
	is = -1;
      }
    } // end seg group identification
    
    if (is == -999) {
      cout << "iseg=" << iseg << endl;
      throw invalid_argument(" ERROR** unknown, contact NREL");
    }
    
    isur[iseg] = is;
    
    if(is >= 0) { //id assocaited sect number
      icheck = false;
      for (size_t i=0; i<n_scts[is]; i++) {
	if( (xa > (xsec_node[is][0]-eps)) && (xb < (xsec_node[is][i+1]+eps)) ) {
	  idsect[iseg] = i;
	  icheck = true;
	  break;
	}
      }
    }
    
    if(!icheck)
      throw invalid_argument(" ERROR** unknown, contact NREL");
    
    if(is == -1) idsect[iseg] = iweb;   //id assocaited web number
    
    double xba = xb - xa;
    double yba = ya - yb;
    yseg[iseg] = ch*(2.*rle-xa-xb)/2.0; //yref coord of mid-seg pt (in r-frame)
    zseg[iseg] = ch*(ya+yb)/2.0;    //zref coord of mid-seg pt (in r-frame)
    wseg[iseg] = ch*sqrt(xba*xba + yba*yba);
    
    double thseg = (is == 0) ? -pi/2.0 : atan(yba/xba); // thseg +ve in new y-z ref frame
    
    sthseg[iseg]  = sin(thseg);
    cthseg[iseg]  = cos(thseg);
    s2thseg[iseg] = sin(2.0*thseg);
    c2thseg[iseg] = cos(2.0*thseg);
  } // end seg loop
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

tuple<double, double, double, double, double, double, double> q_bars(const int mat,    // material id
								     const double thp, // ply orientation
								     const vector<double> density,
								     const vector<double> q11,
								     const vector<double> q22,
								     const vector<double> q12,
								     const vector<double> q66) {
  double ct = cos(thp);
  double st = sin(thp);

  double c2t = ct*ct;
  double c3t = c2t*ct;
  double c4t = c3t*ct;
  double s2t = st*st;
  double s3t = s2t*st;
  double s4t = s3t*st;
  double s2thsq = 4.0*s2t*c2t;

  double k11 = q11[mat];
  double k22 = q22[mat];
  double k12 = q12[mat];
  double k66 = q66[mat];
  double kmm = k11 - k12 - 2.0*k66;
  double kmp = k12 - k22 + 2.0*k66;

  // outputs
  double qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m;
  
  qbar11 = k11*c4t + 0.5*(k12 + 2.0*k66)*s2thsq + k22*s4t;
  qbar22 = k11*s4t + 0.5*(k12 + 2.0*k66)*s2thsq + k22*c4t;
  qbar12 = 0.25*(k11 + k22 - 4.0*k66)*s2thsq + k12*(s4t + c4t);
  qbar16 = kmm*st*c3t + kmp*s3t*ct;
  qbar26 = kmm*s3t*ct + kmp*st*c3t;
  qbar66 = 0.25*(kmm+k22-k12)*s2thsq  + k66*(s4t+c4t);

  rho_m = density[mat];

  return make_tuple(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, rho_m);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

vector<vector<double>> q_tildas(double qbar11, double qbar22, double qbar12,
				double qbar16, double qbar26, double qbar66,
				int mat) {

  vector<vector<double>> qtil(2, vector<double>(2, 0.0));
  
  qtil[0][0] = qbar11 - qbar12*qbar12/qbar22;
  if (qtil[0][0] < 0.0)
    cout << "  ERROR**: check material no, " << mat << " properties; these are not physically realizable." << endl;
    throw invalid_argument("");

  qtil[0][1] = qbar16 - qbar12*qbar26/qbar22;
  qtil[1][1] = qbar66 - qbar26*qbar26/qbar22;
  return qtil;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

struct PropertiesOut {
  double eifbar;      // EI_flap, Section flap bending stiffness about the YE axis (Nm2)
  double eilbar;      // EI_lag, Section lag (edgewise) bending stiffness about the XE axis (Nm2)
  double gjbar;       // GJ, Section torsion stiffness (Nm2)
  double eabar;       // EA, Section axial stiffness (N)
  double eiflbar;     // S_f, Coupled flap-lag stiffness with respect to the XE-YE frame (Nm2)
  double sfbar;       // S_airfoil, Coupled axial-flap stiffness with respect to the XE-YE frame (Nm)
  double slbar;       // S_al, Coupled axial-lag stiffness with respect to the XE-YE frame (Nm.)
  double sftbar;      // S_ft, Coupled flap-torsion stiffness with respect to the XE-YE frame (Nm2)
  double sltbar;      // S_lt, Coupled lag-torsion stiffness with respect to the XE-YE frame (Nm2)
  double satbar;      // S_at, Coupled axial-torsion stiffness (Nm)
  double z_sc;        // X_sc, X-coordinate of the shear-center offset with respect to the XR-YR axes (m)
  double y_sc;        // Y_sc, Chordwise offset of the section shear-center with respect to the reference frame, XR-YR (m)
  double ztc_ref;     // X_tc, X-coordinate of the tension-center offset with respect to the XR-YR axes (m)
  double ytc_ref;     // Y_tc, Chordwise offset of the section tension-center with respect to the XR-YR axes (m)
  double mass;        // Mass, Section mass per unit length (Kg/m)
  double iflap_eta;   // Flap_iner, Section flap inertia about the YG axis per unit length (Kg-m)
  double ilag_zeta;   // Lag_iner, Section lag inertia about the XG axis per unit length (Kg-m)
  double tw_iner;     // Tw_iner, Orientation of the section principal inertia axes with respect the blade reference plane, θ (deg)
  double zcm_ref;     // X_cm, X-coordinate of the center-of-mass offset with respect to the XR-YR axes (m)
  double ycm_ref;     // Y_cm, Chordwise offset of the section center of mass with respect to th  
public:
  PropertiesOut(){}
};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

struct LaminateProperties {
  vector<double> xsec_node; // normalized chord location of sector boundaries
  vector<double> n_lamina;  // number of lamina in each sector
  vector<double> n_plies;  // number of plies for the lamina
  vector<double> t_lam;    // thickness (m) for the lamina
  vector<double> tht_lam;  // orientation (deg) for the lamina
  vector<int> mat_lam;  // material id for the lamina
public:
  LaminateProperties(){}
  LaminateProperties(
		     const vector<double> &xsec_nodeIN, // normalized chord location of sector boundaries
		     const vector<double> &n_laminaIN,  // number of lamina in each sector
		     const vector<double> &n_pliesIN,  // number of plies for the lamina
		     const vector<double> &t_lamIN,    // thickness (m) for the lamina
		     const vector<double> &tht_lamIN,  // orientation (deg) for the lamina
		     const vector<int> &mat_lamIN)  // material id for the lamina
		     : xsec_node(xsec_nodeIN), n_lamina(n_laminaIN), n_plies(n_pliesIN),
		       t_lam(t_lamIN), tht_lam(tht_lamIN), mat_lam(mat_lamIN) {}
 
};

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PropertiesOut properties(
			 // geometry
			 const double chord,        // section chord length (m)
			 const double tw_aero_d,    // section twist angle (deg)
			 const double tw_prime_d,   // derivative of section twist angle w.r.t. span location (deg/m)
			 const double le_loc,       // leading edge location relative to reference axis (normalized by chord)
			 // airfoil coordinates starting at leading edge traversing upper surface and back around lower surface
			 const vector<double> &xnode,
			 const vector<double> &ynode,
			 // material properties: E1, E2, G12, Nu12, density
			 const vector<double> &e1,
			 const vector<double> &e2,
			 const vector<double> &g12,
			 const vector<double> &anu12,
			 const vector<double> &density,
			 // laminates
			 const LaminateProperties &laminatesU, // laminates upper
			 const LaminateProperties &laminatesL, // laminates lower
			 const LaminateProperties &laminatesW) // laminates web
{

  // Unpack some of the inputs
  // laminates upper
  vector<double> loc_web = laminatesW.xsec_node;
  int n_af_nodes	= xnode.size();	// number of airfoil nodes
  int n_materials	= e1.size();		// number of materials
  int n_sctU		= laminatesU.n_lamina.size();	// number of sectors on upper
  //int n_laminaTotalU	= laminatesU.n_plies.size();	// total number of lamina on upper
  int n_sctL		= laminatesL.n_lamina.size();	// number of sectors on lower
  //int n_laminaTotalL	= laminatesL.n_plies.size();	// total number of lamina on lower
  int nweb              = laminatesW.xsec_node.size();	// number of webs
  //int n_laminaTotalW	= laminatesW.n_plies.size();	// total number of lamina on webs

  // allocate and initialize
  int max_sectors = max( max(n_sctU, n_sctL), nweb);
  int max_laminates = max( max(*max_element(laminatesU.n_lamina.begin(), laminatesU.n_lamina.end()),
			       *max_element(laminatesL.n_lamina.begin(), laminatesL.n_lamina.end()) ),
			   *max_element(laminatesW.n_lamina.begin(), laminatesW.n_lamina.end()) );

  vector<vector<int>> n_laminas(2, vector<int>(max_sectors, 0));
  vector<vector<vector<double>>> tht_lam(2, vector<vector<double>>(max_sectors, vector<double>(max_laminates, 0.0)));
  vector<vector<vector<double>>> tlam(2, vector<vector<double>>(max_sectors, vector<double>(max_laminates, 0.0)));
  vector<vector<vector<int>>> mat_id(2, vector<vector<int>>(max_sectors, vector<int>(max_laminates, 0)));
  vector<vector<double>> xsec_node(2, vector<double>(max_sectors+1, 0.0));

  vector<double> yseg(n_af_nodes, 0.0);
  vector<double> zseg(n_af_nodes, 0.0);
  vector<double> wseg(n_af_nodes, 0.0);
  vector<double> sthseg(n_af_nodes, 0.0);
  vector<double> cthseg(n_af_nodes, 0.0);
  vector<double> s2thseg(n_af_nodes, 0.0);
  vector<double> c2thseg(n_af_nodes, 0.0);
  vector<int> isur(n_af_nodes, 0);
  vector<int> idsect(n_af_nodes, 0);
  
  // convert twist angle to radians
  double tw_aero = tw_aero_d / r2d;
  double tw_prime = tw_prime_d / r2d;

  // webs?
  bool webs_exist = true;
  if (nweb == 0) webs_exist = false;

  // ---- checks --------------
  //  check leading edge location
  if (le_loc < 0.0)
    cout << " WARNING** leading edge aft of reference axis **" << endl;

  // check materials
  for (size_t i=0; i<n_materials; i++) {
    if (anu12[i] > sqrt(e1[i]/e2[i]))
      cout << "**WARNING** material" <<  i << "properties not consistent" << endl;
  }

  // check airfoil nodes
  if (n_af_nodes <= 2)
    throw invalid_argument(" ERROR** min 3 nodes reqd to define airfoil geom");

  //   check if the first airfoil node is a leading-edge node and at (0,0)
  int location = distance(xnode.begin(), min_element(xnode.begin(), xnode.end()));
  if (location != 0)
    throw invalid_argument(" ERROR** the first airfoil node not a leading node");

  if ( (fabs(xnode[0]) > eps) || (fabs(ynode[0]) > eps) )
    throw invalid_argument(" ERROR** leading-edge node not located at (0,0)");

  //   identify trailing-edge end nodes on upper and lower surfaces
  double xnode_max = *max_element(xnode.begin(), xnode.end());
  if (fabs(xnode_max) > 1.0)
    throw invalid_argument(" ERROR** trailing-edge node exceeds chord boundary'");

  // ----------------
  //   get th_prime and phi_prime
  //call tw_rate(naf, sloc, tw_aero, th_prime)
  
  //do i = 1, naf
  //    phi_prime[i] = 0.  // later: get it from aeroelastic code
  //    tphip[i] = th_prime[i] + 0.5*phi_prime[i]
  //end do
  double tphip = tw_prime;

  // material properties
  vector<double> anud(n_materials, 0.0);
  vector<double> q11(n_materials, 0.0);
  vector<double> q22(n_materials, 0.0);
  vector<double> q12(n_materials, 0.0);
  vector<double> q66(n_materials, 0.0);
  for (size_t i=0; i<n_materials; i++) {
    anud[i] = 1.0 - anu12[i]*anu12[i]*e2[i]/e1[i];
    q11[i]  = e1[i] / anud[i];
    q22[i]  = e2[i] / anud[i];
    q12[i]  = anu12[i]*e2[i] / anud[i];
    q66[i]  = g12[i];
  }

  // begin blade sections loop sec-sec-sec-sec-sec-sec-sec-sec-sec-sec--------
  // ----------- airfoil data -------------------
  //   identify trailing-edge end nodes on upper and lower surfaces
  int tenode_u, tenode_l;
  tenode_u = distance(xnode.begin(), max_element(xnode.begin(), xnode.end()));

  for (size_t i=tenode_u; i<n_af_nodes; i++) {
    if ( fabs(xnode[i] - xnode[tenode_u]) < eps)
      tenode_l = i;
  }

  //   renumber airfoil nodes
  //   (modify later using equivalence or pointers)
  int nodes_u = tenode_u + 1;
  int nodes_l = n_af_nodes - tenode_l;

  vector<double> xnodeTMP_u, xnodeTMP_l, ynodeTMP_u, ynodeTMP_l;
  for (size_t i=0; i<nodes_u; i++) {
    xnodeTMP_u.push_back( xnode[i] );
    ynodeTMP_u.push_back( ynode[i] );
  }

  xnodeTMP_l.push_back( xnode[0] );
  ynodeTMP_l.push_back( ynode[0] );
  for (size_t i=tenode_l; i<n_af_nodes; i++) {
    xnodeTMP_l.push_back( xnode[i] );
    ynodeTMP_l.push_back( ynode[i] );
  }

  // Sort coordinates
  vector<double> xnode_u, xnode_l, ynode_u, ynode_l;
  for (auto k : sort_indexes(xnodeTMP_u)) {
    xnode_u.push_back( xnodeTMP_u[k] );
    ynode_u.push_back( ynodeTMP_u[k] );
  }
  for (auto k : sort_indexes(xnodeTMP_l)) {
    xnode_l.push_back( xnodeTMP_l[k] );
    ynode_l.push_back( ynodeTMP_l[k] );
  }
  // ----------------------------------------------
  

  // ------ more checks -------------
  //   ensure surfaces are single-valued functions
  for (size_t i=1; i<nodes_u; i++) {
    if ((xnode_u[i] - xnode_u[i-1]) <= eps )
      throw invalid_argument(" ERROR** upper surface not single-valued");
  }

  for (size_t i=1; i<nodes_l; i++) {
    if ((xnode_l[i] - xnode_l[i-1]) <= eps )
      throw invalid_argument(" ERROR** lower surface not single-valued");
  }
  
  //   check clockwise node numbering at the le
  if (ynode_u[1]/xnode_u[1] < ynode_l[1]/xnode_l[1])
    throw invalid_argument(" ERROR** airfoil node numbering not clockwise");

  //   check for single-connectivity of the airfoil shape
  //   (modify later using binary search)
  for (size_t j=1; j<nodes_l-1; j++) {
    double x = xnode_l[j];

    for (size_t i=0; i<nodes_u-1; i++) {
      double xl = xnode_u[i];
      double xr = xnode_u[i+1];

      if(x >= xl && x <= xr) {
	double yl = ynode_u[i];
	double yr = ynode_u[i+1];
	double y  = yl + (yr-yl)*(x-xl)/(xr-xl);

	if(ynode_l[j] >= y) 
	  throw invalid_argument(" ERROR** airfoil shape self-crossing");
      }
    } // upper loop
  } // lower loop
  // ---------- end checks ---------------------


  // -------------- webs ------------------
  //   embed airfoil nodes at web-to-airfoil intersections
  vector<double> weby_u(nweb, 0.0);
  vector<double> weby_l(nweb, 0.0);
  if (webs_exist) {
    for (size_t i=0; i<nweb; i++) {
      auto add_us = embed(loc_web[i], xnode_u, ynode_u);
      weby_u[i] = get<0>(add_us);

      auto add_ls = embed(loc_web[i], xnode_l, ynode_l);
      weby_l[i] = get<0>(add_ls);
    }
    
    // May have inserted nodes, so recount
    nodes_u = xnode_u.size();
    nodes_l = xnode_l.size();
  }
  // ----------------------------------------------


  // ------ internal srtucture data ------------
  vector<int> n_scts{n_sctU, n_sctL};
  for (size_t i=0; i<xsec_node[0].size(); i++) {
    xsec_node[0][i] = laminatesU.xsec_node[i];
    xsec_node[1][i] = laminatesL.xsec_node[i];
  }

  // unpack data
  int k;
  
  k = 0;
  for (size_t i=0; i<n_sctU; i++) {
    n_laminas[0][i] = laminatesU.n_lamina[i];

    for (size_t j=0; j<laminatesU.n_lamina[i]; j++) {
      tlam[0][i][j]    = laminatesU.n_plies[k] * laminatesU.t_lam[k];
      tht_lam[0][i][j] = laminatesU.tht_lam[k] / r2d;
      mat_id[0][i][j]  = laminatesU.mat_lam[k];
      k++;
    }
  }
  
  k = 0;
  for (size_t i=0; i<n_sctL; i++) {
    n_laminas[1][i] = laminatesL.n_lamina[i];

    for (size_t j=0; j<laminatesL.n_lamina[i]; j++) {
      tlam[1][i][j]    = laminatesL.n_plies[k] * laminatesL.t_lam[k];
      tht_lam[1][i][j] = laminatesL.tht_lam[k] / r2d;
      mat_id[1][i][j]  = laminatesL.mat_lam[k];
      k++;
    }
  }

  vector<int> n_weblams(nweb, 0);
  vector<vector<double>> twlam(nweb, vector<double>(6, 0.0));
  vector<vector<double>> tht_wlam(nweb, vector<double>(6, 0.0));
  vector<vector<int>> wmat_id(nweb, vector<int>(6, 0));
  k = 0;
  for (size_t i=0; i<nweb; i++) {
    n_weblams[i] = laminatesW.n_lamina[i];

    for (size_t j=0; j<laminatesW.n_lamina[i]; j++) {
      twlam[i][j]    = laminatesW.n_plies[k] * laminatesW.t_lam[k];
      tht_wlam[i][j] = laminatesW.tht_lam[k] / r2d;
      wmat_id[i][j]  = laminatesW.mat_lam[k];
      k++;
    }
  }

  // begin loop for blade surfaces
  double xu1, xu2, xl1, xl2;
  double yu1, yu2, yl1, yl2;
  double ndu1, ndu2, ndl1, ndl2;
  for (size_t is=0; is<2; is++) {
    int nsects = n_scts[is];

    if (nsects <= 0)
      throw invalid_argument(" ERROR** no of sectors not positive");

    if (xsec_node[is][0] < 0.0)
      throw invalid_argument(" ERROR** sector node x-location not positive");

    if (is == 0) {
      xu1 = xsec_node[is][0];
      xu2 = xsec_node[is][nsects];
      if (xu2 > xnode_u[nodes_u-1])
	throw invalid_argument(" ERROR** upper-surf last sector node out of bounds");
    } else {
      xl1 = xsec_node[is][0];
      xl2 = xsec_node[is][nsects];
      if (xl2 > xnode_l[nodes_l-1])
	throw invalid_argument(" ERROR** lower-surf last sector node out of bounds");
    }
 
    for (size_t i=0; i<nsects; i++)
      if (xsec_node[is][i+1] <= xsec_node[is][i])
	throw invalid_argument(" ERROR** sector nodal x-locations not in ascending order");
    
    // embed airfoil nodes representing sectors bounds
    for (size_t i=0; i<nsects+1; i++) {
      if (is == 0) {
	auto add_us = embed(xsec_node[is][i], xnode_u, ynode_u);
	
	if (i == 0) {
	  yu1 = get<0>(add_us);
	  ndu1 = get<1>(add_us);
	}
	if (i == nsects) {
	  yu2 = get<0>(add_us);
	  ndu2 = get<1>(add_us);
	}
      }

      if (is == 1) {
	auto add_ls = embed(xsec_node[is][i], xnode_l, ynode_l);
	
	if (i == 0) {
	  yl1 = get<0>(add_ls);
	  ndl1 = get<1>(add_ls);
	}
	if (i == nsects) {
	  yl2 = get<0>(add_ls);
	  ndl2 = get<1>(add_ls);
	}
      }
	
    } // nsects loop
  } // end blade surfaces loop
  
  //.... check for le and te non-closures and issue warning ....
  if (fabs(xu1-xl1) > eps) {
    cout << " WARNING** the leading edge may be open; check closure" << endl;
  } else {
    if ((yu1-yl1) > eps) {
      bool wreq = true;
      
      if (webs_exist)
	if (fabs(xu1-loc_web[0]) < eps)
	  wreq = false;
      
      if (wreq)
	cout << " WARNING** open leading edge; check web requirement" << endl;
    }
  }
  
  if (fabs(xu2-xl2) > eps) {
    cout << " WARNING** the trailing edge may be open; check closure" << endl;
  } else {
    if ((yu2-yl2) > eps) {
      bool wreq = true;
      
      if (webs_exist)
	if (fabs(xu2-loc_web[nweb-1]) < eps)
	  wreq = false;
      
      if (wreq)
	cout << " WARNING** open trailing edge; check web requirement" << endl;
    }
  }
  
  //................
  if (webs_exist) {
    if(loc_web[0] < xu1 || loc_web[0] < xl1)
      cout << " WARNING** first web out of sectors-bounded airfoil" << endl;
    
    if(loc_web[nweb-1] > xu2 || loc_web[nweb-1] > xl2)
      cout << " WARNING** last web out of sectors-bounded airfoil" << endl;
  }
  // ------------- Done Processing Inputs ----------------------
  
  
  // ----------- Start Computations ------------------
  //   identify segments groupings and get segs attributes
  int nseg_u = ndu2 - ndu1;
  int nseg_l = ndl2 - ndl1;
  int nseg_p = nseg_u + nseg_l;    // no of peripheral segments
  
  // total no of segments (with and without webs in section)
  int nseg = nseg_p;
  if (webs_exist) nseg += nweb;
  
  seg_info(chord, le_loc, nseg, nseg_u, nseg_p, xnode_u, ynode_u, xnode_l, ynode_l, ndl1, ndu1,
	   loc_web, weby_u, weby_l, n_scts, max_sectors+1, xsec_node, isur, idsect,
	   yseg, zseg, wseg, sthseg, cthseg, s2thseg, c2thseg);
  //------------------------------------------
  
  //   initialize for section (sc)
  double sigma = 0.0;
  double eabar = 0.0;
  double q11ya = 0.0;
  double q11za = 0.0;
  
  //   segments loop for sc
  for (size_t iseg=0; iseg<nseg_p; iseg++) {
    //begin paeripheral segments loop (sc)
    
    //     retrieve seg attributes
    int is      = isur[iseg];
    int idsec   = idsect[iseg];
    double ysg  = yseg[iseg];
    double zsg  = zseg[iseg];
    double w    = wseg[iseg];
    double sths = sthseg[iseg];
    double cths = cthseg[iseg];
    //       s2ths = s2thseg[iseg];
    //       c2ths = c2thseg[iseg];
    
    int nlam = n_laminas[is][idsec];    // for sector seg
    
    //     initialization for seg (sc)
    double tbar    = 0.0;
    double q11t    = 0.0;
    double q11yt_u = 0.0;
    double q11zt_u = 0.0;
    double q11yt_l = 0.0;
    double q11zt_l = 0.0;
    
    for (size_t ilam=0; ilam<nlam; ilam++) { //laminas loop (sc)
      
      double t   = tlam[is][idsec][ilam];    // thickness
      double thp = tht_lam[is][idsec][ilam]; // ply angle
      double mat = mat_id[is][idsec][ilam];  // material

      tbar += t/2.0;
      double y0 = ysg - pow(-1.0, is+1.0)*tbar*sths;
      double z0 = zsg + pow(-1.0, is+1.0)*tbar*cths;
      
      // obtain qtil for specified mat
      auto outdata = q_bars(mat, thp, density, q11, q22, q12, q66);
      double qbar11 = get<0>(outdata);
      double qbar22 = get<1>(outdata);
      double qbar12 = get<2>(outdata);
      double qbar16 = get<3>(outdata);
      double qbar26 = get<4>(outdata);
      double qbar66 = get<5>(outdata);
      //double rho_m  = get<6>(outdata);
      
      vector<vector<double>> qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, mat);
      
      // add seg-laminas contributions (sc)
      double qtil11t = qtil[0][0]*t;
      q11t += qtil11t;
      if(iseg <= nseg_u) {
	q11yt_u = q11yt_u + qtil11t*y0;
	q11zt_u = q11zt_u + qtil11t*z0;
      } else {
	q11yt_l = q11yt_l + qtil11t*y0;
	q11zt_l = q11zt_l + qtil11t*z0;
      }
      
      tbar += t/2.0;
    }  // end laminas loop
    
    // add seg contributions (sc)
    sigma = sigma + w*fabs(zsg + pow(-1.0,is+1.0)*0.5*tbar*cths)*cths;
    eabar = eabar + q11t*w;
    q11ya = q11ya + (q11yt_u + q11yt_l)*w;
    q11za = q11za + (q11zt_u + q11zt_l)*w;
    
  } //end af_periph segment loop (sc)
  
  // get section sc
  double y_sc = q11ya/eabar;     //wrt r-frame
  double z_sc = q11za/eabar;     //wrt r-frame
  //---------------- end section sc -----------

  //   initializations for section (properties)
  eabar		 = 0.0;
  q11ya		 = 0.0;
  q11za		 = 0.0;
  double ap	 = 0.0;
  double bp	 = 0.0;;
  double cp	 = 0.0;
  double dp	 = 0.0;
  double ep	 = 0.0;
  double q11ysqa = 0.0;
  double q11zsqa = 0.0;
  double q11yza	 = 0.0;
  
  double mass	 = 0.0;
  double rhoya	 = 0.0;
  double rhoza	 = 0.0;
  double rhoysqa = 0.0;
  double rhozsqa = 0.0;
  double rhoyza	 = 0.0;
  
  //   segments loop (for properties)
  for (size_t iseg=0; iseg<nseg; iseg++) {
    
    // retrieve seg attributes
    int is 	 = isur[iseg];
    int idsec	 = idsect[iseg];
    double ysg	 = yseg[iseg];
    double zsg	 = zseg[iseg];
    double w	 = wseg[iseg];
    double sths	 = sthseg[iseg];
    double cths	 = cthseg[iseg];
    double s2ths = s2thseg[iseg];
    double c2ths = c2thseg[iseg];

    int nlam, iweb;
    if(is >= 0) {
      nlam = n_laminas[is][idsec];  // for sector seg
    } else {
      iweb = idsec;
      nlam = n_weblams[iweb];      // for web seg
    }
    
    // initialization for seg (properties)
    double tbar	= 0.0;
    double q11t	= 0.0;
    double q11yt	= 0.0;
    double q11zt	= 0.0;
    double dtbar	= 0.0;
    double q2bar	= 0.0;
    double zbart	= 0.0;
    double ybart	= 0.0;
    double tbart	= 0.0;
    double q11ysqt	= 0.0;
    double q11zsqt	= 0.0;
    double q11yzt	= 0.0;
    
    double rhot	= 0.0;
    double rhoyt	= 0.0;
    double rhozt	= 0.0;
    double rhoysqt	= 0.0;
    double rhozsqt	= 0.0;
    double rhoyzt	= 0.0;

    for (size_t ilam=0; ilam<nlam; ilam++) { //laminas loop (properties)

      double t, thp, y0, z0;
      int mat;
      if(is >= 0) {
	t     = tlam[is][idsec][ilam];          //thickness
	thp   = tht_lam[is][idsec][ilam];  // ply angle
	mat   = mat_id[is][idsec][ilam];      // material
	tbar += t/2.0;
	y0    = ysg - pow(-1.0,is+1.0)*tbar*sths - y_sc;
	z0    = zsg + pow(-1.0,is+1.0)*tbar*cths - z_sc;
      } else {
	t     = twlam[iweb][ilam];
	thp   = tht_wlam[iweb][ilam];
	mat   = wmat_id[iweb][ilam];
	tbar += t/2.0;
	y0    = ysg - tbar/2.0 - y_sc;
	z0    = zsg - z_sc;
      }

      double y0sq = y0*y0;
      double z0sq = z0*z0;
      
      // obtain qtil and rho for specified mat
      auto outdata = q_bars(mat, thp, density, q11, q22, q12, q66);
      double qbar11 = get<0>(outdata);
      double qbar22 = get<1>(outdata);
      double qbar12 = get<2>(outdata);
      double qbar16 = get<3>(outdata);
      double qbar26 = get<4>(outdata);
      double qbar66 = get<5>(outdata);
      double rho_m  = get<6>(outdata);
      
      vector<vector<double>> qtil = q_tildas(qbar11, qbar22, qbar12, qbar16, qbar26, qbar66, mat);
      
      double ieta1  = t*t/12.0;
      double izeta1 = w*w/12.0;
      double iepz   = 0.5*(ieta1+izeta1);
      double iemz   = 0.5*(ieta1-izeta1);
      double ipp    = iepz + iemz*c2ths;   // check this block later
      double iqq    = iepz - iemz*c2ths;
      double ipq    = iemz*s2ths;
      
      double rot = rho_m*t;
      double qtil11t = qtil[0][0]*t;
      
      //add laminas contributions (properties) at current segment
      if(is >= 0) { // peripheral segs contribution
	
	double qtil12t = qtil[0][1]*t;
	double qtil22t = qtil[1][1]*t;
	
	q11t  += qtil11t;
	q11yt += qtil11t*y0;
	q11zt += qtil11t*z0;
	
	dtbar += qtil12t*(y0sq+z0sq)*tphip*t;
	q2bar += qtil22t;    // later: retain only this block
	zbart += z0*qtil12t;
	ybart += y0*qtil12t;
	tbart += qtil12t;
	
	q11ysqt += qtil11t*(y0sq+iqq);
	q11zsqt += qtil11t*(z0sq+ipp);
	q11yzt  += qtil11t*(y0*z0+ipq);
	
	rhot    += rot;
	rhoyt   += rot*y0;
	rhozt   += rot*z0;
	rhoysqt += rot*(y0sq+iqq);
	rhozsqt += rot*(z0sq+ipp);
	rhoyzt  += rot*(y0*z0+ipq);
	
      } else {            //web segs contribution
	
	q11t    += qtil11t;
	q11yt   += qtil11t*y0;
	q11zt   += qtil11t*z0;
	q11ysqt += qtil11t*(y0sq+iqq);
	q11zsqt += qtil11t*(z0sq+ipp);
	q11yzt  += qtil11t*(y0*z0+ipq);
	
	rhot    += rot;
	rhoyt   += rot*y0;
	rhozt   += rot*z0;
	rhoysqt += rot*(y0sq+iqq);
	rhozsqt += rot*(z0sq+ipp);
	rhoyzt  += rot*(y0*z0+ipq);
      }
      
      tbar += t/2.0;
    }    // end laminas loop

    // add seg contributions to obtain sec props about ref_parallel axes at sc
    eabar   += q11t*w;
    q11ya   += q11yt*w;
    q11za   += q11zt*w;
    q11ysqa += q11ysqt*w;
    q11zsqa += q11zsqt*w;
    q11yza  += q11yzt*w;
    
    if(is >= 0) {
      double wdq2bar = w/q2bar;
      ap += wdq2bar;
      bp += wdq2bar*tbart;
      cp += wdq2bar*dtbar;
      dp += wdq2bar*zbart;
      ep += wdq2bar*ybart;
    }
    
    mass    += rhot*w;
    rhoya   += rhoyt*w;
    rhoza   += rhozt*w;
    rhoysqa += rhoysqt*w;
    rhozsqa += rhozsqt*w;
    rhoyza  += rhoyzt*w;
  }    //end af_periph segment loop (properties)

  //  get more section properties // about ref_parallel axes at sc
  double y_tc		= q11ya/eabar;
  double z_tc		= q11za/eabar;
  
  double sfbar	= q11za;
  double slbar	= q11ya;
  double eifbar	= q11zsqa;
  double eilbar	= q11ysqa;
  double eiflbar	= q11yza;
  
  double sigm2	= sigma*2.0;
  double gjbar	= sigm2*(sigm2+cp)/ap;
  double sftbar	= -sigm2*dp/ap;
  double sltbar	= -sigm2*ep/ap;
  double satbar	= sigm2*bp/ap;
  
  double ycm_sc	= rhoya/mass; //wrt sc
  double zcm_sc	= rhoza/mass; //wrt sc
  
  double iflap_sc	= rhozsqa; //wrt sc
  double ilag_sc	= rhoysqa;   //wrt sc
  double ifl_sc	= rhoyza;     //wrt sc
  
  // get section tc and cm
  double ytc_ref = y_tc + y_sc;  //wrt the ref axes
  double ztc_ref = z_tc + z_sc;  //wrt the ref axes
  
  double ycm_ref = ycm_sc + y_sc;    //wrt the ref axes
  double zcm_ref = zcm_sc + z_sc;    //wrt the ref axes
  
  // moments of inertia // about ref_parallel axes at cm
  double iflap_cm = iflap_sc - mass*zcm_sc*zcm_sc;
  double ilag_cm  = ilag_sc  - mass*ycm_sc*ycm_sc;
  double ifl_cm   = ifl_sc   - mass*ycm_sc*zcm_sc;
  
  // inertia principal axes orientation and moments of inertia
  double m_inertia = 0.5*(ilag_cm + iflap_cm);
  double r_inertia = sqrt(0.25*pow(ilag_cm-iflap_cm, 2.0) + ifl_cm*ifl_cm);
  
  double iflap_eta, ilag_zeta;
  if(iflap_cm <= ilag_cm) {
    iflap_eta = m_inertia - r_inertia;
    ilag_zeta = m_inertia + r_inertia;
  } else {
    iflap_eta = m_inertia + r_inertia;
    ilag_zeta = m_inertia - r_inertia;
  }
  
  double th_pa;
  if(ilag_cm == iflap_cm) {
    th_pa = pi/4.0;
    if(fabs(ifl_cm/iflap_cm) < 1e-6) th_pa = 0.0;
  } else {
    th_pa = 0.5*fabs(atan(2.0*ifl_cm/(ilag_cm-iflap_cm)));
  }
  
  if(fabs(ifl_cm) < eps) {
    th_pa = 0.0;
  } else {          // check this block later
    if( (iflap_cm >= ilag_cm) && (ifl_cm > 0.) ) th_pa = -th_pa;
    //if( (iflap_cm >= ilag_cm) && (ifl_cm < 0.) ) th_pa = th_pa;
    //if( (iflap_cm < ilag_cm) && (ifl_cm > 0.) ) th_pa = th_pa;
    if( (iflap_cm < ilag_cm) && (ifl_cm < 0.) ) th_pa = -th_pa;
  }

  // elastic principal axes orientation and principal bending stiffneses
  double em_stiff = 0.5*(eilbar + eifbar);
  double er_stiff = sqrt(0.25*pow(eilbar-eifbar, 2.0) + eiflbar*eiflbar);
  
  double pflap_stff, plag_stff;
  if(eifbar <= eilbar) {
    pflap_stff = em_stiff - er_stiff;
    plag_stff  = em_stiff + er_stiff;
  } else {
    pflap_stff = em_stiff + er_stiff;
    plag_stff  = em_stiff - er_stiff;
  }
  
  double the_pa = (eilbar == eifbar) ? pi/4.0 : 0.5*fabs(atan(2.0*eiflbar/(eilbar-eifbar)));
  if(fabs(eiflbar) < eps) {
    the_pa = 0.0;
  } else {           // check this block later
    if( (eifbar >= eilbar) && (eiflbar > 0.) ) the_pa = -the_pa;
    //if( (eifbar >= eilbar) && (eiflbar < 0.) ) the_pa = the_pa;
    //if( (eifbar < eilbar) && (eiflbar > 0.) ) the_pa = the_pa;
    if( (eifbar < eilbar) && (eiflbar < 0.) ) the_pa = -the_pa;
  }
  //---------------- end properties computation -----------

  // ---------- prepare outputs --------------
  PropertiesOut myout = PropertiesOut();
  myout.eifbar    = eifbar;
  myout.eilbar    = eilbar;
  myout.gjbar     = gjbar;
  myout.eabar     = eabar;
  myout.eiflbar   = eiflbar;
  myout.sfbar     = sfbar;
  myout.slbar     = slbar;
  myout.sftbar    = sftbar;
  myout.sltbar    = sltbar;
  myout.satbar    = satbar;
  myout.mass      = mass;
  myout.z_sc      = z_sc;
  myout.y_sc      = y_sc;
  myout.ztc_ref   = ztc_ref;
  myout.ytc_ref   = ytc_ref;
  myout.iflap_eta = iflap_eta;
  myout.ilag_zeta = ilag_zeta;
  //myout.tw_iner   = tw_iner;
  myout.zcm_ref   = zcm_ref;
  myout.ycm_ref   = ycm_ref;
  
  bool id_form = true;  // hardwired for wt's
  
  if (id_form) {
    myout.tw_iner = tw_aero - th_pa;
    //myout.str_tw =  tw_aero - the_pa;
    myout.y_sc = -y_sc;
    myout.ytc_ref = -ytc_ref;
    myout.ycm_ref = -ycm_ref;
  } else {         // for h/c
    //       note: for h/c, th_aero input is +ve acc to h/c convention
    myout.tw_iner = tw_aero + th_pa;
    //myout.str_tw =  tw_aero + the_pa;
  }
  
  // conversions
  myout.eiflbar  = -eiflbar;
  myout.sfbar    = -sfbar;
  myout.sltbar   = -sltbar;
  myout.tw_iner *= r2d;

  return myout;
} // end properties function

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vector<double> tw_rate(vector<double> sloc, vector<double> tw_aero) {
  
  // Initialize outputs
  int naf = sloc.size();
  vector<double> th_prime(naf, 0.0);
  
  for (size_t i=1; i<naf-1; i++) {
    double f0 = tw_aero[i];
    double f1 = tw_aero[i-1];
    double f2 = tw_aero[i+1];
    double h1 = sloc[i] - sloc[i-1];
    double h2 = sloc[i+1] - sloc[i];
    th_prime[i] = (h1*(f2-f0) + h2*(f0-f1))/(2.0*h1*h2);
  }

  th_prime[0]     = (tw_aero[1]     - tw_aero[0]    ) / (sloc[1]    - sloc[0]    );
  th_prime[naf-1] = (tw_aero[naf-1] - tw_aero[naf-2]) / (sloc[naf-1]- sloc[naf-2]);
  return th_prime;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





//MARK: ---------- PYTHON WRAPPER FOR AKIMA ---------------------
namespace py = pybind11;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
struct LaminatePropertiesPY {
  LaminateProperties lamprop = LaminateProperties();
public:
  LaminatePropertiesPY(
		       const py::array_t<double, py::array::c_style> xsec_nodeIN, // normalized chord location of sector boundaries
		       const py::array_t<double, py::array::c_style> n_laminaIN,  // number of lamina in each sector
		       const py::array_t<double, py::array::c_style> n_pliesIN,  // number of plies for the lamina
		       const py::array_t<double, py::array::c_style> t_lamIN,    // thickness (m) for the lamina
		       const py::array_t<double, py::array::c_style> tht_lamIN,  // orientation (deg) for the lamina
		       const py::array_t<int, py::array::c_style> mat_lamIN)  // material id for the lamina
  {
    // allocate std::vector (to pass to the C++ function)
    vector<double> xsec_node(xsec_nodeIN.size());
    vector<double> n_lamina(n_laminaIN.size());
    vector<double> n_plies(n_pliesIN.size());
    vector<double> t_lam(t_lamIN.size());
    vector<double> tht_lam(tht_lamIN.size());
    vector<int> mat_lam(mat_lamIN.size());

    // copy py::array -> std::vector
    memcpy(xsec_node.data(), xsec_nodeIN.data(), xsec_nodeIN.size()*sizeof(double));
    memcpy(n_lamina.data(), n_laminaIN.data(), n_laminaIN.size()*sizeof(double));
    memcpy(n_plies.data(), n_pliesIN.data(), n_pliesIN.size()*sizeof(double));
    memcpy(t_lam.data(), t_lamIN.data(), t_lamIN.size()*sizeof(double));
    memcpy(tht_lam.data(), tht_lamIN.data(), tht_lamIN.size()*sizeof(double));
    memcpy(mat_lam.data(), mat_lamIN.data(), mat_lamIN.size()*sizeof(int));

    // Call constructor
    lamprop = LaminateProperties(xsec_node, n_lamina, n_plies, t_lam, tht_lam, mat_lam);
  }
};
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

py::tuple propertiesPY(
		     const double chord,
		     const double tw_aero_d,
		     const double tw_prime_d,
		     const double le_loc,
		     const py::array_t<double, py::array::c_style> xnodeIN,
		     const py::array_t<double, py::array::c_style> ynodeIN,
		     const py::array_t<double, py::array::c_style> e1IN,
		     const py::array_t<double, py::array::c_style> e2IN,
		     const py::array_t<double, py::array::c_style> g12IN,
		     const py::array_t<double, py::array::c_style> anu12IN,
		     const py::array_t<double, py::array::c_style> densityIN,
		     const py::object laminatesU_obj,
		     const py::object laminatesL_obj,
		     const py::object laminatesW_obj)
{
  // allocate std::vector (to pass to the C++ function)
  vector<double> xnode(xnodeIN.size());
  vector<double> ynode(ynodeIN.size());
  vector<double> e1(e1IN.size());
  vector<double> e2(e2IN.size());
  vector<double> g12(g12IN.size());
  vector<double> anu12(anu12IN.size());
  vector<double> density(densityIN.size());

  // copy py::array -> std::vector
  memcpy(xnode.data(), xnodeIN.data(), xnodeIN.size()*sizeof(double));
  memcpy(ynode.data(), ynodeIN.data(), ynodeIN.size()*sizeof(double));
  memcpy(e1.data(), e1IN.data(), e1IN.size()*sizeof(double));
  memcpy(e2.data(), e2IN.data(), e2IN.size()*sizeof(double));
  memcpy(g12.data(), g12IN.data(), g12IN.size()*sizeof(double));
  memcpy(anu12.data(), anu12IN.data(), anu12IN.size()*sizeof(double));
  memcpy(density.data(), densityIN.data(), densityIN.size()*sizeof(double));

  // Set input py-objects as correct type
  LaminatePropertiesPY laminatesU = laminatesU_obj.cast<LaminatePropertiesPY>();
  LaminatePropertiesPY laminatesL = laminatesL_obj.cast<LaminatePropertiesPY>();
  LaminatePropertiesPY laminatesW = laminatesW_obj.cast<LaminatePropertiesPY>();

  PropertiesOut myout = properties(chord, tw_aero_d, tw_prime_d, le_loc,
				   xnode, ynode, e1, e2, g12, anu12, density,
				   laminatesU.lamprop, laminatesL.lamprop, laminatesW.lamprop);
  
  return py::make_tuple(myout.eifbar, myout.eilbar, myout.gjbar, myout.eabar, myout.eiflbar,
			myout.sfbar, myout.slbar, myout.sftbar, myout.sltbar, myout.satbar,
			myout.z_sc, myout.y_sc, myout.ztc_ref, myout.ytc_ref, myout.mass, myout.iflap_eta,
			myout.ilag_zeta, myout.tw_iner, myout.zcm_ref, myout.ycm_ref);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

py::array_t<double> tw_ratePY(const py::array_t<double, py::array::c_style> slocIN,
			      const py::array_t<double, py::array::c_style> tw_aeroIN) {
    // allocate std::vector (to pass to the C++ function)
    vector<double> sloc(slocIN.size());
    vector<double> tw_aero(tw_aeroIN.size());

    // copy py::array -> std::vector
    memcpy(sloc.data(), slocIN.data(), slocIN.size()*sizeof(double));
    memcpy(tw_aero.data(), tw_aeroIN.data(), tw_aeroIN.size()*sizeof(double));

    // Call function
    vector<double> th_prime = tw_rate(sloc, tw_aero);

    // Stuff std vector into python array
    auto th_primeB = py::buffer_info(
				     // Pointer to buffer
				     th_prime.data(),
				     // Size of one scalar
				     sizeof(double),
				     // Python struct-style format descriptor
				     py::format_descriptor<double>::format(),
				     // Number of dimensions
				     1,
				     // Buffer dimensions
				     { th_prime.size() },
				     // Strides (in bytes) for each index
				     { sizeof(double) }
				     );
    //auto th_primePY = py::array_t<double>(th_primeB);

    return py::array_t<double>(th_primeB);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



// MARK: --------- PYTHON MODULE ---------------

PYBIND11_MODULE(_precomp, m)
{
  m.doc() = "PreComp python plugin module";

  py::class_<LaminatePropertiesPY>(m, "LaminateProperties")
    .def(py::init<py::array_t<double, py::array::c_style>,
	 py::array_t<double, py::array::c_style>,
	 py::array_t<double, py::array::c_style>,
	 py::array_t<double, py::array::c_style>,
	 py::array_t<double, py::array::c_style>,
	 py::array_t<int, py::array::c_style>>());
  
  m.def("properties", &propertiesPY, "properties from PreComp");
  m.def("tw_rate", &tw_ratePY, "tw_rate from PreComp");
}
