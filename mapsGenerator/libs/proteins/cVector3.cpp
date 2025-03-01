/*************************************************************************\

  Copyright 2004 The University of North Carolina at Chapel Hill.
  All Rights Reserved.

  Permission to use, copy, modify OR distribute this software and its
  documentation for educational, research and non-profit purposes, without
  fee, and without a written agreement is hereby granted, provided that the
  above copyright notice and the following three paragraphs appear in all
  copies.

  IN NO EVENT SHALL THE UNIVERSITY OF NORTH CAROLINA AT CHAPEL HILL BE
  LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
  CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
  USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY
  OF NORTH CAROLINA HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH
  DAMAGES.

  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
  PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
  NORTH CAROLINA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

  The author may be contacted via:

  US Mail:             Stephane Redon
                       Department of Computer Science
                       Sitterson Hall, CB #3175
                       University of N. Carolina
                       Chapel Hill, NC 27599-3175

  Phone:               (919) 962-1930

  EMail:               redon@cs.unc.edu

\**************************************************************************/

#include <iostream>

#include "cVector3.hpp"

cVector3::cVector3() {}
cVector3::cVector3(double x, double y, double z) { v[0]=x;v[1]=y;v[2]=z; }
cVector3::cVector3(double val[3]) { v[0]=val[0];v[1]=val[1];v[2]=val[2]; }

void cVector3::setZero() { v[0]=v[1]=v[2]=0.0; }
bool cVector3::isZero() const { return v[0]==0.0 && v[1]==0.0 && v[2]==0.0; }
cVector3& cVector3::normalize() { double norm=(double)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);v[0]/=norm;v[1]/=norm;v[2]/=norm; return *this; }
double cVector3::norm() const { return (double)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
double cVector3::norm2() const { return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]; }
cVector3 cVector3::normalizedVersion() const { double norm=(double)sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);return cVector3(v[0]/norm,v[1]/norm,v[2]/norm); }
void cVector3::scale(const cVector3 &u) { v[0] *= u.v[0]; v[1] *= u.v[1]; v[2] *= u.v[2]; } 
void cVector3::get(double *x, double *y, double *z) const { *x = v[0]; *y = v[1]; *z = v[2]; }
void cVector3::set(double x, double y, double z) { v[0] = x; v[1] = y; v[2] = z; }
void cVector3::rotate(cVector3 *u, double theta) const {
  cVector3 e = *this / norm();
  *u = e * (e|*u) * (1-cos(theta)) + (e^*u) * sin(theta) + *u * cos(theta);
}
cVector3 cVector3::proj(const cVector3 &u) const {
  if (isZero())
    return cVector3(0.0, 0.0, 0.0);
  return operator*(operator|(u) / norm2());
}
void cVector3::orthogonalize(cVector3 *u) const { *u -= proj(*u); }


double cVector3::operator|(const cVector3& u) const { return v[0]*u.v[0]+v[1]*u.v[1]+v[2]*u.v[2]; }
cVector3 cVector3::operator^(const cVector3& u) const { return cVector3(v[1]*u.v[2]-v[2]*u.v[1],v[2]*u.v[0]-v[0]*u.v[2],v[0]*u.v[1]-v[1]*u.v[0]); }
cVector3 cVector3::operator+(const cVector3 &u) const { return cVector3(v[0]+u.v[0],v[1]+u.v[1],v[2]+u.v[2]); }
cVector3& cVector3::operator+=(const cVector3 &u) { v[0]+=u.v[0];v[1]+=u.v[1];v[2]+=u.v[2];return *this; }
cVector3 cVector3::operator-(const cVector3 &u) const { return cVector3(v[0]-u.v[0],v[1]-u.v[1],v[2]-u.v[2]); }
cVector3& cVector3::operator-=(const cVector3 &u) { v[0]-=u.v[0];v[1]-=u.v[1];v[2]-=u.v[2];return *this; }
cVector3 cVector3::operator*(double d) const { return cVector3(v[0]*d,v[1]*d,v[2]*d); }
cVector3& cVector3::operator*=(double d) { v[0]*=d;v[1]*=d;v[2]*=d;return *this; }
cVector3 cVector3::operator/(double d) const { return cVector3(v[0]/d,v[1]/d,v[2]/d); }
cVector3& cVector3::operator/=(double d) { v[0]/=d;v[1]/=d;v[2]/=d;return *this; }
cVector3 cVector3::operator-() { return cVector3(-v[0],-v[1],-v[2]); }
bool cVector3::operator==(const cVector3& u) const { return (v[0]==u.v[0])&&(v[1]==u.v[1])&&(v[2]==u.v[2]); }
bool cVector3::operator!=(const cVector3& u) const { return (v[0]!=u.v[0])||(v[1]!=u.v[1])||(v[2]!=u.v[2]); }
bool cVector3::operator<(const cVector3& u) const { if (v[0]>=u.v[0]) return false;if (v[1]>=u.v[1]) return false;if (v[2]>=u.v[2]) return false;return true; }

void cVector3::updateMinMax(cVector3 *min, cVector3 *max) const {
	
	if (v[0]<min->v[0]) min->v[0]=v[0]; else if (v[0]>max->v[0]) max->v[0]=v[0];
	if (v[1]<min->v[1]) min->v[1]=v[1]; else if (v[1]>max->v[1]) max->v[1]=v[1];
	if (v[2]<min->v[2]) min->v[2]=v[2]; else if (v[2]>max->v[2]) max->v[2]=v[2];
	
}

void cVector3::print() const {
	
	std::cout << " [" << v[0] << "]" << std::endl;
	std::cout << " [" << v[1] << "]" << std::endl;
	std::cout << " [" << v[2] << "]" << std::endl;
	std::cout << std::endl;
	
}

cVector3 operator*(double d, const cVector3 &u) {
	
	return cVector3(d*u[0], d*u[1], d*u[2]);
	
}

cVector3 cVector3::operator*(const cVector3 &u) const { return cVector3(v[0]*u.v[0],v[1]*u.v[1],v[2]*u.v[2]); }

std::ostream&	operator<<(std::ostream &out, const cVector3 &v) {
	out<<v[0]<<" "<<v[1]<<" "<<v[2];
	return out;
};
