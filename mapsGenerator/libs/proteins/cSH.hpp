#pragma once

#include "cVector3.hpp"
#include <iostream>
#include <fstream>
using namespace std;

// typedef double Real;
typedef float Real;
typedef struct complex {
    Real x, y;
} sComplex;








class cSH {

    constexpr static Real TOLERANCE = 2.2204460492503131e-16;
    constexpr static Real TOLERANCE_ROOT_4  = 1.2207031250000000e-04;
    // constexpr static double TOLERANCE = 2.2204460492503131e-8;
    // constexpr static double TOLERANCE_ROOT_4  = 1.2207031250000000e-02;



    //Wigner variables
    int j2, j3, m1, m2, m3, jmin, jmax, jnum, jp, Jn, ik, flag1, flag2, jmid;
    int num_types;
    Real xjmin, yjmin, yjmax, zjmax, xj, zj;
    Real *rs, *wl, *wu;
    Real *w3j0;
    Real *w3jm;
    Real *wignerConstant;


    int jindex(int j);
    Real a(int j);
    Real y(int j);
    Real x(int j);
    Real z(int j);
    void normw3j(Real *w3j);
    void fixsign(Real *w3j);
    void calculateWigner3jSymbols(int _j2, int _j3, int _m1, int _m2, int _m3, Real *w3j);

    void initWigner3j0(Real *w, int P, int P2);
    void initWigner3jm(Real *w, int P, int P2);
    void initWigner3jMatrices( int P);
    void initWignerConstant(Real *w, int P, int P2);

    void initWigner3jmConv6D(Real *w, int L);
    void initWigner3j0Conv6D(Real *w, int L);
    void initWignerConstantConv6D(Real *w, int L);


    //constants
    const Real fourPi = 4*3.14159265358979323846;

    //math arrays
    Real     *factorial;
    Real     **Fcoeff;
    Real    **LegPoly;
    sComplex    **Y_C;
    
    sComplex    *Yxy;

    //aux init
    void makeFactorial(int P);
    void makeSphCoeff(int P);

    //aux functions
    void computeLegendre(const Real xval);
    void computeFourier(const Real  b);



    //memory function
    void allocHalfArray( Real** &A, int P);
    void dellocHalfArray( Real** &A, int P);
    void allocComplexArray( sComplex** &M, int P);
    void dellocComplexArray(sComplex** &M, int P);

    

    //parameters of the class
    int order;
    
public:
    cSH();
    ~cSH();
    sComplex    ** const getSH() const {return Y_C;}

    sComplex    ****Coefficients;
    sComplex    ***FirstStruct;
    sComplex    ***SecondStruct;

    void init(int _P, int num_types = 167);
    void initConv6D(int L);
    static cVector3 cart2Sph( const cVector3 &v);
    void computeSpharmonic( const cVector3  &sv );

    void computeCoefficients( const cVector3  &sv, int nFourierShells, Real maxQ, int type, Real SIG = 1 );
    void computeCoefficientsTripleArray(sComplex*** CoefficientsToFill, const cVector3  &sv, int nFourierShells, Real deltaQ, Real SIG = 1);
    void computeBasisFunctions( const cVector3  &sv, int nFourierShells, Real maxQ);

    void fillCoefficientsWithZeros( );

    

    void      initWigner(int P, int P2); // P for the initial order, P2 - for the transformed one. For the simplicity, we can make them equal
    void      initWignerConv6D(int L);
    static    void makeWignerTheta (Real theta, Real ***WignerArray, int order);
    static    void makeWignerPhase (Real alpha,  sComplex *phase, int order);
    static    void Rotate(sComplex *const alphaPhase, Real ***const wignerD, sComplex *const gammaPhase,   sComplex **targetAlm,sComplex **const referenceAlm, int order);
    static    void writeSlaterMatrices(int order, sComplex *const WignerPhaseA, Real ***const WignerD, sComplex *const  WignerPhaseG, FILE * outputSlaterMatricesFile);
    
    static    int computeSphBessel(const Real x, Real *_SphBesPoly, int order);
    int       writeBesselMatrices(Real * const sphericalBessel, int P, int P2, FILE * outputBesselMatricesFile);
    void      translateZ(sComplex **targetAlm, sComplex **referenceAlm,  Real * const sphericalBessel, int P, int P2);
    void      Convolution6D(sComplex ***inputFunction, sComplex ***filterFunction, sComplex ***outputFunction, int L, int P, int L_max = 50);


};
