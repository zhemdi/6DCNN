#include "cSH.hpp"
#include <math.h>       /* sqrt */
#include "cMemoryManager.cpp" // it's a template
#include <cstring>

//#define SIG 1.0

cSH::cSH() {
    
    // math arrays
    factorial = NULL;
    Fcoeff = NULL;
    Yxy = NULL;
    LegPoly = NULL;
    Y_C = NULL;

    //Wigner arrays
    w3j0 = NULL;
    w3jm = NULL;

    wignerConstant = NULL;

    wl = NULL;
    wu = NULL;
    rs = NULL;


    
}

cSH::~cSH() {

    // math arrays
    if (factorial) delete [] factorial;

    if (Fcoeff) dellocHalfArray(Fcoeff, order);

    if (Yxy) delete [] Yxy;

    if (LegPoly) dellocHalfArray(LegPoly, order);

    if (Y_C) dellocComplexArray(Y_C, order);

    if (Coefficients) cMemoryManager::dellocQuadArray(Coefficients, this->num_types, order, order);

    // Wigner arrays
    
    if (wl) delete [] wl;
    if (wu) delete [] wu;
    if (rs) delete [] rs;
    

    if (w3j0)     cMemoryManager::dellocWigner3j0( w3j0 );
    if (w3jm)     cMemoryManager::dellocWigner3jm( w3jm );
    if (wignerConstant)     cMemoryManager::dellocWigner3jm( wignerConstant );

}

void cSH::init(int _P, int num_types) {

    order = _P;

    this->num_types = num_types;

    factorial=new Real[2*order+2];
    makeFactorial(order);

    allocHalfArray(Fcoeff, order);
    makeSphCoeff(order);

    allocComplexArray(Y_C, order);
    cMemoryManager::allocQuadArray(Coefficients, num_types, order, order);

    Yxy=new sComplex [order];
    allocHalfArray(LegPoly, order);

}


void cSH::initConv6D(int L) {

    order = L;

    
    factorial=new Real[2*order+2];
    makeFactorial(order);

    allocHalfArray(Fcoeff, order);
    makeSphCoeff(order);

    allocComplexArray(Y_C, order);
    //cMemoryManager::allocQuadArray(Coefficients, num_types, order, order);
    cMemoryManager::allocTripleArray(FirstStruct, order, order);
    cMemoryManager::allocTripleArray(SecondStruct, order, order);

    Yxy=new sComplex [order];
    allocHalfArray(LegPoly, order);

}

//FIXME - at low rho Maclaurin series should be used instead!!!
cVector3 cSH::cart2Sph( const cVector3 &v)
{
    cVector3 s;

    /* rho */
    s[0]=v.norm();

    if (s[0] < fabs(v[2])) s[0]=fabs(v[2]);

    /* alpha (Theta) */
    if (s[0]==0.0) s[1]=0.0;
    else s[1]=acos(v[2]/s[0]);

    /* beta (Phi) */
    if ((v[0]==0.0) && (v[1]==0.0)) s[2]=0.0;
    else s[2]=atan2(v[1], v[0]);

    return s;
} /* Cart2Sph */

void cSH::computeSpharmonic( const cVector3  &sv ) {

    int     m, n;
    Real    ytemp;

    computeLegendre(cos(sv[1]));
    computeFourier(sv[2]);

    //order expansion order
    for (n=0; n < order; n++) {
        for (m=0; m <= n; m++) {
            ytemp=  Fcoeff[n][m] * LegPoly[n][m];
            Y_C[n][m].x=ytemp * Yxy[m].x;
            Y_C[n][m].y=ytemp * Yxy[m].y;
        } /* for m */
    } /* for n */
} /* computeSpharmonic */



void cSH::fillCoefficientsWithZeros( ) {
    for (int i = 0; i < this->num_types; i++){
        for (int p = 0; p < order; p ++) {
            for(int l = 0; l < order; l++){
                for (int m = 0; m < l+1; m++){
                    Coefficients[i][p][l][m].x = 0.0;
                    Coefficients[i][p][l][m].y = 0.0;

                }
            }

        }
    }

    

   
} /* computeSpharmonic */


void cSH::computeBasisFunctions(const cVector3  &sv, int nFourierShells, Real maxQ){
    int     m, n, p;
    Real    ytemp, R;
    Real    **SphBesPoly;

    cMemoryManager::alloc2DArray(SphBesPoly, nFourierShells, order);

    R = sv[0];
    // std::cout << R << std::endl;

    computeLegendre(cos(sv[1]));
    computeFourier(sv[2]);

    // std::cout << "t=" << type << std::endl;
    //order expansion order
    for (p=0; p < nFourierShells; p++){
        Real q = p*maxQ/(nFourierShells - 1); // from 0 to maxQ
        computeSphBessel(R * q, SphBesPoly[p], nFourierShells);
        // std::cout << "p=" << p << std::endl;
        for (n=0; n < order; n++) {
            // std::cout << "n="  << n << std::endl;
            
            for (m=0; m <= n; m++) {
                // std::cout << "m="  << m << std::endl;
                ytemp=  Fcoeff[n][m] * LegPoly[n][m];
                
                Coefficients[0][p][n][m].x  =     std::pow(2/M_PI,1/2)*q*SphBesPoly[p][n]*ytemp * Yxy[m].x;
                Coefficients[0][p][n][m].y  =     std::pow(2/M_PI,1/2)*q*SphBesPoly[p][n]*ytemp * Yxy[m].y;
            } /* for m */
        } /* for n */
        // Coefficients[type][p][n][m].x = SphBesPoly[p][n]*Y_C[n][m].x;
        // Coefficients[type][p][n][m].y = SphBesPoly[p][n]*Y_C[n][m].y;

    }
    cMemoryManager::delloc2DArray(SphBesPoly);
}


void cSH::computeCoefficients( const cVector3  &sv, int nFourierShells, Real maxQ, int type, Real SIG ) {

    int     m, n, p;
    Real    ytemp, R;
    Real    **SphBesPoly;

    cMemoryManager::alloc2DArray(SphBesPoly, nFourierShells, order);

    R = sv[0];
    // std::cout << R << std::endl;

    computeLegendre(cos(sv[1]));
    computeFourier(sv[2]);
    Real factor1, factor2;
    Real factor0 = 4*M_PI*(std::pow(2 *M_PI *SIG*SIG,3/2));

    // std::cout << "t=" << type << std::endl;
    //order expansion order
    for (p=0; p < nFourierShells; p++){
        Real q = p*maxQ/(nFourierShells - 1); // from 0 to maxQ
        computeSphBessel(R * q, SphBesPoly[p], nFourierShells);
        factor1 = std::exp(-SIG*SIG*q*q/2)*factor0;
        // std::cout << "p=" << p << std::endl;
        for (n=0; n < order; n++) {
            // std::cout << "n="  << n << std::endl;
            factor2 = SphBesPoly[p][n]*factor1;
            for (m=0; m <= n; m++) {
                // std::cout << "m="  << m << std::endl;
                ytemp=  Fcoeff[n][m] * LegPoly[n][m];
                if (n%4 == 0){
                    Coefficients[type][p][n][m].x+=      factor2*ytemp * Yxy[m].x;
                    Coefficients[type][p][n][m].y+=    - factor2*ytemp * Yxy[m].y;
                }else if(n%4 == 1){
                    Coefficients[type][p][n][m].x+=    - factor2*ytemp * Yxy[m].y;
                    Coefficients[type][p][n][m].y+=    - factor2*ytemp * Yxy[m].x;
                }else if(n%4 == 2){
                    Coefficients[type][p][n][m].x+=    - factor2*ytemp * Yxy[m].x;
                    Coefficients[type][p][n][m].y+=      factor2*ytemp * Yxy[m].y;
                }else{
                    Coefficients[type][p][n][m].x+=      factor2*ytemp * Yxy[m].y;
                    Coefficients[type][p][n][m].y+=      factor2*ytemp * Yxy[m].x;
                }
            } /* for m */
        } /* for n */
        // Coefficients[type][p][n][m].x = SphBesPoly[p][n]*Y_C[n][m].x;
        // Coefficients[type][p][n][m].y = SphBesPoly[p][n]*Y_C[n][m].y;

    }
    cMemoryManager::delloc2DArray(SphBesPoly);
} /* computeSpharmonic */




void cSH::computeCoefficientsTripleArray(sComplex ***CoefficientsToFill, const cVector3  &sv, int nFourierShells, Real deltaQ, Real SIG ) {

    int     m, n, p;
    Real    ytemp, R;
    Real    **SphBesPoly;

    cMemoryManager::alloc2DArray(SphBesPoly, nFourierShells, order);

    R = sv[0];
    // std::cout << R << std::endl;

    computeLegendre(cos(sv[1]));
    computeFourier(sv[2]);
    Real factor1, factor2;
    Real factor0 = 4*M_PI*(std::pow(2 *M_PI *SIG*SIG,3/2));

    // std::cout << "t=" << type << std::endl;
    //order expansion order
    for (p=0; p < nFourierShells; p++){
        Real q = p*deltaQ; // from 0 to maxQ
        computeSphBessel(R * q, SphBesPoly[p], nFourierShells);
        factor1 = std::exp(-SIG*SIG*q*q/2)*factor0;
        // std::cout << "p=" << p << std::endl;
        for (n=0; n < order; n++) {
            // std::cout << "n="  << n << std::endl;
            factor2 = SphBesPoly[p][n]*factor1;
            for (m=0; m <= n; m++) {
                // std::cout << "m="  << m << std::endl;
                ytemp=  Fcoeff[n][m] * LegPoly[n][m];
                if (n%4 == 0){
                    CoefficientsToFill[p][n][m].x+=      factor2*ytemp * Yxy[m].x;
                    CoefficientsToFill[p][n][m].y+=    - factor2*ytemp * Yxy[m].y;
                }else if(n%4 == 1){
                    CoefficientsToFill[p][n][m].x+=    - factor2*ytemp * Yxy[m].y;
                    CoefficientsToFill[p][n][m].y+=    - factor2*ytemp * Yxy[m].x;
                }else if(n%4 == 2){
                    CoefficientsToFill[p][n][m].x+=    - factor2*ytemp * Yxy[m].x;
                    CoefficientsToFill[p][n][m].y+=      factor2*ytemp * Yxy[m].y;
                }else{
                    CoefficientsToFill[p][n][m].x+=      factor2*ytemp * Yxy[m].y;
                    CoefficientsToFill[p][n][m].y+=      factor2*ytemp * Yxy[m].x;
                }
            } /* for m */
        } /* for n */
        

    }
    cMemoryManager::delloc2DArray(SphBesPoly);
}

void cSH::computeLegendre(const Real xval) {

    int Li, Lj;
    Real negterm, oddfact, nextodd, sqroot, sqrtterm;

    negterm=1.0;
    oddfact=1.0;
    nextodd=1.0;
    sqroot=sqrt(1.0 - xval*xval);
    sqrtterm=1.0;
    for (Li=0;Li < order;Li++) {
        LegPoly[Li][Li]=negterm*oddfact*sqrtterm;
        negterm=-negterm;
        oddfact *= nextodd;
        nextodd += 2.0;
        sqrtterm *= sqroot;
        if (Li < order-1) {
            LegPoly[Li+1][Li]=xval * (Real)(2*Li+1) * LegPoly[Li][Li];
            for (Lj=Li+2;Lj < order;Lj++) {
                LegPoly[Lj][Li]=(xval*(Real)(2*Lj-1)*LegPoly[Lj-1][Li] -
                                 (Real)(Lj+Li-1)*LegPoly[Lj-2][Li])/(Real)(Lj-Li);
            }
        }
    }
}

void cSH::computeFourier(const Real  b) {

    int   m;

    for (m=0; m < order; m++) {
        Yxy[m].x=cos((Real) m * b);
        Yxy[m].y=sin((Real) m * b);
    } /* for m */

} /* computeFourier */

void cSH::makeFactorial(int P) {

    int n;
    factorial[0]=1.0;
    for (n=1; n < 2 * (P + 1); n++) {
        factorial[n]=(Real) n *factorial[n - 1];
    } /* for n */
}

void cSH::makeSphCoeff(int P) {
    int l, m;
    for (l=0; l < P; l++) {
        Real coeff = (2*l+1)/fourPi;
        for (m=0; m <= l; m++) {
            Fcoeff[l][m]= sqrt(coeff*factorial[l-m]/factorial[l+m]);
        } /* for m */
    } /* for l */
}
void cSH::allocHalfArray( Real** &A, int P) {

    A=new Real* [P];
    for (int i=0;i<P;i++)
        *(A+i)=new Real[i+1];
}

void cSH::dellocHalfArray( Real** &A, int P) {

    for (int i=0;i<P;i++)
        delete [] A[i];
    delete [] A;
}

void cSH::allocComplexArray( sComplex** &M, int P) {

    M=(sComplex**) new sComplex* [P];
    for (int i=0;i<P;i++)
        *(M+i)=new sComplex[i+1];
}

void cSH::dellocComplexArray(sComplex** &M, int P) {

    for (int i=0;i<P;i++)
        delete [] M[i];
    delete [] M;
}

// static
void cSH::makeWignerTheta (Real theta, Real ***WignerArray, int order)  {

    int         i, j, k;
    Real     cosTheta, sinTheta, cosTheta2, sinTheta2, tanTheta2, d1_0_0, d1_1_1;
    Real     phase;

    cosTheta=cos (theta);
    sinTheta=sin (theta);
    cosTheta2=cos (theta * 0.5);
    sinTheta2=sin (theta * 0.5);
    tanTheta2=sinTheta2 / cosTheta2;

    /* P==0 */
    if (order > 0) {

        WignerArray[0][0][0]=1.;
    }

    /* P==1 */
    if (order > 1) {

        WignerArray[1][0][1 + 0]=cosTheta;
        WignerArray[1][1][1 - 1]=sinTheta2 * sinTheta2;
        WignerArray[1][1][1 + 0]=-sinTheta / sqrt (2.);
        WignerArray[1][1][1 + 1]=cosTheta2 * cosTheta2;
        WignerArray[1][0][1 + 1]=-WignerArray[1][1][1 + 0];
        WignerArray[1][0][1 - 1]= WignerArray[1][1][1 + 0];
    }

    d1_0_0=WignerArray[1][0][1 + 0];
    d1_1_1=WignerArray[1][1][1 + 1];

    /* P > 2 */
    for (i=2; i < order; i++) {

        Real two_i_m_1=i + i - 1.;
        Real sq_i=i * i;
        Real sq_i_m_1=(i - 1.) * (i - 1.);

        /* j > 0, -j <= k <= j */
        for (j=0; j <= i - 2; j++) {

            Real sq_j=j * j;

            for (k=-j; k <= j; k++) {

                Real sq_k=k * k;
                Real a =
                (i * two_i_m_1) / sqrt ((sq_i - sq_j) * (sq_i - sq_k));
                Real b=(d1_0_0 - ((j * k) / (i * (i - 1.))));
                Real c=sqrt ((sq_i_m_1 - sq_j) * (sq_i_m_1 - sq_k)) /
                ((i - 1.) * two_i_m_1);

                WignerArray[i][j][i + k] =
                a * (b * WignerArray[i - 1][j][(i - 1) + k] -
                     c * WignerArray[i - 2][j][(i - 2) + k]);
            }
        }

        /* last two diagonal terms */
        WignerArray[i][i][i + i]=d1_1_1 *
        WignerArray[i - 1][i - 1][(i - 1) + (i - 1)];

        WignerArray[i][i - 1][i + i - 1]=(i * d1_0_0 - i + 1.) *
        WignerArray[i - 1][i - 1][(i - 1) + (i - 1)];

        /* last column terms */
        for (k=i; k >= 1 - i; k--) {

            WignerArray[i][i][i + k - 1]=-sqrt ((i + k) / (i - k + 1.)) * tanTheta2 *
            WignerArray[i][i][i + k];
        }

        /* penultimate column */
        for (k=i - 1; k >= 2 - i; k--) {

            Real a=sqrt (((Real)i+k)/(((Real)i+i)*(i-k+1.)));

            WignerArray[i][i - 1][i + k - 1]=(i*cosTheta-k+1.) * a *
            WignerArray[i][i][i + k] / d1_1_1;
        }

        /* extra diagonal terms (|k| > j) */
        for (k=1; k <= i; k++) {
            for (j=0; j < k; j++) {
                phase=1.0 - 2.0 * (Real) (0x0001 & (j + k)); // -1^ (j+k)
                                                             // todo: remove it!
                                                             //                phase=pow(-1.0,(int) j+k);
                WignerArray[i][j][i + k]=phase * WignerArray[i][k][i + j];
                WignerArray[i][j][i - k]=WignerArray[i][k][i - j];
            }
        }
    }

    //   printW(P, WignerD);
}

//static
void cSH::makeWignerPhase (Real alpha,  sComplex *phase, int order) {


    for (int n=0; n < order; n++) {

        phase[n].x=cos((Real) n * alpha);
        phase[n].y=-sin((Real) n * alpha);

    } /* for n */
}

void cSH::writeSlaterMatrices(int order, sComplex *const WignerPhaseA, Real ***const WignerD, sComplex *const  WignerPhaseG, FILE * outputSlaterMatricesFile) {
    // FILE* binFile = fopen(outputSlaterMatrices.c_str(),"wb");
    Real ElementX_pos, ElementY_pos, ElementX_neg, ElementY_neg, tempX_pos, tempY_pos, tempX_neg, tempY_neg, amp, AbsValue;
    
    for (int l=0; l < order; l++) { // l loop
        for (int m1=0; m1 <= l; m1++) { // m1 loop
            amp=WignerD[l][m1][l];
            ElementX_pos =  amp*WignerPhaseG[m1].x*WignerPhaseA[0].x;
            ElementY_pos =  amp*WignerPhaseG[m1].y*WignerPhaseA[0].x;
            // std::cout << l << " " << m1 << " " << 0 << ":   " <<  ElementX_pos << "+" << ElementY_pos  << "j" << std::endl;
            // std::cout << l << " " << m1 << " " << 0 << ":   " << WignerD[l][m1][l] << std::endl;
            fwrite(&ElementX_pos,sizeof(Real), 1 ,outputSlaterMatricesFile);
            fwrite(&ElementY_pos,sizeof(Real), 1 ,outputSlaterMatricesFile);
            for (int m2=1; m2 <= l; m2++) { // m2 loop
                
                tempX_pos = WignerPhaseG[m1].x*WignerPhaseA[m2].x - WignerPhaseG[m1].y*WignerPhaseA[m2].y;
                tempY_pos = (WignerPhaseG[m1].x*WignerPhaseA[m2].y + WignerPhaseG[m1].y*WignerPhaseA[m2].x);
                tempX_neg = WignerPhaseG[m1].x*WignerPhaseA[m2].x + WignerPhaseG[m1].y*WignerPhaseA[m2].y;
                tempY_neg = (-WignerPhaseG[m1].x*WignerPhaseA[m2].y + WignerPhaseG[m1].y*WignerPhaseA[m2].x);
                ElementX_pos  = ( WignerD[l][m1][l+m2] )*tempX_pos;
                ElementY_pos  = ( WignerD[l][m1][l+m2] )*tempY_pos;
                ElementX_neg = pow(-1,m2)*( WignerD[l][m1][l-m2] )*tempX_neg;//pow(-1,m2)*
                ElementY_neg = pow(-1,m2)*( WignerD[l][m1][l-m2] )*tempY_neg;
                // AbsValue = sqrt(ElementX*ElementX + ElementY*ElementY);
                // std::cout << l << " " << m1 << " " << m2 << ":   " <<  ElementX << "+" << ElementY  << "j" << std::endl;//  "  << tempX << "+" << tempY << "j "  << WignerD[l][m1][l+m2] << "+" << WignerD[l][m1][l - m2] << "j" << std::endl;
                // std::cout << l << " " << m1 << " " <<  m2 << ": " << ElementX_pos << "+" << ElementY_pos << "j" <<std::endl;
                // std::cout << l << " " << m1 << " " << -m2 << ": " << pow(-1,m2)*ElementX_neg << "+" << pow(-1,m2)*ElementY_neg << "j" <<std::endl;
                // std::cout << l << " " << m1 << " " << m2 << ": " << WignerD[l][m1][l+m2] << std::endl;
                // std::cout << l << " " << m1 << " " << -m2 << ": " << WignerD[l][m1][l-m2] << std::endl;
                fwrite(&ElementX_neg,sizeof(Real), 1 ,outputSlaterMatricesFile);
                fwrite(&ElementY_neg,sizeof(Real), 1 ,outputSlaterMatricesFile);
                fwrite(&ElementX_pos,sizeof(Real), 1 ,outputSlaterMatricesFile);
                fwrite(&ElementY_pos,sizeof(Real), 1 ,outputSlaterMatricesFile);
                // fwrite(&AbsValue,sizeof(Real), 1 ,outputSlaterMatricesFile);
            }
        }
    }
    /*
    for (int l=0; l < order; l++) { // l loop
        for (int m1=0; m1 <= l; m1++) { // m1 loop
            for (int m2=-l; m2 <= -1; m2++) { // m2 loop
                
                tempX = WignerPhaseG[m1].x*WignerPhaseA[abs(m2)].x - WignerPhaseG[m1].y*WignerPhaseA[abs(m2)].y;
                tempY = WignerPhaseG[m1].x*WignerPhaseA[abs(m2)].y + WignerPhaseG[m1].y*WignerPhaseA[abs(m2)].x;
                ElementX = ( WignerD[l][m1][l+m2])*tempX;
                ElementY = ( -WignerD[l][m1][l + m2])*tempY;
                std::cout << l << " " << m1 << " " << m2 << ":   " <<  ElementX << "+" << ElementY  << "j" << std::endl;//  "  << tempX << "+" << tempY << "j "  << WignerD[l][m1][l+m2] << "+" << WignerD[l][m1][l - m2] << "j" << std::endl;
                fwrite(&ElementX,sizeof(Real), 1 ,outputSlaterMatricesFile);
                fwrite(&ElementY,sizeof(Real), 1 ,outputSlaterMatricesFile);
            }
            
            amp=WignerD[l][m1][l];
            ElementX =  amp*WignerPhaseG[m1].x;
            ElementY =  -amp*WignerPhaseG[m1].y;
            std::cout << l << " " << m1 << " " << 0 << ":   " <<  ElementX << "+" << ElementY  << "j" << std::endl;
            fwrite(&ElementX,sizeof(Real), 1 ,outputSlaterMatricesFile);
            fwrite(&ElementY,sizeof(Real), 1 ,outputSlaterMatricesFile);
            for (int m2=1; m2 <= l; m2++) { // m2 loop
                
                tempX = WignerPhaseG[m1].x*WignerPhaseA[m2].x - WignerPhaseG[m1].y*WignerPhaseA[m2].y;
                tempY = WignerPhaseG[m1].x*WignerPhaseA[m2].y + WignerPhaseG[m1].y*WignerPhaseA[m2].x;
                ElementX = ( WignerD[l][m1][l+m2])*tempX;
                ElementY = ( -WignerD[l][m1][l + m2])*tempY;
                std::cout << l << " " << m1 << " " << m2 << ":   " <<  ElementX << "+" << ElementY  << "j" << std::endl;//  "  << tempX << "+" << tempY << "j "  << WignerD[l][m1][l+m2] << "+" << WignerD[l][m1][l - m2] << "j" << std::endl;
                fwrite(&ElementX,sizeof(Real), 1 ,outputSlaterMatricesFile);
                fwrite(&ElementY,sizeof(Real), 1 ,outputSlaterMatricesFile);
            }
        }
    }
    */
}

/* alpha is applied first, followed by beta &&  finally gamma */
//static
void cSH::Rotate( sComplex *const alphaPhase,  Real ***const wignerD,  sComplex *const gammaPhase,   sComplex **targetAlm,sComplex **const referenceAlm, int order){

    for (int l=0; l < order; l++) { // l loop
        for (int m=0; m <= l; m++) { // m loop

            /* 0-order term */
            Real amp=wignerD[l][m][l]; // swap Wigner indices

            Real phaseX = amp*referenceAlm[l][0].x;
            Real phaseY = amp*referenceAlm[l][0].y;

            for (int j=1; j<=l; j++){ /* high-order terms */

                /* positive j */
                phaseX += wignerD[l][m][l+j] * (referenceAlm[l][j].x*alphaPhase[j].x - referenceAlm[l][j].y*alphaPhase[j].y);
                phaseY += wignerD[l][m][l+j] * (referenceAlm[l][j].x*alphaPhase[j].y + referenceAlm[l][j].y*alphaPhase[j].x);


                phaseX +=  pow(-1,l+j)*wignerD[l][m][l - j] * (referenceAlm[l][j].x*alphaPhase[j].x - referenceAlm[l][j].y*alphaPhase[j].y);
                phaseY += -pow(-1,l+j)*wignerD[l][m][l - j] * (referenceAlm[l][j].x*alphaPhase[j].y + referenceAlm[l][j].y*alphaPhase[j].x);

            }

            /* &&  finally multiply everything with the gamma phase */
            targetAlm[l][m].x =  phaseX*gammaPhase[m].x - phaseY*gammaPhase[m].y;
            targetAlm[l][m].y =  phaseX*gammaPhase[m].y + phaseY*gammaPhase[m].x;
        }
    }
}

// not the best stable, but should be OK for the order up to ~ 80
// static
int  cSH::computeSphBessel(const Real x, Real *_SphBesPoly, int P) {

    const int lmax = P-1;

    int sign = 1;

    // special cases
    if(lmax < 0 /*|| x < 0.0*/) {
        std::cerr<<"bad order || argument in computeSphBessel"<<std::endl;
//        info->jlog["Error"]="bad order || argument in computeSphBessel";
        //        exit(0);
        return 1;
    }
    else if(x == 0.0) {
        int j;
        for(j=1; j<=lmax; j++) _SphBesPoly[j] = 0.0;
        _SphBesPoly[0] = 1.0;
        return 0;
    }
    else if(fabs(x) < 2.0*TOLERANCE_ROOT_4) {
        /* first two terms of Taylor series */
        Real inv_fact = 1.0;  /* 1/(1 3 5 ... (2l+1)) */
        Real x_l      = 1.0;  /* x^l */
        int l;
        for(l=0; l<=lmax; l++) {
            _SphBesPoly[l]  = x_l * inv_fact;
            _SphBesPoly[l] *= 1.0 - 0.5*x*x/(2.0*l+3.0);
            inv_fact /= 2.0*l+3.0;
            x_l      *= x;
        }
        return 0;
    }
    else {

        if (x < 0.0) {
            sign = -1;
        }

        /* Steed/Barnett algorithm [Comp. Phys. Comm. 21, 297 (1981)] */
        Real x_inv = sign * 1.0/x;
        Real W = 2.0*x_inv;
        Real F = 1.0;
        Real FP = (lmax+1.0) * x_inv;
        Real B = 2.0*FP + x_inv;
        Real end = B + 20000.0*W;
        Real D = 1.0/B;
        Real del = -D;

        FP += del;

        /* continued fraction */
        do {
            B += W;
            D = 1.0/(B-D);
            del *= (B*D - 1.);
            FP += del;
            if(D < 0.0) F = -F;
            if(B > end) {
                std::cerr<<"max number of iterations reached in computeSphBessel"<<std::endl;
                return 1;
            }
        }
        while(fabs(del) >= fabs(FP) * TOLERANCE);

        FP *= F;

        if(lmax > 0) {
            /* downward recursion */
            Real XP2 = FP;
            Real PL = lmax * x_inv;
            int L  = lmax;
            int LP;
            _SphBesPoly[lmax] = F;
            for(LP = 1; LP<=lmax; LP++) {
                _SphBesPoly[L-1] = PL * _SphBesPoly[L] + XP2;
                FP = PL*_SphBesPoly[L-1] - _SphBesPoly[L];
                XP2 = FP;
                PL -= x_inv;
                --L;
            }
            F = _SphBesPoly[0];
        }

        /* normalization */
        W = x_inv / hypot(FP, F);
        _SphBesPoly[0] = W*F;
        if(lmax > 0) {
            int L;
            for(L=1; L<=lmax; L++) {
                _SphBesPoly[L] *= W*pow(sign,L);
            }
        }

        return 0;
    }
}

//For Wigner 3-j symbols

void cSH::initWigner(int P, int P2) {

    int nElementsWigner0 = cMemoryManager::allocWigner3j0(w3j0, P);
    
    int nElementsWignerm =  cMemoryManager::allocWigner3jm(w3jm, P);
    // w3jm = new Real[nElementsWignerm];
    initWigner3jMatrices(P);
    initWigner3j0(w3j0, P, P2);
    initWigner3jm(w3jm, P, P2);

    cMemoryManager::allocWigner3jm(wignerConstant, P);
    // wignerConstant = new Real[nElementsWignerm];
    initWignerConstant(wignerConstant, P, P2);

    std::cout<<"Number of reduced 3j-symbols "<<nElementsWigner0<<std::endl;
    std::cout<<"Number of precomputed Clebsch Gordan coefficients "<<nElementsWignerm<<std::endl;

}

void cSH::initWignerConv6D(int L) {

    int nElementsWigner0 = cMemoryManager::allocWigner3j0(w3j0, L);
    
    int nElementsWignerm =  cMemoryManager::allocWigner3jmConv6D(w3jm, L);
    // w3jm = new Real[nElementsWignerm];
    initWigner3jMatrices(L);
    initWigner3j0Conv6D(w3j0, L);
    initWigner3jmConv6D(w3jm, L);

    cMemoryManager::allocWigner3j0(wignerConstant, L);
    // wignerConstant = new Real[nElementsWignerm];
    initWignerConstantConv6D(wignerConstant, L);

    std::cout<<"Number of reduced 3j-symbols "<<nElementsWigner0<<std::endl;
    std::cout<<"Number of precomputed Clebsch Gordan coefficients "<<nElementsWignerm<<std::endl;

}

void cSH::initWigner3jMatrices(int P) {

    int max_jnum = 2*P;
    wl = new Real [max_jnum];
    wu = new Real [max_jnum];
    rs = new Real [max_jnum];

}

int cSH::jindex(int j){
    return j-jmin;
}

Real cSH::a(int j){
    Real a=(Real(j*j)-Real(j2-j3)*Real(j2-j3))*(Real(j2+j3+1)*Real(j2+j3+1)-Real(j*j))*(Real(j*j)-Real(m1*m1));
    return sqrt(a);
}

Real cSH::y(int j){
    return -(2*j+1)*(m1*(j2*(j2+1)-j3*(j3+1))-(m3-m2)*j*(j+1));
}

Real cSH::x(int j){
    return j*a(j+1);
}

Real cSH::z(int j){
    return (j+1)*a(j);
}

void cSH::normw3j(Real *w3j){
    Real norm=0.0;
    int u;
    for(int j=jmin; j<=jmax; j++){
        u=jindex(j);
        norm = norm + (2*j+1)*w3j[u]*w3j[u];
    }
    for(int j=0; j<jnum; j++){
        w3j[j]=w3j[j]/sqrt(norm);
        //printf("%4.34f %d \n", w3j[j], j+jmin-1);
    }
}

void cSH::fixsign(Real *w3j){
    if ((w3j[jindex(jmax)] < 0.0 &&  pow(-1,j2-j3+m2+m3) > 0) || (w3j[jindex(jmax)] > 0.0 &&  pow(-1, j2-j3+m2+m3) < 0) ){
        for(int u=0; u<jnum; u++){
            w3j[u]= -w3j[u];
        }
    }
}

// iterates on j1
void cSH::calculateWigner3jSymbols(int _j2, int _j3, int _m1, int _m2, int _m3, Real *w3j) {

    Real wnmid, wpmid, scalef, denom;

    j2=_j2; j3=_j3; m1=_m1; m2=_m2; m3=_m3;
    flag1 = 0;
    flag2 = 0;
    scalef = 1000.0;
    jmin = std::max(abs(j2-j3), abs(m1));
    jmax = j2 + j3;
    jnum = jmax - jmin + 1;
    if (abs(m2) > j2 || abs(m3) > j3){
        std::cout<<"not good 1"<<std::endl;
        return;
    }
    else {
        if(m1+m2+m3!=0) {
            std::cout<<"not good 2"<<std::endl;
            return;
        }
        else{
            if(jmax<jmin){
                std::cout<<"not good 3"<<std::endl;
                return;
            }
        }
    }


    // Only one term is present

    if (jnum == 1) {
        w3j[0] = 1.0/sqrt(2.0*jmin+1.0);
        if ((w3j[0] < 0.0 &&  pow((-1),(j2-j3+m2+m3)) > 0) || (w3j[0] > 0.0 &&
                                                               pow((-1),(j2-j3+m2+m3)) < 0) ){
            w3j[0] = -w3j[0];
            fixsign(w3j);
            normw3j(w3j);
            return;
        }
    }

    //    Calculate lower non-classical values for [jmin, jn]. If the second term
    //    can not be calculated because the recursion relationsips give rise to a
    //    1/0, then set flag1 to 1.  If all m's are zero, then this is not a problem
    //    as all odd terms must be zero.


    xjmin = x(jmin); // 7.74
    yjmin = y(jmin); // 6


    if (m1 == 0 &&  m2 == 0 &&  m3 == 0){
        wl[jindex(jmin)] = 1.0;
        wl[jindex(jmin+1)] = 0.0;
        Jn = jmin+1;
    }

    else{
        if(yjmin==0.0){      // The second terms is either zero
            if (xjmin==0.0){      // || undefined
                flag1=1;
                Jn=jmin;
            }
            else{
                wl[jindex(jmin)]=1.0;
                wl[jindex(jmin+1)]=0.0;
                Jn=jmin+1;
            }
        }
        else{
            if(xjmin*yjmin>=0){
                wl[jindex(jmin)] = 1.0;
                wl[jindex(jmin+1)] = -yjmin / xjmin;
                Jn = jmin+1;
            }
            else{
                rs[jindex(jmin)] = -xjmin / yjmin;
                Jn = jmax;
                for(int j=jmin+1; j<=jmax-1; j++){
                    denom =  y(j) + z(j)*rs[jindex(j-1)];
                    xj=x(j);
                    if (fabs(xj) > fabs(denom) || xj*denom >= 0.0 || denom == 0.0){
                        Jn = j-1;
                        break;
                    }
                    else{
                        rs[jindex(j)] = -xj / denom;
                    }
                }

                wl[jindex(Jn)]=1.0;

                for(int k=1; k<=Jn-jmin; k++){
                    wl[jindex(Jn-k)] = wl[jindex(Jn-k+1)] * rs[jindex(Jn-k)];
                }
                if(Jn==jmin){
                    wl[jindex(jmin+1)] = -yjmin/xjmin;
                    Jn=jmin+1;
                }
            }
        }
    }
    if(Jn==jmax){
        for(int u=0; u<jnum; u++){
            w3j[u]=wl[u];
        }
        fixsign(w3j);
        normw3j(w3j);
        return;

    }


    //    Calculate upper non-classical values for [jp, jmax].
    //    If the second last term can not be calculated because the
    //    recursion relations give a 1/0, then set flag2 to 1.
    //    (Note, I don't think that this ever happens).

    yjmax = y(jmax);
    zjmax = z(jmax);

    if (m1 == 0 &&  m2 == 0 &&  m3 == 0){

        wu[jindex(jmax)] = 1.0;
        if (jmax > jmin) wu[jindex(jmax-1)] = 0.0;
        // std::cout << jindex(jmax-1) << std::endl;
        jp = jmax-1;
    }
    else{
        if(yjmax == 0.0){
            if(zjmax == 0.0){
                flag2=1;
                jp=jmax;
            }
            else{
                wu[jindex(jmax)] = 1.0;
                wu[jindex(jmax-1)] = - yjmax / zjmax;
                jp = jmax-1;
            }
        }
        else{
            if(yjmax * zjmax >= 0.0){
                wu[jindex(jmax)] = 1.0;
                wu[jindex(jmax-1)] = - yjmax / zjmax;
                jp = jmax-1;
            }
            else{
                rs[jindex(jmax)] = -zjmax / yjmax;

                jp = jmin;
                for(int j=jmax-1; j>=Jn; j--){
                    denom = y(j) + x(j)*rs[jindex(j+1)];
                    zj = z(j);
                    if(fabs(zj) > fabs(denom) || zj*denom >= 0.0 || denom == 0.0){
                        jp = j+1;
                        break;
                    }
                    else{
                        rs[jindex(j)] = -zj / denom;
                    }
                }
                wu[jindex(jp)] = 1.0;
                for(int k=1; k<=jmax-jp; k++){
                    wu[jindex(jp+k)] = wu[jindex(jp+k-1)]*rs[jindex(jp+k)];
                }
                if (jp == jmax){
                    wu[jindex(jmax-1)] = - yjmax / zjmax;
                    jp = jmax-1;
                }
            }
        }
    }

    //    Calculate classical terms for [jn+1, jp-1] using standard three
    //    term recursion relationship. Start from both jn &&  jp &&  stop at the
    //    midpoint. If flag1 is set, then perform the recursion solely from high to
    //    low values. If flag2 is set, then perform the recursion solely from low to high.

    if (flag1 == 0){
        jmid = (Jn + jp)/2;


        for(int j=Jn; j<=jmid - 1; j++){

            wl[jindex(j+1)] = - (z(j)*wl[jindex(j-1)] +y(j)*wl[jindex(j)]) / x(j);
            if (fabs(wl[jindex(j+1)]) > 1.0){
                for(int u=jindex(jmin); u<=jindex(j+1);u++){
                    wl[u] = wl[u] / scalef;
                }
            }
            if (fabs(wl[jindex(j+1)]/wl[jindex(j-1)]) < 1.0 &&  wl[jindex(j+1)]!= 0.0){
                jmid = j+1;
                break;
            }
        }

        wnmid = wl[jindex(jmid)];

        if (jmid > jmin &&  fabs(wnmid/wl[jindex(jmid-1)]) < 0.000001 &&    wl[jindex(jmid-1)] != 0.0){

            wnmid = wl[jindex(jmid-1)];
            jmid = jmid - 1;
        }

        for(int j=jp; j>=jmid+1; j--){
            wu[jindex(j-1)] = - (x(j)*wu[jindex(j+1)] + y(j)*wu[jindex(j)] ) / z(j);
            if (fabs(wu[jindex(j-1)]) > 1.0){
                for(int l=jindex(j-1); l<=jindex(jmax); l++){
                    wu[l] = wu[l] / scalef;
                }
            }
        }

        wpmid = wu[jindex(jmid)];


        if (jmid == jmax){
            for(int u=0; u<jnum; u++){
                w3j[u] = wl[u];
            }
        }
        else{
            if(jmid == jmin){
                for(int u=0; u<jnum; u++){
                    w3j[u] = wu[u];
                }
            }
            else{
                for(int u=0; u<=jindex(jmid); u++){
                    w3j[u] = wl[u] * wpmid / wnmid;
                }
                for(int u=jindex(jmid+1); u<=jindex(jmax); u++){
                    w3j[u] = wu[u];
                }
            }
        }
    }
    else{
        if(flag1==1 &&  flag2==0){
            for(int j=jp; j>=jmin+1; j--){
                wu[jindex(j-1)] = - (x(j)*wu[jindex(j+1)] + y(j)*wu[jindex(j)]) / z(j);
                if (fabs(wu[jindex(j-1)]) > 1){
                    for(int u=jindex(j-1); u<jindex(jmax); u++){
                        wu[u] = wu[u] / scalef;
                    }
                }
            }
            for(int u=0; u<jnum; u++){
                w3j[u]=wu[u];
            }
        }
        else{
            if(flag2 == 1 &&  flag1 == 0){
                for(int j=Jn; j<=jp-1; j++){
                    wu[jindex(j+1)] = - (z(j)*wl[jindex(j-1)] + y(j)*wl[jindex(j)]) / x(j);
                    if (fabs(wl[jindex(j+1)]) > 1){
                        for(int u=jindex(jmin); u<=jindex(j+1); u++){
                            wl[u] = wl[u] / scalef;
                        }
                    }
                }
                for(int u=0; u<jnum; u++){
                    w3j[u]=wl[u];
                }
            }
            else{
                if(flag1 == 1 &&  flag2 == 1){
                    std::cout<<"Fatal Error --- Wigner3j \n Can not calculate function for input values, both flag1 &&  flag 2 are set."<<std::endl;
                }
            }
        }
    }
    fixsign(w3j);
    normw3j(w3j);
    return;
}

void cSH::initWigner3j0(Real *w, int P, int P2) {

    int memory = 0;
    for (int j2=0; j2<P; ++j2) { // l

        for (int j3=0; j3<P; j3+=4) { // p
            int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, P2-1);
            int _jnum = _jmax - _jmin + 1; // k
            calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            memory+= _jnum;
        }

        for (int j3=1; j3<P; j3+=4) { // p
            int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, P2-1);
            int _jnum = _jmax - _jmin + 1; // k
            calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            memory+= _jnum;
        }

        for (int j3=2; j3<P; j3+=4) { // p
            int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, P2-1);
            int _jnum = _jmax - _jmin + 1; // k
            calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            memory+= _jnum;
        }

        for (int j3=3; j3<P; j3+=4) { // p
            int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, P2-1);
            int _jnum = _jmax - _jmin + 1; // k
            calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            memory+= _jnum;
        }

            //    for (int j3=0; j3<P; ++j3) { // p
            //        int _jmin = abs(j2-j3);
            //        int _jmax = j2 + j3;
            //        int _jnum = _jmax - _jmin + 1; // k
            //        calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            //        memory+= _jnum;
            //    }
    }
    std::cout << memory << std::endl;

}

void cSH::initWigner3jm(Real *w, int P, int P2) {
    int memory = 0;
    for (int j2=0; j2<P; ++j2) { // l
        for (int m1=0; m1<=j2; ++m1) { // m

            for (int j3=0; j3<P; j3+=4) { // p
                int _jmin = std::max(abs(j2-j3), abs(m1));
                //                int _jmin = abs(j2-j3);
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1; // k
                calculateWigner3jSymbols(j2,j3,m1,-m1,0,w+memory);

                memory += _jnum;
            }

            for (int j3=1; j3<P; j3+=4) { // p
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1; // k
                calculateWigner3jSymbols(j2,j3,m1,-m1,0,w+memory);

                memory += _jnum;
            }

            for (int j3=2; j3<P; j3+=4) { // p
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1; // k
                calculateWigner3jSymbols(j2,j3,m1,-m1,0,w+memory);

                memory += _jnum;
            }

            for (int j3=3; j3<P; j3+=4) { // p
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1; // k
                calculateWigner3jSymbols(j2,j3,m1,-m1,0,w+memory);

                memory += _jnum;
            }
            //            for (int j3=0; j3<P; ++j3, ++index) { // p
            //                int _jmin = std::max(abs(j2-j3), abs(m1));
            //                int _jmax = j2 + j3;
            //                int _jnum = _jmax - _jmin + 1; // k
            //                calculateWigner3jSymbols(j2,j3,m1,-m1,0,w+memory);
            //
            //                memory += _jnum;
            //            }
        }
    }
    std::cout << memory << std::endl;
}



void cSH::initWigner3j0Conv6D(Real *w, int L) {

    int memory = 0;
    for (int j2=0; j2<L; ++j2) { // l

        for (int j3=0; j3<L; ++j3) { // p
            int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, L-1);
            int _jnum = _jmax - _jmin + 1; // k
            calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            memory+= _jnum;
        }

        // for (int j3=1; j3<P; j3+=4) { // p
        //     int _jmin = abs(j2-j3);
        //     int _jmax = std::min<int>(j2 + j3, P2-1);
        //     int _jnum = _jmax - _jmin + 1; // k
        //     calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
        //     memory+= _jnum;
        // }

        // for (int j3=2; j3<P; j3+=4) { // p
        //     int _jmin = abs(j2-j3);
        //     int _jmax = std::min<int>(j2 + j3, P2-1);
        //     int _jnum = _jmax - _jmin + 1; // k
        //     calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
        //     memory+= _jnum;
        // }

        // for (int j3=3; j3<P; j3+=4) { // p
        //     int _jmin = abs(j2-j3);
        //     int _jmax = std::min<int>(j2 + j3, P2-1);
        //     int _jnum = _jmax - _jmin + 1; // k
        //     calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
        //     memory+= _jnum;
        // }

            //    for (int j3=0; j3<P; ++j3) { // p
            //        int _jmin = abs(j2-j3);
            //        int _jmax = j2 + j3;
            //        int _jnum = _jmax - _jmin + 1; // k
            //        calculateWigner3jSymbols(j2,j3,0,0,0,w+memory);
            //        memory+= _jnum;
            //    }
    }
    std::cout << memory << std::endl;

}



void cSH::initWigner3jmConv6D(Real *w, int L) {
    int memory = 0;
    for (int j2=0; j2<L; ++j2) { // l
        for (int j3=0; j3<L; ++j3) { // p
            int _jmin = abs(j2-j3);//std::max(abs(j2-j3), abs(m1));
            //                int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, L-1);
            int _jnum = _jmax - _jmin + 1; // k

            for (int m2=0; m2<=j2; ++m2) { // m
                for (int m3=0; m3<=j3; ++m3) {
                    
                        calculateWigner3jSymbols(j2,j3,m2-m3,-m2, m3,w+memory);

                        memory += _jnum;
                    
                }
            }

        }
    }
    std::cout << memory << std::endl;
}


void cSH::initWignerConstantConv6D(Real *w, int L) {
    int w_index = 0;
    Real factor0, factor1;
    for (int j1=0; j1<L; ++j1) { // l
        factor0 = pow(2*j1+1, 0.5);
        for (int j2=0; j2<L; ++j2) { // p
            factor1 = pow(2*j2+1, 0.5)*factor0;
            for (int j3=0; j3<L; ++j3){
                w[w_index] = factor1*pow(2*j3+1, 0.5);
                w_index += 1;
            }
        }
    }

}


void cSH::initWignerConstant(Real *w, int P, int P2) {

    //   kk= (2*p+1)*sqrt((2*k+1))*w3j0[wigner3j0index]*w3jm[wigner3jmindex];
    //   klm=pow(-1, m)*sqrt((2*l+1));

    int index = 0;
    int index3j0 = 0;
    int index3j0Old = 0;
    int j, j2, j3, m1;
    for (j2=0; j2<P; ++j2) { // l
        index3j0Old = index3j0;
        Real lFactor = sqrt(2*j2+1);
        for (m1=0; m1<=j2; ++m1) { // m
            index3j0 = index3j0Old;
            Real lmFactor = pow(-1, m1)*lFactor;

            for (j3=0; j3<P; j3+=4) { // p, i^p =1

                Real plmFactor = (2*j3+1)*lmFactor;
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);

                //                if ( ((_jmin-abs(j2-j3)) % 2)) {
                //                    _jmin++;
                //                }


                int _jnum = _jmax - _jmin + 1;
                int shift = _jmin-abs(j2-j3);
                index3j0 += shift;
                int k;
                int k0;

                for (j= _jmin, k=index, k0=index3j0; j<= _jmax; j++, k++, k0++) { //k,
                    w[k] = sqrt(2*j+1)*plmFactor*w3jm[k]*w3j0[k0];
                }
                index  += _jnum;
                index3j0 += _jnum;
            }

            for (j3=1; j3<P; j3+=4) { // p, i^p =i

                Real plmFactor = (2*j3+1)*lmFactor;
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1;
                int shift = _jmin-abs(j2-j3);
                index3j0 += shift;
                int k;
                int k0;
                for (j= _jmin, k=index, k0=index3j0; j<= _jmax; j++, k++, k0++) { //k,
                    w[k] = sqrt(2*j+1)*plmFactor*w3jm[k]*w3j0[k0];
                }
                index  += _jnum;
                index3j0 += _jnum;
            }

            for (j3=2; j3<P; j3+=4) { // p, i^p =-1

                Real plmFactor = (2*j3+1)*lmFactor;
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1;
                int shift = _jmin-abs(j2-j3);
                index3j0 += shift;
                int k;
                int k0;
                for (j= _jmin, k=index, k0=index3j0; j<= _jmax; j++, k++, k0++) { //k,
                    w[k] = sqrt(2*j+1)*plmFactor*w3jm[k]*w3j0[k0];
                }
                index  += _jnum;
                index3j0 += _jnum;
            }

            for (j3=3; j3<P; j3+=4) { // p, i^p =-i

                Real plmFactor = (2*j3+1)*lmFactor;
                int _jmin = std::max(abs(j2-j3), abs(m1));
                int _jmax = std::min<int>(j2 + j3, P2-1);
                int _jnum = _jmax - _jmin + 1;
                int shift = _jmin-abs(j2-j3);
                index3j0 += shift;
                int k;
                int k0;
                for (j= _jmin, k=index, k0=index3j0; j<= _jmax; j++, k++, k0++) { //k,
                    w[k] = sqrt(2*j+1)*plmFactor*w3jm[k]*w3j0[k0];
                }
                index  += _jnum;
                index3j0 += _jnum;
            }

            //            for (j3=0; j3<P; ++j3) { // p
            //                Real plmFactor = (2*j3+1)*lmFactor;
            //                int _jmin = std::max(abs(j2-j3), abs(m1));
            //                int _jmax = j2 + j3;
            //                int _jnum = _jmax - _jmin + 1;
            //                int shift = _jmin-abs(j2-j3);
            //                index3j0 += shift;
            //                int k;
            //                int k0;
            //                for (j= _jmin, k=index, k0=index3j0; j<= _jmax; j++, k++, k0++) { //k,
            //                    w[k] = sqrt(2*j+1)*plmFactor*w3jm[k]*w3j0[k0];
            //                }
            //                index  += _jnum;
            //                index3j0 += _jnum;
            //            }
        }
    }
    //    printf("debug initWignerConstant : wigner3j0index = %d, wigner3jmindex = %d\n",index3j0,index);

}


int cSH::writeBesselMatrices(Real * const sphericalBessel, int P, int P2, FILE * outputBesselMatricesFile) {
    Real b;
    int wigner3jmindex;

    Real ** BesselMatricesReal = new  Real* [P];
    Real ** BesselMatricesImag = new  Real* [P];


    // init the output array, can do it through memset

    sComplex zero_complex;
    zero_complex.x = 0.0;
    zero_complex.y = 0.0;

    for (int n=0; n < P; n++) {
        BesselMatricesReal[n] = new Real[(P-n)*(P-n)];
        BesselMatricesImag[n] = new Real[(P-n)*(P-n)];

    }
    for (int n=0; n < P; n++) {
        for (int l = 0; l < P-n; l++){
            for (int k = 0; k < P-n; k++){
                BesselMatricesReal[n][l*(P-n)+k] = 0.0;
                BesselMatricesImag[n][l*(P-n)+k] = 0.0;
            }
        }
        // std::memset(BesselMatrices[n], 0, (P2-n)*(P2-n)*sizeof(BesselMatrices[n][0]));
        //            for (int m=0; m <= n; m++) {
        //                targetAlm[s][n][m].x = 0.0;
        //                targetAlm[s][n][m].y= 0.0;
        //            }
    }

    int k;
    wigner3jmindex = 0;
    // for (int m=0; m < P; m++) { // m loop
    //     for (int l=m; l < P; l++) { // l loop
    for (int l=0; l < P; l++) { // l loop
        for (int m=0; m <= l; m++) { // m loop

            /* p -loop */
            for (int p=0; p<P; p+=4){ // i^p = 1
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                //                int kmin = abs(l-p);
                int kmax = std::min<int>(l+p,P-1);
                //                int kmax = l+p;
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                    if ( (k-kmin) % 2) {
                        continue;
                        // printf("k1=%4d Wc=%6g kmin=%4d kmax=%4d l=%4d m=%4d p=%4d\n",k,wignerConstant[wigner3jmindex],kmin,kmax,l,m, p);
                    }
                    // targetAlm[l][m].x += wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    // targetAlm[l][m].y += wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    BesselMatricesReal[m][(l-m)*(P-m)+k-m] +=  wignerConstant[wigner3jmindex]*b;
                    // if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " " <<  wignerConstant[wigner3jmindex]*b << std::endl;
                    int temp = ((P-m)*(P-m) - ((l-m)*(P-m)+k-m));
                    if (temp < 0) std::cout << temp << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
                //                printf("%8d kmin=%4d kmax=%4d l=%4d m=%4d p=%4d\n",wigner3jmindex,kmin,kmax,l,m, p);
            }

            for (int p=1; p<P; p+=4){ // i^p = i
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P-1);
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k2=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    
                    // targetAlm[l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    // targetAlm[l][m].y +=  wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    BesselMatricesImag[m][(l-m)*(P-m)+k-m] +=  wignerConstant[wigner3jmindex]*b;
                    // if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " i" <<  wignerConstant[wigner3jmindex]*b << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }

            for (int p=2; p<P; p+=4){ // i^p = -1
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P-1);

                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k3=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    
                    // targetAlm[l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    // targetAlm[l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    BesselMatricesReal[m][(l-m)*(P-m)+k-m] +=  -wignerConstant[wigner3jmindex]*b;
                    // if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " " <<  -wignerConstant[wigner3jmindex]*b << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }

            for (int p=3; p<P; p+=4){ // i^p = -i
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P-1);
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }
                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k4=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    
                    // targetAlm[l][m].x +=  wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    // targetAlm[l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    BesselMatricesImag[m][(l-m)*(P-m)+k-m] +=  -wignerConstant[wigner3jmindex]*b;
                    // if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " i" <<  -wignerConstant[wigner3jmindex]*b << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }
        }
    }
    int full_size = 0;
    for (int m=0; m < P; m++) { // m loop
        for (int l=0; l < P-m; l++) { // l loop
            for (int k=0; k < P-m; k++) { // k loop
                int temp = (l)*(P-m)+k;
                if (temp < 0) std::cout << temp << std::endl;
                fwrite(&BesselMatricesReal[m][(l)*(P-m)+k],sizeof(Real), 1 ,outputBesselMatricesFile);
                fwrite(&BesselMatricesImag[m][(l)*(P-m)+k],sizeof(Real), 1 ,outputBesselMatricesFile);
                full_size += 2*sizeof(Real);
            }
        }
    }
    
    for (int n=0; n < P; n++){ 
        delete [] BesselMatricesReal[n];
        delete [] BesselMatricesImag[n];
    }
    
    delete [] BesselMatricesReal;
    delete [] BesselMatricesImag;
    return full_size;


}


/* translation for a single s */
void cSH::translateZ(sComplex **targetAlm, sComplex **referenceAlm,  Real * const sphericalBessel, int P, int P2){

    Real b;
    int wigner3jmindex;


    // init the output array, can do it through memset
    for (int n=0; n < P; n++) {
        std::memset(targetAlm[n], 0, (n+1)*sizeof(targetAlm[n][0]));
        //            for (int m=0; m <= n; m++) {
        //                targetAlm[s][n][m].x = 0.0;
        //                targetAlm[s][n][m].y= 0.0;
        //            }
    }

    int k;
    wigner3jmindex = 0;
    for (int l=0; l < P; l++) { // l loop
        for (int m=0; m <= l; m++) { // m loop

            /* p -loop */
            for (int p=0; p<P; p+=4){ // i^p = 1
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                //                int kmin = abs(l-p);
                int kmax = std::min<int>(l+p,P2-1);
                //                int kmax = l+p;
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                    if ( (k-kmin) % 2) {
                        continue;
                        printf("k1=%4d Wc=%6g kmin=%4d kmax=%4d l=%4d m=%4d p=%4d\n",k,wignerConstant[wigner3jmindex],kmin,kmax,l,m, p);
                    }
                    targetAlm[l][m].x += wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    targetAlm[l][m].y += wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " " << wignerConstant[wigner3jmindex]*b << " "<< targetAlm[l][m].x << " " << targetAlm[l][m].y << " " << referenceAlm[k][m].x << " " << referenceAlm[k][m].y << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
                //                printf("%8d kmin=%4d kmax=%4d l=%4d m=%4d p=%4d\n",wigner3jmindex,kmin,kmax,l,m, p);
            }

            for (int p=1; p<P; p+=4){ // i^p = i
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P2-1);
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k2=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    targetAlm[l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    targetAlm[l][m].y +=  wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " i" << wignerConstant[wigner3jmindex]*b <<" " <<  targetAlm[l][m].x << " " << targetAlm[l][m].y << " " << referenceAlm[k][m].x << " " << referenceAlm[k][m].y << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }

            for (int p=2; p<P; p+=4){ // i^p = -1
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P2-1);

                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }

                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k3=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    targetAlm[l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    targetAlm[l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k << " " << -wignerConstant[wigner3jmindex]*b <<" " <<  targetAlm[l][m].x << " " << targetAlm[l][m].y << " " << referenceAlm[k][m].x << " " << referenceAlm[k][m].y << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }

            for (int p=3; p<P; p+=4){ // i^p = -i
                b=sphericalBessel[p];
                int kmin = std::max(abs(l-p),abs(m));
                int kmax = std::min<int>(l+p,P2-1);
                if ( ((kmin-abs(l-p)) % 2)) {
                    kmin++;
                    wigner3jmindex++;
                }
                for (k=kmin; k<=kmax; k+=2, wigner3jmindex+=2){ // k-loop
                                                                //                    printf("k4=%4d Wc=%6g\n",k,wignerConstant[wigner3jmindex]);
                    targetAlm[l][m].x +=  wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].y;
                    targetAlm[l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[k][m].x;
                    if ((m == 0)  && (l==1) && (k==0)) std::cout << p << " "<< m << " " << l << " " << k <<" i" << -wignerConstant[wigner3jmindex]*b << " " <<  targetAlm[l][m].x << " " << targetAlm[l][m].y << " " << referenceAlm[k][m].x << " " << referenceAlm[k][m].y << std::endl;
                }
                if(!((kmax-kmin)%2))
                    wigner3jmindex--;
            }

            //                for (int p=0; p<P; p++){
            //                klm=pow(-1, m)*sqrt((2*l+1));
            //                    b=SphBesPoly[s][p];
            //
            //                    int kmin = std::max(abs(l-p),abs(m));
            //                    int kmax = l+p;
            //                    int k;
            //
            //                    /*   kk= (2*p+1)*sqrt((2*k+1))*w3j0[wigner3j0index]*w3jm[wigner3jmindex]; */ // old piece
            //
            //                    for (k=kmin; k<=kmax; k++, wigner3jmindex++){
            //                        if ((p % 4) == 0) { // i^p == 1
            //                            targetAlm[s][l][m].x += wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].x;
            //                            targetAlm[s][l][m].y += wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].y;
            //                        } else if ((p % 4) == 1) { // i^p == i
            //                            targetAlm[s][l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].y;
            //                            targetAlm[s][l][m].y +=  wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].x;
            //                        } else if ((p % 4) == 2) { // i^p == -1
            //                            targetAlm[s][l][m].x += -wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].x;
            //                            targetAlm[s][l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].y;
            //                        } else if ((p % 4) == 3) { // i^p == -i
            //                            targetAlm[s][l][m].x +=  wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].y;
            //                            targetAlm[s][l][m].y += -wignerConstant[wigner3jmindex]*b*referenceAlm[s][k][m].x;
            //                        }
            //                    }
            //                }
        }
    }

    //    printf("debug translateZ : wigner3jmindex = %d\n",wigner3jmindex);
}


void cSH::Convolution6D(sComplex ***inputFunction, sComplex ***filterFunction, sComplex ***outputFunction, int L, int P, int L_max){
    Real factor0, factor1, factor2, factor3;
    int wigner3j0index = 0;
    int wigner3jmindex = 0;
    int d;
    factor0 = 8*M_PI*M_PI;
    
    for (int l=0; l < L; l++) { 
        for (int l1=0; l1 < L; l1++) { 
            factor1 = factor0/(2*l1+1);
            for (int l2= abs(l-l1); l2 < std::min<int>(l-l1+1, L); l2 ++){
                d = std::min<int>(l-l1, L-1) - abs(l-l1);
                factor2 = factor1*w3j0[wigner3j0index]*wignerConstant[wigner3j0index];
                wigner3j0index += 1;
                for (int m=0; m <= l; m++) {
                    for (int m1=0; m1 <= l1; m1++) {
                        factor3 = factor2*w3jm[wigner3jmindex];
                        wigner3jmindex += d;
                        int m2 = m-m1;
                        if ((m)%2 == 1){
                            factor3 *= (-1);
                        }
                        if (m2 >= 0){
                            for (int p=0; p < P; p++){
                                outputFunction[p][l][m].x += factor3*(filterFunction[p][l1][m1].x*inputFunction[p][l2][m2].x - filterFunction[p][l1][m1].y*inputFunction[p][l2][m2].y); 
                                outputFunction[p][l][m].y += factor3*(filterFunction[p][l1][m1].y*inputFunction[p][l2][m2].x + filterFunction[p][l1][m1].x*inputFunction[p][l2][m2].y); 
                            }
                        }else{
                            if (l2+m2 %2 == 1){
                                factor3 *= (-1);
                            }
                            for (int p=0; p < P; p++){
                                outputFunction[p][l][m].x += factor3*(filterFunction[p][l1][m1].x*inputFunction[p][l2][-m2].x + filterFunction[p][l1][m1].y*inputFunction[p][l2][-m2].y); 
                                outputFunction[p][l][m].y += factor3*(filterFunction[p][l1][m1].y*inputFunction[p][l2][-m2].x - filterFunction[p][l1][m1].x*inputFunction[p][l2][-m2].y); 
                            }
                             
                        }

                    
                        
                    }
                    

                }
                wigner3jmindex -= (l+1)*(l1+1)*d - d - 1;
            }
            wigner3jmindex += (l+1)*(l1+1)*(std::min<int>(l-l1, L_max-1) - std::min<int>(l-l1, L-1) );
            

        }
        wigner3jmindex += 2*(l+1)*(L_max - L)*(2*L*L +L*(2*L_max +3) + 2*L_max*L_max +3*L_max +1)/6;
    }
    // cMemoryManager::dellocWigner3j0(w3j0);
    
    // cMemoryManager::dellocWigner3jmConv6D(w3jm);
    // cMemoryManager::dellocWigner3j0(wignerConstant);

}
