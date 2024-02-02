
#include "cMemoryManager.hpp"
#include <stdlib.h>
#include <algorithm> // std::max
#include <memory>
#include <iostream>


template <class T>
void cMemoryManager::dellocWigner3j0( T* &A) {

    delete [] A;
}

template <class T>
int cMemoryManager::allocWigner3jm( T* &A, int P, int L3) {
    int memory = 0;
//    int j3=L3;
//    for (int j2=0; j2<P; ++j2) {
//        for (int m1=0; m1<=j2; ++m1) {
//            int jmin = std::max(abs(j2-j3), abs(m1));
//            int jmax = j2 + j3;
//            int jnum = jmax - jmin + 1;
//            if (jnum < 0) continue;
//            memory += jnum;
//        }
//    }

    for (int l=0; l<P; ++l) {
        for (int m=-l; m <= l; m++) {
            for (int mp=std::max(-L3+m,-P+1); mp <= std::min(L3+m, P-1); mp++) {

                int _jmin = std::max(abs(l-L3), abs(mp));
                int _jmax =l+L3;
                int _jnum = _jmax - _jmin + 1; // k
                memory += _jnum;
            }
        }
    }

    A = new T[memory];
    return memory;
    
}


template <class T>
int cMemoryManager::allocWigner3jmConv6D( T* &A, int L) {
    int index = 0;
    int memory = 0;
    // for (int j2=0; j2<P; ++j2) {
    //     for (int m1=0; m1<=j2; ++m1) {
    //         for (int j3=0; j3<P; ++j3, ++index) {
    //             int jmin = std::max(abs(j2-j3), abs(m1));
    //             int jmax = j2 + j3;
    //             int jnum = jmax - jmin + 1;
    //             memory += jnum;
    //         }
    //     }
    // }

    for (int j2=0; j2<L; ++j2) { // l
        for (int j3=0; j3<L; ++j3) { // p
            int _jmin = abs(j2-j3);//std::max(abs(j2-j3), abs(m1));
            //                int _jmin = abs(j2-j3);
            int _jmax = std::min<int>(j2 + j3, L-1);
            int _jnum = _jmax - _jmin + 1; // k

            for (int m2=0; m2<=j2; ++m2) { // m
                for (int m3=0; m3<=j3; ++m3) {
                    
                        memory += _jnum;
                    
                }
            }

        }
    }
    A = new T[memory];
    return memory;
}

template <class T>
void cMemoryManager::dellocWigner3jmConv6D( T* &A) {

    delete [] A;
}

template <class T>
int cMemoryManager::allocWigner3jm( T* &A, int P) {
    int index = 0;
    int memory = 0;
    for (int j2=0; j2<P; ++j2) {
        for (int m1=0; m1<=j2; ++m1) {
            for (int j3=0; j3<P; ++j3, ++index) {
                int jmin = std::max(abs(j2-j3), abs(m1));
                int jmax = j2 + j3;
                int jnum = jmax - jmin + 1;
                memory += jnum;
            }
        }
    }
    A = new T[memory];
    return memory;
}

template <class T>
void cMemoryManager::dellocWigner3jm( T* &A) {

    delete [] A;
}

template <class T>
int cMemoryManager::dellocWignerRotation( T*** &W, int P) {
    for (int i=0;i<P;i++) {
        for (int j=0;j<=i;j++) {
            delete [] W[i][j];
        }
        delete [] W[i];
    }
    delete [] W;
    return 0;
}

template <class T>
int cMemoryManager::allocWignerRotation( T*** &W, int P) {

    W=new T** [P];
    for (int i=0;i<P;i++) {
        W[i]=new T* [i+1];
        for (int j=0;j<=i;j++) {
            W[i][j]=new T [2*(i+1)];
        }
    }

    return 0;
}


template <class T> // contiguous version
void cMemoryManager::allocHalfArray( T** &A, int P) {
	
//	void *pointer = malloc (P*sizeof(T*)+P*(P+1)/2*sizeof(T));
//	
//	A = (T**) pointer;
//	
//	for (int i=0;i<P;i++)
//		*(A+i)= (T *) ((char*) pointer + P*sizeof(T*) + i*(i+1)/2*sizeof(T));

	A = new T*[P];
	for (int i=0;i<P;i++)
		A[i] = new T [i+1];
}

template <class T> // contiguous version

void cMemoryManager::dellocHalfArray( T** &A, int P) {
	
//	free(A);
	for (int i=0;i<P;i++)
		delete [] A[i];
	delete [] A;

}

template <class T> // contiguous version
void cMemoryManager::alloc2DArray( T** &A, int I, int J) {

	// std::cout << I*sizeof(T*) << std::endl;
    // std::cout << I*J*sizeof(T) << std::endl;
    
    void *pointer = malloc (I*sizeof(T*)+I*J*sizeof(T));
    
    

	A = (T**) pointer;

	for (int i=0;i<I;i++)
		*(A+i)= (T *) ((char*) pointer + I*sizeof(T*) + i*J*sizeof(T));
}

template <class T> // contiguous version
void cMemoryManager::delloc2DArray( T** &A) {
	
	free( A );
}

template <class T>  // contiguous version
void cMemoryManager::alloc1DArray( T* &A, int I) {
	
	A=new T [I];
}

template <class T>  // contiguous version
void cMemoryManager::delloc1DArray( T* &A) {
	
	delete [] A;
}

template <class T>  // contiguous version
void cMemoryManager::allocTripleArray( T*** &A, int I, int P) {//try to create triple array
	
//	void *pointer = malloc (I*sizeof(T**) + I*P*sizeof(T*)+I*P*(P+1)/2*sizeof(T));
//	
//	A = (T***) pointer;
//	
//	for (int j=0;j<I;j++) {
//		void * jPointer = (char*) pointer + I*sizeof(T**) + j*(P*sizeof(T*) + P*(P+1)/2*sizeof(T));
//		*(A+j)= (T **) jPointer;
//		for (int i=0;i<P;i++)
//			A[j][i] = (T *) ((char*) jPointer + P*sizeof(T*) + i*(i+1)/2*sizeof(T));
//	}
	A = new T**[I];
	for (int i=0;i<I;i++) {
		allocHalfArray( A[i], P);
	}
}

template <class T>  // contiguous version
void cMemoryManager::allocTripleArrayNotHalf( T*** &A, int I, int P, int K) {//try to create triple array

//	void *pointer = malloc (I*sizeof(T**) + I*P*sizeof(T*)+I*P*(P+1)/2*sizeof(T));
//
//	A = (T***) pointer;
//
//	for (int j=0;j<I;j++) {
//		void * jPointer = (char*) pointer + I*sizeof(T**) + j*(P*sizeof(T*) + P*(P+1)/2*sizeof(T));
//		*(A+j)= (T **) jPointer;
//		for (int i=0;i<P;i++)
//			A[j][i] = (T *) ((char*) jPointer + P*sizeof(T*) + i*(i+1)/2*sizeof(T));
//	}
    A = new T**[I];
    for (int i=0;i<I;i++) {
        alloc2DArray( A[i], P, K);
    }
}

//template <class T>  // contiguous version
//void cMemoryManager::dellocTripleArray( T*** &A) {
//	
//	free( A );
//
//}




template <class T>  // contiguous version
void cMemoryManager::dellocTripleArray( T*** &A, int I, int P) {
	
	//	free( A );
	for (int i=0;i<I;i++) {
		dellocHalfArray( A[i], P);
	}
	delete [] A;
}

template <class T>  // contiguous version
void cMemoryManager::dellocTripleArrayNotHalf( T*** &A, int I) {

    //	free( A );
    for (int i=0;i<I;i++) {
        delloc2DArray(A[i]);
    }
    delete [] A;
}

template <class T>  // non-contiguous version
void cMemoryManager::allocQuadArray( T**** &A, int I, int P, int K) {//try to create triple array

    A = new T***[I];
    for (int i=0; i<I; i++) {
        allocTripleArray(A[i], K, P);
    }
}

template <class T>  // non-contiguous version
void cMemoryManager::allocQuadArrayNotHalf( T**** &A, int I, int P, int K, int L) {//try to create triple array

    A = new T***[I];
    for (int i=0; i<I; i++) {
        allocTripleArrayNotHalf(A[i], P, K, L);
    }
}

template <class T>  // non-contiguous version
void cMemoryManager::dellocQuadArray( T**** &A, int I, int P, int K) {//try to create triple array

    for (int i=0; i<I; i++) {
        dellocTripleArray(A[i], K, P);
    }
    delete [] A;
}

template <class T>  // non-contiguous version
void cMemoryManager::dellocQuadArrayNotHalf( T**** &A, int I, int P) {//try to create triple array

    for (int i=0; i<I; i++) {
        dellocTripleArrayNotHalf(A[i], P);
    }
    delete [] A;
}
template <class T>
void cMemoryManager::allocQuintArrayNotHalf( T***** &A, int I, int P, int K, int L, int M){
    A = new T****[I];
    for (int i=0; i<I; i++) {
        allocQuadArrayNotHalf(A[i], P, K, L, M);
    }
}

template <class T>
void cMemoryManager::dellocQuintArrayNotHalf( T***** &A, int I, int P, int K){
    for (int i=0; i<I; i++) {
        dellocQuadArrayNotHalf(A[i], P, K);
    }
    delete [] A;
}


//Wigner functions
template <class T>
int cMemoryManager::allocWigner3j0( T* &A, int P, int L3) {

    int memory = 0;
    int j3=L3;
    for (int j2=0; j2<P; ++j2) {
        int jmin = abs(j2-j3);
        int jmax = j2 + j3;
        int jnum = jmax - jmin + 1;
        memory+= jnum;
    }
    A = new T[memory];
    return memory;
}

template <class T>
int cMemoryManager::allocWigner3j0( T* &A, int P) {

    int index = 0;
    int memory = 0;
    for (int j2=0; j2<P; ++j2) {
        for (int j3=0; j3<P; ++j3, ++index) {
            int jmin = abs(j2-j3);
            int jmax = j2 + j3;
            int jnum = jmax - jmin + 1;
            memory+= jnum;
        }
    }
    A = new T[memory];
    return memory;

}

template <class T>
int cMemoryManager::allocWigner3j01( T* &A, int P) {

    int memory = 0;

    //    int n, np;
    for (int l=0; l < P; l++) {

        { // m = 0
            int m = 0;
            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int _jmin = std::max(abs(l-1), abs(m+n));
                int _jmax = l+1; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k

                memory += _jnum;

            }


        }

        for (int m=1; m <= std::min(l, P-1); m++) { // positive m // m = 3...P-3, positive m

            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int _jmin = std::max(abs(l-1), abs(m+n));
                int _jmax = l+1; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k

                memory +=2*_jnum;

            }
        }

    } // for l
    
    A = new T[memory];
    return memory;
}


template <class T>
int cMemoryManager::allocWigner3j02( T* &A, int P) {

    int memory = 0;

    //    int n, np;
    for (int l=0; l < P; l++) {

        { // m = 0
            int m = 0;
            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]

                int _jmin = std::max(abs(l-2), abs(mp));
                int _jmax = l; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k

                if (abs(m+n)<= l-1) {

                    memory += _jnum;
                }

            }


        }

        for (int m=1; m <= std::min(l, P-1); m++) { // positive m // m = 3...P-3, positive m

            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]

                int _jmin = std::max(abs(l-2), abs(mp));
                int _jmax = l; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k

                if (abs(m+n)<= l-1) {

                    memory += 2*_jnum;
                }
                
            }
        }
        
    } // for l
    
    A = new T[memory];
    return memory;
}


template <class T>
int cMemoryManager::allocWigner3j03( T* &A, int P) {

    int memory = 0;

    for (int l=0; l < P; l++) {

        { // m = 0
            int m = 0;
            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]

                int _jmin = std::max(l, abs(mp));
                int _jmax = l+2; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k

                memory += _jnum;

            }


        }

        for (int m=1; m <= std::min(l, P-1); m++) { // positive m // m = 3...P-3, positive m

            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]

                int _jmin = std::max(l, abs(mp));
                int _jmax = l+2; // can be larger than P-1

                int _jnum = _jmax - _jmin + 1; // k


                memory += 2*_jnum;
            }
        }

    } // for l

    A = new T[memory];
    return memory;
}


template <class T>
void cMemoryManager::dellocWigner3j( T* &A) {

    delete [] A;
}


template <class T>
int cMemoryManager::allocWignerG1( T* &A, int P) {

    int wIdx = 0;


    //    int n, np;
    for (int l=0; l < P; l++) {

        { // m = 0
            int m = 0;
            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]
                int mp_reminder = (l+abs(mp))%2; // lp has to have the same 2-reminder as l

                for (int lp= std::max(abs(l-2), abs(mp)+mp_reminder); lp <= std::min(l+2, P-1); lp+=2){
                    if (abs(mp) <= lp) {
                        wIdx++;
                    }
                }

            }


        }

        for (int m=1; m <= std::min(l, P-1); m++) { // positive m // m = 3...P-3, positive m

            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]
                int mp_reminder = (l+abs(mp))%2; // positive, 0 or 1

                for (int lp= std::max(abs(l-2), abs(mp)+mp_reminder); lp <= std::min(l+2, P-1); lp+=2){
                    if (abs(mp) <= lp) {
                        wIdx++;
                    }
                }
            }
        }

        for (int m=-1; m >= std::max(-l, -P+1); m--) { // m = -P+1...-1, negative m

            for (int n=-1;n<=1;n++)   for (int np=-1;np<=1;np++) {

                int mp = m + n - np; // negative mp => [0][2]
                int mp_reminder = (l+abs(mp))%2; // positive, 0 or 1
                
                for (int lp= std::max(abs(l-2), abs(mp)+mp_reminder); lp <= std::min(l+2, P-1); lp+=2){
                    if (abs(mp) <= lp) {
                        wIdx++;
                    }
                }
                
            }
        }
    } // for l
    
    A = new T[wIdx];
    return wIdx;
}


