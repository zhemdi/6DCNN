
class cMemoryManager {
	
public:
	template <class T>
	static void allocHalfArray( T** &A, int);
	template <class T>
	static void dellocHalfArray( T** &A, int);
	
	template <class T>
	static void allocTripleArray( T*** &M, int, int);
	template <class T>
	static void dellocTripleArray( T*** &M);
	template <class T>
	static void dellocTripleArray( T*** &M, int I, int P);

	template <class T>
	static void allocTripleArrayNotHalf( T*** &M, int I, int P, int K);
	template <class T>
	static void dellocTripleArrayNotHalf( T*** &M, int I);
	
	template <class T>
	static void alloc2DArray( T** &A, int I, int J);
	template <class T>
	static void delloc2DArray( T** &A);
	
	template <class T>
	static void alloc1DArray( T* &A, int I);
	template <class T>
	static void delloc1DArray( T* &A);

    template <class T>
    static void allocQuadArray( T**** &A, int I, int P, int K);
    template <class T>
    static void dellocQuadArray( T**** &A, int I, int P, int K);

	template <class T>
	static void allocQuadArrayNotHalf( T**** &A, int I, int P, int K, int L);
	template <class T>
	static void dellocQuadArrayNotHalf( T**** &A, int I, int P);

	template <class T>
	static void allocQuintArrayNotHalf( T***** &A, int I, int P, int K, int L, int M);
	template <class T>
	static void dellocQuintArrayNotHalf( T***** &A, int I, int P, int K);


	template <class T>
    static int allocWigner3j0( T* &A, int I, int L3);

    template <class T>
    static int allocWigner3j01( T* &A, int I);

    template <class T>
    static int allocWigner3j02( T* &A, int I);

    template <class T>
    static int allocWigner3j03( T* &A, int I);

    template <class T>
    static int allocWigner3jm( T* &A, int I, int L3);

    template <class T>
    static void dellocWigner3j( T* &A);

    template <class T>
    static int allocWignerG1( T* &A, int P);

    template <class T>
    static int allocWignerRotation( T*** &W, int P);
    template <class T>
    static int dellocWignerRotation( T*** &W, int P);
    template <class T>
    static int allocWigner3j0( T* &A, int I);
    template <class T>
    static void dellocWigner3j0( T* &A);

    template <class T>
    static int allocWigner3jm( T* &A, int I);
    template <class T>
    static void dellocWigner3jm( T* &A);

	template <class T>
    static int allocWigner3jmConv6D( T* &A, int L);
    template <class T>
    static void dellocWigner3jmConv6D( T* &A);

};
