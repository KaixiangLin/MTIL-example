/*==========================================================
 * MTLncvBCDgradW.cpp - Calculate gradient of non convex formulation    #checked.
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs grad_W dxK   %the gradient of each tensor is same
 *
 * The calling syntax is:
 *
 *		outMatrix = ncvBCDgradW(multiplier, inMatrix)
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2007-2012 The MathWorks, Inc.
 *
 *========================================================*/
 
 #include "mex.h"
 #include <iostream>
 using namespace std;


// INPUT:  0.samples X K*1 cell, 1.label Y K*1 cell, 2.W d*K, 3.q: r*r*K, 4.B: d*r
//         5. d, 6. K,  7. r,  
// OUTPUT: grad_W d*K, grad_q = r*r*K, grad_B d*r, f 1*1 function value;

void matrixMultiply(int m, int p, int n, double *matA, double*matB, double* matC) //#checked. 
{// matA: m*p, matB: p*n, matC: m*n.   // required input matrix store in cloumn major. 

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			double sum = 0;
			for (int k = 0; k < p; k++)
			{
				sum += matA[i+k*m] * matB[k+j*p]; // column major.
			}
			matC[i + j*m] = sum;
		}
	}

}

void matrixTrans(int m, int n, double *A, double *At) // #checked
{// transpose matrix from A to At, A: m*n,  At: n*m

	for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<n; j++)
		{
			At[j + i*n] = A[i + j*m]; // column major

		}
	}
}




void mexFunction(int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[])    
{
    if(nrhs!= 8)
    {  
        mexErrMsgTxt("Wrong number of input arguments.");
    }
    
	
    
    mwSize  d,  K, r;
    double *q, *W, *B,*Xt, *Y;;
    
    d = mxGetScalar(prhs[5]);
    K = mxGetScalar(prhs[6]);
    r = mxGetScalar(prhs[7]);
    
    Xt = mxGetPr(prhs[0]); // X m*d
    Y = mxGetPr(prhs[1]); // Y m*K
    
    W = mxGetPr(prhs[2]); //W
    q = mxGetPr(prhs[3]); //q
    B = mxGetPr(prhs[4]); //B
    
    size_t mt;
    mt  = mxGetM(prhs[0]);  // mt, row numbers
    
    double *Bt = new double [r*d];
    matrixTrans(d,r,B,Bt); //Bt = B^T
    
    // OUTPUT
    plhs[0] = mxCreateDoubleMatrix(d, K,                     mxREAL); // grad_W
    plhs[1] = mxCreateDoubleMatrix(1, 1,                     mxREAL); // function value
    
    double *grad_W, *f;  // set gradient of W, B, Q.
    grad_W  = mxGetPr(plhs[0]);
	f       = mxGetPr(plhs[1]);
	f[0] = 0;
 
    // initialization 
    for (int j=0; j<d*K; j++){ grad_W[j] = 0;}
 
    
    // checked input output correct. can get the zero matrix, tensor.
    
    for (int icell=0;icell<K;icell++)  // the icell th task
    {
 
        double *Yt;
 
        Yt = Y + icell*mt; // mt*1
        
        
        // Qt = B*qt*B'; d*d
        double *Qt = new double [d*d];
        double *Qtemp = new double [d*r];
                
        double *qtT  = new double [r*r];
        double *qt   = q+icell*r*r; // qt
        matrixTrans(r,r,qt,qtT); //qtT = qt^T        
        
        
        matrixMultiply(d, r, r, B,     qt, Qtemp); // Qtemp = B*qt

        matrixMultiply(d, r, d, Qtemp, Bt, Qt);  // Qt = B*qt*B'
        
        // # above code checked. 
        
       
 
        for(int nsamp = 0; nsamp < mt; nsamp++)
        {
            double *Xtj = new double [d];
            for (int kk = 0;kk<d;kk++)
            {
                Xtj[kk] = Xt[nsamp+mt*kk]; // # checked
            }
            
            // ZZ = Xtj*Wt - Yt(j);
            double zz = 0; 
            for (int kk = 0; kk<d; kk++)
            { // W d*K
                zz += Xtj[kk] * W[icell*d+kk]; // # checked
            }
            zz -= Yt[nsamp];
            
            // Xtemp = (Xtj*Qt*Xtj' + ZZ);
             double *XtjQt = new double [d];
             matrixMultiply(1, d, d, Xtj, Qt, XtjQt);//Xtj*Qt
             double XtjQtXtj = 0; //Xtj*Qt*Xtj'
             for (int ii = 0;ii<d;ii++) 
             {
                 XtjQtXtj += XtjQt[ii]*Xtj[ii]; //Xtj*Qt*Xtj'  // # checked
             }
             double Xtemp = XtjQtXtj + zz; 
             
             // grad_W(:,i) = grad_W(:,i) + Xtj'*Xtemp;
             for (int kk = 0; kk < d; kk++)
             {
                 grad_W[icell*d + kk] += Xtemp * Xtj[kk];  // # checked
             }
             
             f[0] += 0.5 * Xtemp*Xtemp;
             
             delete[] XtjQt;
             delete[] Xtj;
 
        }
        delete [] Qt;    
        delete [] qtT; 
        delete [] Qtemp;
    }
    
        delete [] Bt;

}
 
 
 