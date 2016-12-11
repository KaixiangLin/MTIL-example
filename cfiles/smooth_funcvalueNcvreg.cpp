/*==========================================================
 * smooth_funcvalueNcvreg.cpp - Calculate smooth function value of non convex formulation
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs grad_W dxK grad_q = rxrxK grad_B dxr %the gradient of each tensor is same
 *
 * The calling syntax is:
 *
 *		outMatrix = gradientNcv(multiplier, inMatrix)
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
// OUTPUT:  f 1*1 function value;

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




void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])    
{
    if(nrhs!= 10)
    {  
        mexErrMsgTxt("Wrong number of input arguments.");
    }
    
	
    mxArray *X, *Y;
    mwSize  d,  K, r;
    double *q, *W, *B,lambdaB,lambdaq;
    d = mxGetScalar(prhs[5]);
    K = mxGetScalar(prhs[6]);
    r = mxGetScalar(prhs[7]);
    
    W = mxGetPr(prhs[2]); //W
    q = mxGetPr(prhs[3]); //q
    B = mxGetPr(prhs[4]); //B
    lambdaB = mxGetScalar(prhs[8]);
    lambdaq = mxGetScalar(prhs[9]);
    
    double *Bt = new double [r*d];
    matrixTrans(d,r,B,Bt); //Bt = B^T
    
    // OUTPUT
	const mwSize *dim_q = mxGetDimensions(prhs[3]);; // three order tensor     	//dim_q[0] = r; dim_q[1] = r; dim_q[2] = K;
	size_t num_dim_q = mxGetNumberOfDimensions(prhs[3]);
   
    plhs[0] = mxCreateDoubleMatrix(1, 1,                     mxREAL); // function value
    
    double   *f;  // set gradient of W, B, Q.

	f       = mxGetPr(plhs[0]);
	f[0] = 0;
 
 
    
    // checked input output correct. can get the zero matrix, tensor.
    
    for (int icell=0;icell<K;icell++)  // the icell th task
    {
        X = mxGetCell(prhs[0],icell); // X{i}
        Y = mxGetCell(prhs[1],icell); // Y{i}
        
        
        double *Xt, *Yt;
        Xt = mxGetPr(X); // mt*d
        Yt = mxGetPr(Y); // mt*1
        
        size_t mt;
        mt  = mxGetM(X);  // mt, row numbers
        
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
             
             
             f[0] += 0.5 * Xtemp*Xtemp;

             delete [] XtjQt;
             delete [] Xtj;
        }
        delete [] Qt;    
        delete [] qtT; 
        delete [] Qtemp;
        
        for (int ii =0;ii<r;ii++)
        {
            for(int jj = 0;jj<r;jj++)
            {
                
                f[0] += 0.5 *lambdaq*qt[ii + jj*r]*qt[ii + jj*r];
            }
        }

        
    }
        for (int ii = 0; ii<d; ii++)
        {
            for (int jj = 0; jj<r; jj++)
            {
                f[0] += 0.5*lambdaB*B[ii+jj*d]*B[ii+jj*d];
            }
        }
    
    
       delete [] Bt;

}
 
 
 