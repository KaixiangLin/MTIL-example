/*==========================================================
 * MTLncvBCDgradq.cpp - Calculate gradient of non convex formulation  # checked
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
    
	
    double *Xt, *Y;
    mwSize  d,  K, r;
    double *q, *W, *B;
    d = mxGetScalar(prhs[5]);
    K = mxGetScalar(prhs[6]);
    r = mxGetScalar(prhs[7]);
    
    Xt = mxGetPr(prhs[0]); // X m*d
    Y = mxGetPr(prhs[1]); // Y m*K
    size_t mt;
    mt  = mxGetM(prhs[0]);  // mt, row numbers    
    
    
    
    W = mxGetPr(prhs[2]); //W
    q = mxGetPr(prhs[3]); //q
    B = mxGetPr(prhs[4]); //B
    
    double *Bt = new double [r*d];
    matrixTrans(d,r,B,Bt); //Bt = B^T
    
    // OUTPUT
	const mwSize *dim_q = mxGetDimensions(prhs[3]);; // three order tensor     	//dim_q[0] = r; dim_q[1] = r; dim_q[2] = K;
	size_t num_dim_q = mxGetNumberOfDimensions(prhs[3]);
	plhs[0] = mxCreateNumericArray(num_dim_q, dim_q, mxDOUBLE_CLASS, mxREAL); // grad_q
    plhs[1] = mxCreateDoubleMatrix(1, 1,                     mxREAL); // function value
    
    double *grad_q, *f;  // set gradient of W, B, Q.
 
    grad_q  = mxGetPr(plhs[0]);
	f       = mxGetPr(plhs[1]);
	f[0] = 0;
 
    // initialization 
 
	int qnum = r*r*K;
	for (int j = 0; j<qnum; j++){ grad_q[j] = 0;}            
 
    
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


             // XtjB  = Xtj*B; % 1*r
             double *XtjB = new double [r];
             matrixMultiply(1, d, r, Xtj,B, XtjB);
             
             
             // grad_q = (XtjB')*Xtemp*XtjB
             // gradq1 = (XtjB')*Xtemp; r*1
             double *gradq1 = new double [r]; 
             for(int ii = 0; ii<r;ii++)
             {
               gradq1[ii] = XtjB[ii] * Xtemp;    // # checked
             }
             
             // grad_q = gradq1*XtjB
             for (int ii =0;ii<r;ii++)
             {
                 for(int jj = 0;jj<r;jj++)
                 { 
                     grad_q[icell*r*r + ii + jj*r] += gradq1[ii]*XtjB[jj];    // # checked
                 }
             }
             
             f[0] += 0.5 * Xtemp*Xtemp;
              // free memory
             delete [] Xtj;
             delete [] XtjQt;
             delete [] XtjB;
             delete [] gradq1;
  
 
        }
        delete [] Qt;    
        delete [] qtT; 
        delete [] Qtemp;
  
    }
    
        delete [] Bt;   

}
 
 
 