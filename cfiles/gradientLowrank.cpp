/*==========================================================
 * gradientLowrank.cpp - Calculate gradient of non convex formulation
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


// INPUT:  0.samples X K*1 cell, 1.label Y K*1 cell, 2.W d*K, 3.Q 3*1 cell, 4. d, 5. K
// OUTPUT: grad_W d*K, grad_Q = d*d*K

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




void mexFunction(int nlhs, mxArray *plhs[],     // output
                 int nrhs, mxArray *prhs[])      // input
{
    if(nrhs!= 6)
    {  
        mexErrMsgTxt("Wrong number of input arguments.");
    }
    
	
    mxArray *X, *Y, *Q1arr,*Q2arr,*Q3arr;
    mwSize  d,  K;
    double *Q1,*Q2,*Q3, *W ;
    
    W = mxGetPr(prhs[2]); //W d*K
//     Q = mxGetPr(prhs[3]); //Q 3*1 cell    
    d = mxGetScalar(prhs[4]);
    K = mxGetScalar(prhs[5]);
 
    Q1arr = mxGetCell(prhs[3],0); //d*d*K
    Q2arr = mxGetCell(prhs[3],1);
    Q3arr = mxGetCell(prhs[3],2);
    Q1    = mxGetPr(Q1arr);
    Q2    = mxGetPr(Q2arr);
    Q3    = mxGetPr(Q3arr);

    // OUTPUT
	mwSize *dim_q = new int [3]; // three order tensor     
	dim_q[0] = d; dim_q[1] = d; dim_q[2] = K;
    size_t num_dim_q = 3;
    plhs[0] = mxCreateDoubleMatrix(d, K,                     mxREAL); // grad_W
	plhs[1] = mxCreateNumericArray(num_dim_q, dim_q, mxDOUBLE_CLASS, mxREAL); // grad_q

    
    double *grad_W, *grad_Q, *f;  // set gradient of W, B, Q.
    grad_W  = mxGetPr(plhs[0]);
    grad_Q  = mxGetPr(plhs[1]);
    
    // initialization 
    for (int j=0; j<d*K; j++){ grad_W[j] = 0; }
	int qnum = d*d*K;
	for (int j = 0; j<qnum; j++){ grad_Q[j] = 0;}            

   
    
    for (int icell=0;icell<K;icell++)  // the icell th task
    {
        X = mxGetCell(prhs[0],icell); // X{i}
        Y = mxGetCell(prhs[1],icell); // Y{i}
        
        
        double *Xt, *Yt;
        Xt = mxGetPr(X); // mt*d
        Yt = mxGetPr(Y); // mt*1
        
        size_t mt;
        mt  = mxGetM(X);  // mt, row numbers
        
        // Qt = Q{1}(:,:,i) + Q{2}(:,:,i) + Q{3}(:,:,i); d*d
        double *Qt = new double [d*d];
        int icelld = icell*d*d;
        for(int ii = 0;ii<d*d;ii++)
        {
             Qt[ii] = Q1[ii+icelld] + Q2[ii+icelld] + Q3[ii+icelld];
        }
             
 
        for(int nsamp = 0; nsamp < mt; nsamp++)
        {
            // Xtj = Xt(j,:);  % 1 * d
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
             
             double *Xtj_scale = new double [d];
             
             //grad_W(:,i) = grad_W(:,i) + Xtj'*(Xtj*Wt + Xtj*Qt*Xtj' - Yt(j));
             for (int ii=0; ii<d;ii++)
             {
                 Xtj_scale[ii] = Xtj[ii] * Xtemp;
                 grad_W[ii+icell*d] += Xtj_scale[ii];
             }
             
             
             //grad_Qt+= Xtj'*(Xtj*Qt*Xtj' - Yt(j)+ Wt'*Xtj')*Xtj;  
             for (int ii =0;ii<d; ii++)
             {
                 for(int jj=0;jj<d;jj++)
                 {
                     grad_Q[icelld+ii+jj*d] += Xtj[ii]*Xtj_scale[jj];
                 }
             }
             delete [] Xtj_scale;
             delete [] XtjQt;
             delete [] Xtj;
        }
        
        delete []Qt;
        
    }
    
    delete [] dim_q;

}
 
 
 