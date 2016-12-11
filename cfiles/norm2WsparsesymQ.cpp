/*==========================================================
 * gradFuncMTLExpConvx.cpp - Calculate gradient of convex formulation and smooth part function value.
 * 
 * INPUT: matrix X m*d. matrix Y m*1.
 * Output grad_W dxK grad_Q d*d  %the gradient of each tensor is same
 *
 * The calling syntax is:
 *
 *		outMatrix = gradFuncMTLExpConvx(multiplier, inMatrix)
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




void mexFunction(int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[])      // input
{
    if(nrhs!= 6)
    {  
        mexErrMsgTxt("Wrong number of input arguments.");
    }
    
	
     
    mwSize  d,  K;
    double *Qt, *W ,*Xt, *Yt,rho1;
    
    Xt = mxGetPr(prhs[0]); // X m*d
    Yt = mxGetPr(prhs[1]); // Y m*1
    W = mxGetPr(prhs[2]); //W d*1   
    Qt  = mxGetPr(prhs[3]);
    d = mxGetScalar(prhs[4]);
    rho1 = mxGetScalar(prhs[5]); // the l2 regularization parameter on w's l2 norm
 

    size_t mt;
    mt  = mxGetM(prhs[0]);  // mt, row numbers

    // OUTPUT
    
 
    plhs[0] = mxCreateDoubleMatrix(d, 1, mxREAL); // grad_W
    plhs[1] = mxCreateDoubleMatrix(d, d, mxREAL); // grad_Q    
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL); // function value
    double *f;
    f  = mxGetPr(plhs[2]);
 
    double *grad_W, *grad_Q;  // set gradient of W,   Q.
    grad_W  = mxGetPr(plhs[0]);
    grad_Q  = mxGetPr(plhs[1]);
    
    // initialization 
    for (int j=0; j<d ; j++){ grad_W[j] = 0; }
	int qnum = d*d ;
	for (int j = 0; j<qnum; j++){ grad_Q[j] = 0;}            
    f[0] = 0;
             
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
            zz += Xtj[kk] * W[kk]; // # checked
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
             grad_W[ii] += Xtj_scale[ii];
         }


         //grad_Qt+= Xtj'*(Xtj*Qt*Xtj' - Yt(j)+ Wt'*Xtj')*Xtj;  
         for (int ii =0;ii<d; ii++)
         {
             for(int jj=0;jj<d;jj++)
             {
                 grad_Q[ii+jj*d] += Xtj[ii]*Xtj_scale[jj];
             }
         }

         f[0] +=  Xtemp*Xtemp;

         delete [] Xtj_scale;
         delete [] XtjQt;
         delete [] Xtj;
    }
    
         double norm2 = 0;
         for (int ii = 0;ii<d;ii++)
         {
             norm2 += W[ii]*W[ii];
             grad_W[ii] += rho1*W[ii];
         }
         
         
         f[0]  = 0.5*(rho1*norm2+f[0]);
   

}
 
 
 