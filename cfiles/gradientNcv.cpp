/*==========================================================
 * gradientNcv.cpp - Calculate gradient of non convex formulation
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
    
	
    mxArray *X, *Y;
    mwSize  d,  K, r;
    double *q, *W, *B;
    d = mxGetScalar(prhs[5]);
    K = mxGetScalar(prhs[6]);
    r = mxGetScalar(prhs[7]);
    
    W = mxGetPr(prhs[2]); //W
    q = mxGetPr(prhs[3]); //q
    B = mxGetPr(prhs[4]); //B
    
    double *Bt = new double [r*d];
    matrixTrans(d,r,B,Bt); //Bt = B^T
    
    // OUTPUT
	const mwSize *dim_q = mxGetDimensions(prhs[3]);; // three order tensor     	//dim_q[0] = r; dim_q[1] = r; dim_q[2] = K;
	size_t num_dim_q = mxGetNumberOfDimensions(prhs[3]);
    plhs[0] = mxCreateDoubleMatrix(d, K,                     mxREAL); // grad_W
	plhs[1] = mxCreateNumericArray(num_dim_q, dim_q, mxDOUBLE_CLASS, mxREAL); // grad_q
    plhs[2] = mxCreateDoubleMatrix(d, r,                     mxREAL); // grad_B create output gradient matrix
    plhs[3] = mxCreateDoubleMatrix(1, 1,                     mxREAL); // function value
    
    double *grad_W, *grad_B, *grad_q, *f;  // set gradient of W, B, Q.
    grad_W  = mxGetPr(plhs[0]);
    grad_q  = mxGetPr(plhs[1]);
    grad_B  = mxGetPr(plhs[2]);
	f       = mxGetPr(plhs[3]);
	f[0] = 0;
 
    // initialization 
    for (int j=0; j<d*K; j++){ grad_W[j] = 0;}
	int qnum = r*r*K;
	for (int j = 0; j<qnum; j++){ grad_q[j] = 0;}            
	for (int j=0; j<d*r; j++){ grad_B[j] = 0;}
    
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
             
             // grad_W(:,i) = grad_W(:,i) + Xtj'*Xtemp;
             for (int kk = 0; kk < d; kk++)
             {
                 grad_W[icell*d + kk] += Xtemp * Xtj[kk];  // # checked
             }
             
             // XtjB  = Xtj*B; % 1*r
             double *XtjB = new double [r];
             matrixMultiply(1, d, r, Xtj,B, XtjB);
             
             // Xtj*ZZ;  Xtj'*ZZ  for one dimension vector. the storage is same. 
             double *XtjZ = new double [d];
             for (int kk = 0;kk<d;kk++){ XtjZ[kk] = Xtj[kk] * zz;}
            
             
             // XtjB*qt' 1*r = 1*r * r*r
             double *XtjBqtT = new double [r];
             matrixMultiply(1,r,r,XtjB,qtT,XtjBqtT);
             
             // XtjB*qt 1*r
             double *XtjBqt = new double [r];
             matrixMultiply(1,r,r,XtjB,qt,XtjBqt);  // # checked
             
             // gradB1 = (Xtj'*ZZ)*(XtjB*qt')
             double *gradB1 = new double [d*r];
             matrixMultiply(d,1,r,XtjZ,XtjBqtT,gradB1);
             
             // gradB2 = (Xtj'*ZZ')*(XtjB*qt)
             double *gradB2 = new double [d*r];
             matrixMultiply(d,1,r,XtjZ,XtjBqt,gradB2);
             
             // gradB3 = (Xtj')*((XtjB*qt')*XtjB')*(XtjB*qt)
             // gradB3_1 = ((XtjB*qt')*XtjB') 1*1
             double gradB3_1 = 0;
             for (int ii = 0;ii<r;ii++) 
             {
                 gradB3_1 += XtjBqtT[ii]*XtjB[ii];  // # checked
             }
             // gradB3_2 = (Xtj')*gradB3_1 = (Xtj')*((XtjB*qt')*XtjB')
             double *gradB3_2 = new double [d]; // d*1
             for (int ii = 0;ii<d;ii++) 
             {
                 gradB3_2[ii] = Xtj[ii] * gradB3_1;
             }
             
             // gradB3 = gradB3_2*(XtjB*qt) d*r
             double *gradB3 = new double [d*r];
             matrixMultiply(d,1,r,gradB3_2, XtjBqt,gradB3);  // # checked
             
             // gradB4 = (Xtj')*((XtjB*qt)* XtjB')*(XtjB*qt')
             double gradB4_1 = 0; //(XtjB*qt)* XtjB'
             for (int ii = 0; ii<r;ii++)
             {
                 gradB4_1 += XtjBqt[ii]*XtjB[ii];  // # checked
             }
             
             // gradB4_2 = (Xtj')* gradB4_1 = (Xtj')*((XtjB*qt)* XtjB')
             double *gradB4_2 = new double [d];
             for (int ii = 0;ii<d;ii++) 
             {
                 gradB4_2[ii] = Xtj[ii] * gradB4_1;  // # checked
             }
             
             // gradB4 = gradB4_2*(XtjB*qt') = (Xtj')*((XtjB*qt)* XtjB')*(XtjB*qt')
             double *gradB4 = new double [d*r];
             matrixMultiply(d,1,r,gradB4_2,XtjBqtT,gradB4);  // # checked
             
             // gradB d*r
             for (int ii =0;ii<d;ii++)
             {
                 for(int jj = 0;jj<r;jj++)
                 {
                     grad_B[ii + jj*d]+=gradB1[ii+jj*d]+gradB2[ii+jj*d]+gradB3[ii+jj*d]+gradB4[ii+jj*d];  // # checked
                 }
             }
             
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
              
             //  free memory
             delete [] Xtj;
             delete [] XtjQt;
             delete [] XtjB;
             delete [] XtjZ;
             delete [] XtjBqtT;
             delete [] XtjBqt;
             
             delete [] gradB1;
             delete [] gradB2;
             delete [] gradB3_2;
             delete [] gradB3;
             delete [] gradB4_2;
             delete [] gradB4;
             delete [] gradq1;
             
                    
                    
//              delete Xtj,XtjQt,XtjB,XtjZ,XtjBqtT,XtjBqt;
//              delete gradB1,gradB2,gradB3_2,gradB3,gradB4_2,gradB4,gradq1;
 
        }
        delete [] Qt;    
        delete [] qtT; 
        delete [] Qtemp;
    }
    
    delete [] Bt;

}
 
 
 