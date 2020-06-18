#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define PI 3.14159265358979323846
typedef struct tridiag_matrix{
	double *a;
	double *b;
	double *c;
} Tridiag_M;

inline
double interpl_4(double a,double b,double c,double d,double frac){
    double cminusb = c-b;
    return b*(1-frac)+c*frac;
};
double cubic_interpolate( float y0, float y1, float y2, float y3, float mu ) {
 
   double a0, a1, a2, a3, mu2;
 
   mu2 = mu*mu;
   a0 = y3 - y2 - y0 + y1; //p
   a1 = y0 - y1 - a0;
   a2 = y2 - y0;
   a3 = y1;
 
   return ( a0*mu*mu2 + a1*mu2 + a2*mu + a3 );
}
inline double cos_interpl(double a,double b,double frac){
    double mu2=(1-cos(frac*PI))/2;
    return a*(1-mu2)+b*mu2;
};

void solve_tridiagonal(Tridiag_M *t, double *r,double *x,int N) {
    int in;
    double *cprime=(double*) malloc(N*sizeof(double));
    cprime[0] = t->c[0] / t->b[0];
    x[0] = r[0] / t->b[0];
 
    /* loop from 1 to N - 1 inclusive */
    for (in = 1; in < N; in++) {
        double m = 1.0 / (t->b[in] - t->a[in] * cprime[in - 1]);
        cprime[in] = t->c[in] * m;
        x[in] = (r[in] - t->a[in] * x[in - 1]) * m;
    }
    /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
    for (in = N - 1; in-- > 0; ){
        x[in] = x[in] - cprime[in] * x[in + 1]; /*wrong cprime[in] ebasta!*/
    }
        /* free scratch space */
 	free(cprime);
}

void delay_line(double *Y, int *delay0,int *delay1,int *delay2,int *delay3,double *dev,double *out,int M,int N){
	int i;
	for(i=0;i<N;i++){
        int k=M*i;
        if(dev[i]<1){
        	out[i]=cubic_interpolate(Y[k+delay0[i]],Y[k+delay1[i]],Y[k+delay2[i]],Y[k+delay3[i]],dev[i]);
        }
        else{
        	out[i]=cubic_interpolate(Y[k+(delay0[i]+1)%M],Y[k+(delay1[i]+1)%M],Y[k+(delay2[i]+1)%M],Y[k+(delay3[i]+1)%M],dev[i]-1);
        }

	}
}

void calculate_g(double *V,double *Y,double *sherad_factor,double *sheraD,double *sheraRho,double *Yzweig,double *omega,double *g,double d_m_factor,const int n){
	int i;
	g[0]=d_m_factor*V[0];
	for(i=1;i<n;i++){
		g[i]=sherad_factor[i]*sheraD[i]*V[i]+omega[i]*omega[i]*(Y[i]+sheraRho[i]*Yzweig[i]);
	}
}
