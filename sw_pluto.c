#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (N >= 1) {
  lbp=1;
  ubp=N;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7)
  for (t2=lbp;t2<=ubp;t2++) {
    for (t3=1;t3<=N;t3++) {
      m2[t2][t3] = INT_MIN;;
      m1[t2][t3] = INT_MIN;;
    }
  }
  for (t2=2;t2<=2*N;t2++) {
    lbp=max(1,t2-N);
    ubp=min(N,t2-1);
#pragma omp parallel for private(lbv,ubv,t4,t5,t6,t7)
    for (t3=lbp;t3<=ubp;t3++) {
      if (t2 >= 2*t3+1) {
        for (t5=t3+1;t5<=t2-t3;t5++) {
          m1[(t2-t3)][t3] = MAX(m1[(t2-t3)][t3] ,H[(t2-t3)-(-t3+t5)][t3] + W[(-t3+t5)]);;
        }
        for (t5=t2-t3+1;t5<=t2;t5++) {
          m2[(t2-t3)][t3] = MAX(m2[(t2-t3)][t3] ,H[(t2-t3)][t3-(-t2+t3+t5)] + W[(-t2+t3+t5)]);;
          m1[(t2-t3)][t3] = MAX(m1[(t2-t3)][t3] ,H[(t2-t3)-(-t3+t5)][t3] + W[(-t3+t5)]);;
        }
      }
      if (t2 <= 2*t3-1) {
        for (t5=t2-t3+1;t5<=t3;t5++) {
          m2[(t2-t3)][t3] = MAX(m2[(t2-t3)][t3] ,H[(t2-t3)][t3-(-t2+t3+t5)] + W[(-t2+t3+t5)]);;
        }
        for (t5=t3+1;t5<=t2;t5++) {
          m2[(t2-t3)][t3] = MAX(m2[(t2-t3)][t3] ,H[(t2-t3)][t3-(-t2+t3+t5)] + W[(-t2+t3+t5)]);;
          m1[(t2-t3)][t3] = MAX(m1[(t2-t3)][t3] ,H[(t2-t3)-(-t3+t5)][t3] + W[(-t3+t5)]);;
        }
      }
      if (t2 == 2*t3) {
        if (t2%2 == 0) {
          for (t5=ceild(t2+2,2);t5<=t2;t5++) {
            m2[(t2/2)][(t2/2)] = MAX(m2[(t2/2)][(t2/2)] ,H[(t2/2)][(t2/2)-((-t2+2*t5)/2)] + W[((-t2+2*t5)/2)]);;
            m1[(t2/2)][(t2/2)] = MAX(m1[(t2/2)][(t2/2)] ,H[(t2/2)-((-t2+2*t5)/2)][(t2/2)] + W[((-t2+2*t5)/2)]);;
          }
        }
      }
      H[(t2-t3)][t3] = MAX(0, MAX( H[(t2-t3)-1][t3-1] + s(a[(t2-t3)], b[(t2-t3)]), MAX(m1[(t2-t3)][t3], m2[(t2-t3)][t3])));;
    }
  }
}
