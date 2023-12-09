#include <omp.h>
#include <math.h>
#define ceild(n,d)  (((n)<0) ? -((-(n))/(d)) : ((n)+(d)-1)/(d))
#define floord(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
if (n >= 3) {
  for (t2=-n+2;t2<=-1;t2++) {
    lbp=-t2+2;
    ubp=n;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9,t10)
    for (t4=lbp;t4<=ubp;t4++) {
      for (t8=-t2+1;t8<=t4-1-7;t8+=8) {
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][t8]+ck[t8][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+1)]+ck[(t8+1)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+2)]+ck[(t8+2)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+3)]+ck[(t8+3)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+4)]+ck[(t8+4)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+5)]+ck[(t8+5)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+6)]+ck[(t8+6)][t4]);;
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][(t8+7)]+ck[(t8+7)][t4]);;
      }
      for (;t8<=t4-1;t8++) {
        ck[-t2][t4] = MIN(ck[-t2][t4], w[-t2][t4]+ck[-t2][t8]+ck[t8][t4]);;
      }
    }
  }
}
