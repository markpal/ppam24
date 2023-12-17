if (N >= 2) {
  for (t2=1;t2<=N-1;t2++) {
    lbp=t2;
    ubp=N-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
    for (t4=lbp;t4<=ubp;t4++) {
      for (t6=0;t6<=t2-1;t6++) {
        S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t6+(-t2+t4)] + S[t6+(-t2+t4)+1][t4], S[(-t2+t4)][t4]);;
      }
      S[(-t2+t4)][t4] = MAX(S[(-t2+t4)][t4], S[(-t2+t4)+1][t4-1] + can_pair(RNA, (-t2+t4), t4));;
    }
  }
}
