for( c1 = 2; c1 < 2 * n - 3; c1 += 1)
  #pragma omp parallel for
  for( c3 = max(1, -n + c1 + 2); c3 <= c1 / 2; c3 += 1)
    for( c5 = 0; c5 < c3; c5 += 1)
      ck[(n-c1+c3-1)][(n-c1+2*c3)] = MIN(ck[(n-c1+c3-1)][(n-c1+2*c3)], w[(n-c1+c3-1)][(n-c1+2*c3)]+ck[(n-c1+c3-1)][(n-c1+c3+c5)]+ck[(n-c1+c3+c5)][(n-c1+2*c3)]);
 
