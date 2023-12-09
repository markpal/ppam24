for( c1 = 1; c1 < n + floord(n - 2, 16) - 1; c1 += 1)
  #pragma omp parallel for
  for( c3 = max(1, c1 - (n + 14) / 16 + 1); c3 <= c1 - (c1 + 18) / 17 + 1; c3 += 1)
    for( c5 = 0; c5 < c3; c5 += 1)
      for( c7 = max(-n + c3 + 1, -n + 16 * c1 - 16 * c3 + 1); c7 <= min(-1, -n + 16 * c1 - 16 * c3 + 16); c7 += 1)
        ck[(-c7)][(c3-c7+1)] = MIN(ck[(-c7)][(c3-c7+1)], w[(-c7)][(c3-c7+1)]+ck[(-c7)][(c5-c7+1)]+ck[(c5-c7+1)][(c3-c7+1)]);

 