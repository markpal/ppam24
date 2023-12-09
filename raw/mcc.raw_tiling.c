for( c1 = 1; c1 < 2 * N - 2; c1 += 1)
  #pragma omp parallel for
  for( c3 = max(0, -N + c1 + 1); c3 < (c1 + 1) / 2; c3 += 1)
    for( c4 = 0; c4 <= 1; c4 += 1) {
      if (c4 == 1) {
        for( c5 = 0; c5 <= c3; c5 += 1)
          for( c12 = 15; c12 <= 16; c12 += 1)
            {
            if( c12 == 13 ) Q1[(N-c1+c3-1)][(N-c1+2*c3)] = Q1[(N-c1+c3-1)][(N-c1+2*c3)-1];
            if( c12 == 15 ) Qbp1[c5+(N-c1+c3-1)][(N-c1+2*c3)] = Q1[c5+(N-c1+c3-1)+1][(N-c1+2*c3)-1] * ERT * paired(c5+(N-c1+c3-1),(N-c1+2*c3)-1);
            if( c12 == 16 ) Q1[(N-c1+c3-1)][(N-c1+2*c3)] += Q1[(N-c1+c3-1)][c5+(N-c1+c3-1)] * Qbp1[c5+(N-c1+c3-1)][(N-c1+2*c3)];
            }
      } else {
        Q1[(N-c1+c3-1)][(N-c1+2*c3)] = Q1[(N-c1+c3-1)][(N-c1+2*c3)-1];
      }
    } 

