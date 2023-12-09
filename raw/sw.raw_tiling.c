for( c1 = 0; c1 < 2 * N - 1; c1 += 1)
  #pragma omp parallel for
  for( c3 = max(0, -N + c1 + 1); c3 <= min(N - 1, c1); c3 += 1)
    for( c4 = 0; c4 <= 4; c4 += 1) {
      if (c4 == 4) {
        H[(c1-c3+1)][(c3+1)] = MAX(0, MAX( H[(c1-c3+1)-1][(c3+1)-1] + s(a[(c1-c3+1)], b[(c1-c3+1)]), MAX(m1[(c1-c3+1)][(c3+1)], m2[(c1-c3+1)][(c3+1)])));
      } else if (c4 == 3) {
        for( c5 = 0; c5 <= c3; c5 += 1)
          m2[(c1-c3+1)][(c3+1)] = MAX(m2[(c1-c3+1)][(c3+1)] ,H[(c1-c3+1)][(c3+1)-(c5+1)] + W[(c5+1)]);
      } else if (c4 == 2) {
        m2[(c1-c3+1)][(c3+1)] = INT_MIN;
      } else if (c4 == 1) {
        for( c5 = 0; c5 <= c1 - c3; c5 += 1)
          m1[(c1-c3+1)][(c3+1)] = MAX(m1[(c1-c3+1)][(c3+1)] ,H[(c1-c3+1)-(c5+1)][(c3+1)] + W[(c5+1)]);
      } else {
        m1[(c1-c3+1)][(c3+1)] = INT_MIN;
      }
    }

