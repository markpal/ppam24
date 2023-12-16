for( c1 = 1; c1 < 2 * n - 2; c1 += 1)
  #pragma omp parallel for
  for( c3 = max(0, -n + c1 + 1); c3 < (c1 + 1) / 2; c3 += 1) {      
        for( c5 = 0; c5 <= c3; c5 += 1)
          S[(n-c1+c3-1)][(n-c1+2*c3)] = MAX(S[(n-c1+c3-1)][(n-c1+c3+c5-1)] + S[(n-c1+c3+c5-1)+1][(n-c1+2*c3)], S[(n-c1+c3-1)][(n-c1+2*c3)]);
       S[(n-c1+c3-1)][(n-c1+2*c3)] = MAX(S[(n-c1+c3-1)][(n-c1+2*c3)], S[(n-c1+c3-1)+1][(n-c1+2*c3)-1] + sigma((n-c1+c3-1), (n-c1+2*c3)));
    
    }
