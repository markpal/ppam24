for (int w0 = 2; w0 < n; w0 += 1) {
  #pragma omp parallel for
  for (int h0 = -n + w0; h0 < 0; h0 += 1) {
    for (int i2 = -h0 + 1; i2 < w0 - h0; i2 += 1) {
      ck[-h0][w0 - h0] = MIN(ck[-h0][w0 - h0], (w[-h0][w0 - h0] + ck[-h0][i2]) + ck[i2][w0 - h0]);
    }
  }
}

