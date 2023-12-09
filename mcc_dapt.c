if (l >= 0 && l <= 5) {
  for (int w0 = 1; w0 < N; w0 += 1) {
    #pragma omp parallel for
    for (int h0 = -N + w0 + 1; h0 <= 0; h0 += 1) {
      Q1[-h0][w0 - h0] = Q1[-h0][w0 - h0 - 1];
      for (int i3 = 0; i3 < -l + w0; i3 += 1) {
        Qbp1[-h0 + i3][w0 - h0] = ((Q1[-h0 + i3 + 1][w0 - h0 - 1] * (ERT)) * paired((-h0 + i3), (w0 - h0 - 1)));
        Q1[-h0][w0 - h0] += (Q1[-h0][-h0 + i3] * Qbp1[-h0 + i3][w0 - h0]);
      }
    }
  }
}

