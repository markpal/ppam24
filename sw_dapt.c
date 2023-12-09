#define min(x,y)    ((x) < (y) ? (x) : (y))
#define max(x,y)    ((x) > (y) ? (x) : (y))

for (int w0 = 2; w0 <= 2 * N; w0 += 1) {
  #pragma omp parallel for
  for (int h0 = max(1, -N + w0); h0 <= min(N, w0 - 1); h0 += 1) {
    {
      m1[h0][w0 - h0] = (INT_MIN);
      for (int i3 = 1; i3 <= h0; i3 += 1) {
        m1[h0][w0 - h0] = MAX(m1[h0][w0 - h0], H[h0 - i3][w0 - h0] + W[i3]);
      }
      m2[h0][w0 - h0] = (INT_MIN);
      for (int i3 = 1; i3 <= w0 - h0; i3 += 1) {
        m2[h0][w0 - h0] = MAX(m2[h0][w0 - h0], H[h0][w0 - h0 - i3] + W[i3]);
      }
    }
    H[h0][w0 - h0] = MAX(0, MAX(H[h0 - 1][w0 - h0 - 1] + s(a[h0], b[h0]), MAX(m1[h0][w0 - h0], m2[h0][w0 - h0])));
  }
}

