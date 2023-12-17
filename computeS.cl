// computeS.cl
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int sigma_host(int a, int b) {
    return a + b;
}

__kernel void computeS(__global int* d_S, int n, int CHUNK_SIZE, int c1) {
    int globalThreadIdx = get_global_id(0);
    int c3_start = globalThreadIdx * CHUNK_SIZE;
    int c3_end = c3_start + CHUNK_SIZE;

    for (int c3 = c3_start; c3 < c3_end && c3 < (c1 + 1) / 2; c3++) {
        if (c3 >= max(0, -n + c1 + 1)) {
            for (int c5 = 0; c5 <= c3; c5++) {
                d_S[(n-c1+c3-1) * n + (n-c1+2*c3)] = MAX(d_S[(n-c1+c3-1) * n + (n-c1+c3+c5-1)] + d_S[(n-c1+c3+c5) * n + (n-c1+2*c3)], d_S[(n-c1+c3-1) * n + (n-c1+2*c3)]);
            }
            d_S[(n-c1+c3-1) * n + (n-c1+2*c3)] = MAX(d_S[(n-c1+c3-1) * n + (n-c1+2*c3)], d_S[(n-c1+c3) * n + (n-c1+2*c3-1)] + sigma_host(n-c1+c3-1, n-c1+2*c3));
        }
    }


}


__kernel void computeS_pluto(__global int* d_S, int n, int CHUNK_SIZE, int t2) {
    int globalThreadIdx = get_global_id(0);
    int t4_base = globalThreadIdx * CHUNK_SIZE + t2;

    for (int offset = 0; offset < CHUNK_SIZE && (t4_base + offset) <= n - 1; offset++) {
        int t4 = t4_base + offset;

        for (int t6 = 0; t6 <= t2 - 1; t6++) {
            d_S[(-t2 + t4) * n + t4] = max(d_S[(-t2 + t4) * n + t6 + (-t2 + t4)] + d_S[(t6 + (-t2 + t4) + 1) * n + t4], d_S[(-t2 + t4) * n + t4]);
        }

        d_S[(-t2 + t4) * n + t4] = max(d_S[(-t2 + t4) * n + t4], d_S[(-t2 + t4 + 1) * n + t4 - 1] + sigma_host(-t2+t4, t4));
    }
}

__kernel void computeS_dapt(__global int* d_S, int n, int CHUNK_SIZE, int w0) {
    int globalThreadIdx = get_global_id(0);
    int h0_base = globalThreadIdx * CHUNK_SIZE - n + w0 + 1;

    for (int offset = 0; offset < CHUNK_SIZE && (h0_base + offset) <= 0; offset++) {
        int h0 = h0_base + offset;

        for (int i3 = 0; i3 < w0; i3++) {
            d_S[(-h0) * w0 + (w0 - h0)] = max(
                d_S[(-h0) * w0 + (-h0 + i3)] + d_S[(-h0 + i3 + 1) * w0 + (w0 - h0)],
                d_S[(-h0) * w0 + (w0 - h0)]
            );
        }

        d_S[(-h0) * w0 + (w0 - h0)] = max(
            d_S[(-h0) * w0 + (w0 - h0)],
            d_S[(-h0 + 1) * w0 + (w0 - h0 - 1)] + sigma_host(-h0, w0 - h0));
        
    }
}
