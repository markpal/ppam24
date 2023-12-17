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
