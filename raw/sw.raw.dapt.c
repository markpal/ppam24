int INT_MIN = 3;
int MAX(int, int);
int s(int, int);

void foo2(int N, int **m1, int **m2, int **H, int *W, int *a, int *b){
#pragma scop
	for (int i=1; i <=N; i++)
		for (int j=1; j <=N; j++){
			// Block S
			m1[i][j] = INT_MIN;
			for (int k=1; k <=i; k++)
	            m1[i][j] = MAX(m1[i][j] ,H[i-k][j] + W[k]);
	        m2[i][j] = INT_MIN;
	        for (int k=1; k <=j; k++)
	            m2[i][j] = MAX(m2[i][j] ,H[i][j-k] + W[k]);
	        H[i][j] = MAX(0, MAX( H[i-1][j-1] + s(a[i], b[i]), MAX(m1[i][j], m2[i][j])));
	    }
#pragma endscop
}