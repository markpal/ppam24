int MIN(int, int);

void foo2(int n, int  **ck, int **w){
#pragma scop
	for(int i=n-1; i>=1; i--)
		for(int j=i+1; j<=n; j++)
			for(int k=i+1; k<j; k++)
				ck[i][j] = MIN(ck[i][j], w[i][j]+ck[i][k]+ck[k][j]);
#pragma endscop
}