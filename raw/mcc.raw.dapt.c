int paired(int, int);

void foo2(int N, int l, int **Q1, int **Qbp1, int ERT){
#pragma scop
	if(N>=1 && l>=0 && l<=5)
		for(int i=N-1; i>=0; i--){
			for(int j=i+1; j<N; j++){
				//printf("%.f\n", Q1[i][j]);
				Q1[i][j] =  Q1[i][j-1];
				for(int k=0; k<j-i-l; k++){
					Qbp1[k+i][j] = Q1[k+i+1][j-1] * ERT * paired(k+i,j-1);
					Q1[i][j] +=  Q1[i][k+i] * Qbp1[k+i][j];
					//printf("%.f\n", Q1[i][j]);
				}
			}
		}
#pragma endscop
}
