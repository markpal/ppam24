        #pragma scop
        if(N>=1 && l>=0 && l<=5)
        for(i=N-1; i>=0; i--){
         for(j=i+1; j<N; j++){
            Q1[i][j] =  Q1[i][j-1];
           for(k=0; k<j-i-l; k++){
             Qbp1[k+i][j] = Q1[k+i+1][j-1] * ERT * paired(k+i,j-1);
             Q1[i][j] +=  Q1[i][k+i] * Qbp1[k+i][j];
           }
         }
        }
       #pragma endscop