        #pragma scop
        for(i=n-1; i>=1; i--){
           for(j=i+1; j<=n; j++)
               for(k=i+1; k<j; k++)
                  ck[i][j] = MIN(ck[i][j], w[i][j]+ck[i][k]+ck[k][j]);
        }
        #pragma endscop