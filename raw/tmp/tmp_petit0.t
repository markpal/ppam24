builtin integer myfun()
integer i
integer n
integer j
integer k
integer ck(0:9999,0:9999)
integer w(0:9999,0:9999)
for i = n-1 to 1 by -1 do
for j = i+1 to n do
for k = i+1 to j-1 do
ck(i,j)=MIN(ck(i,j),w(i,j)+ck(i,k)+ck(k,j))
endfor
endfor
endfor
