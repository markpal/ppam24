builtin integer myfun()
builtin integer paired()
integer N
integer i
integer j
integer Q1(0:9999,0:9999)
integer k
integer Qbp1(0:9999,0:9999)
integer ERT
if(N>=1) then
for i = N-1 to 0 by -1 do
for j = i+1 to N-1 do
Q1(i,j)=Q1(i,j-1)
for k = 0 to j-i-1 do
Qbp1(k+i,j)=Q1(k+i+1,j-1)*ERT*paired(k+i,j-1)
Q1(i,j)+=Q1(i,k+i)*Qbp1(k+i,j)
endfor
endfor
endfor
endif

