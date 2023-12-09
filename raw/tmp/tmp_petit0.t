builtin integer myfun()
builtin integer s()
integer i
integer N
integer j
integer m1(0:9999,0:9999)
integer INT_MIN
integer k
integer H(0:9999,0:9999)
integer W(0:9999)
integer m2(0:9999,0:9999)
integer a(0:9999)
integer b(0:9999)
for i = 1 to N do
for j = 1 to N do

m1(i,j)=INT_MIN
for k = 1 to i do
m1(i,j)=MAX(m1(i,j),H(i-k,j)+W(k))
endfor
m2(i,j)=INT_MIN
for k = 1 to j do
m2(i,j)=MAX(m2(i,j),H(i,j-k)+W(k))
endfor
H(i,j)=MAX(0,MAX(H(i-1,j-1)+s(a(i),b(i)),MAX(m1(i,j),m2(i,j))))
endfor
endfor
