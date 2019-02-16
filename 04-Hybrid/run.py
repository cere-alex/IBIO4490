import numpy as np
a= np.array([1,2,3,4,5])
a.dtype
b = np.array([1,2,3,4.5,3,4,])
b.dtype
#print (a)
#print(b)
np.zeros((3,4))
c=np.ones((2,3,4),dtype=np.int16)
#print(c)
#print(c.shape)
np.arange(10,30,5)
a=np.arange(10,30,5)
#print(a.shape)
[i for i in range(10,30,5)]
np.arange(0,2,0.3)
b=np.arange(12).reshape(4,3)
print(b)
A=np.array([[1,1],[0,1]])
print(A)
print()#imprime todo los prints
B=np.array([[2,0],[2,4]])
print(B)
print(A*B)
print(A.dot(B))

print(np.dot(A,B))
np.vstack((A,B))
b=np.random.random((2,3))
print(b)
b.sum(axis=0)
a=np.arange(12)
b=a
print(a)
print(b)
print(b is a)
b.shape=3,4
a.shape
print(b)
print(a)
print('---')
print(id(a))
print(id(b))
c=a.view()
print(c is a)
d=a.copy()
print(d)
print(d is a)
print(d.base is a)
d[0,0]=9999
print(d)
d[0][1]=234
print(d)
d[0,1]=666
print(d)
