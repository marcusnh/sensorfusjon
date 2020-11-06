import numpy as np
i = 1
ind = 2 * i # starting postion of the ith landmark into H
inds = slice(ind, ind + 2)
a = np.array([1,2,3,4,5,6,7,8])
print(a[1:2])


b = np.array([[-np.sin(1+1)],
                        [np.cos(2)]])
b = np.array([-np.sin(1+1),

                        np.cos(2)]).T

c = a.T
print(a[3:])
print(b.shape)
