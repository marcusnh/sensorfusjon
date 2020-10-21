import numpy as np
import scipy.linalg as la
print(np.eye(6))
A = np.zeros((15, 15))
M = la.block_diag(np.eye(6),np.eye(6),np.eye(6))
print(np.eye(6))
print(M)

Hx = np.hstack((np.eye(3), np.zeros((3, 13))))

print(Hx)
