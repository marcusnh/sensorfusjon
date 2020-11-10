Why dont we use the laplace approximation or alternative The Gauss-Newton method?
  - is this because it's an iterated EKF and we are actually using it?


What happens when we decrease R? why does the algorithm go slower?
- Is it because the algorithm will add several landmarks in pose to the same real landmark, making the process slower.
- Or is this because it makes the covariance larger?

What type of errors are we supposed to see in simulated:
- can be very slow with changed varibels in R and Q and alphas?
