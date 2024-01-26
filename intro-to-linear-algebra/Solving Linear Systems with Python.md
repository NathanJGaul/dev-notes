
![[Pasted image 20240126074806.png]]

System:

e = t
e = 4(t - 30)

Matrix Form:

$$

\begin{bmatrix}  
1 & -1\\  
4 & -1 
\end{bmatrix}

\begin{bmatrix}  
t\\
e
\end{bmatrix}

=

\begin{bmatrix}  
0\\
120
\end{bmatrix}

$$

Solution to 1:

```python
import numpy as np

# Define the coefficients of the equations in a matrix A
A = np.array([[1, -1], [4, -1]])

# Define the constants in a vector b
b = np.array([0, 120])

# Solve the system of equations
x = np.linalg.solve(A, b)

print(x) # [40, 40]

# Answer: t = 40 days
# Answer 2: e x 2 = 40 x 2 = 80 kJ
```