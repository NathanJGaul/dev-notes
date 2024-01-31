## Classes
```python
class Animal(object):
	def __init__(self, species, age):
		self.species, = species
		self.age = age

	def is_person(self):
		return self.species

	def age_one_year(self):
		self.age += 1

class Dog(Animal):
	def age_one_year(self):
		self.age += 7
```

## Model Classes
```python
import torch.nn as nn

class Model(nn.Module):
	def __init__():
		...

	def forward()
		...
```

## Importing Package Modules
```python
# Import 'os' and 'time' modules
import os, time

# Import under an alias
import numpy as np
np.dot(x, y) # Access components with pkg.fn

# Import specific submodules/functions
from numpy import linalg as la, dot as matrix_multiply
# Can result in namespace collisions...
```

## NumPy

```python
x = np.array([1, 2, 3])
y = np.array([[3, 4, 5]])
z = np.array([[6, 7], [8, 9]])
print(x, y, z)

print(x.shape) # >> (3,)
print(y.shape) # >> (1,3)
print(z.shape) # >> (2,2)
```

### Reduction Operations

```python
np.max, np.min, np.amax, np.sum, np.mean
```
Always reduces along an axis (or will reduce along all axes if not specified)
```python
# shape: (3, 2)
x = np.array([[1,2], [3,4], [5,6]])
# shape: (3,)
print(np.max(x, axis = 1)) # >> [2 4 6]
# shape: (3, 1)
print(np.max(x, axis = 1, keepdims = True)) # >> [[2] [4] [6]]
```

### Matrix Operations
```python
np.dot, np.matmul, np.linalg.norm, .T, +, ...
```
Infix operations `(i.e. +, -, *, **, /)` are element-wise.

Element-wise product (Hadamard product) of matrix A and B, A ᐤ B, can be computed:
`A * B`

Dot product and matrix vector product (between 1-D array vectors), can be computed:
```python
np.dot(u, v)
np.dot(x, w)
```

Matrix product / multiplication of matrix A and B, AB, can be computed:
```python
np.matmul(A, B) # or A @ B
```

Transpose with `x.T`

### Indexing
```python
x = np.random.random((3, 4)) # random (3, 4) matrix
x[:] # Selects everything in x
x[np.array([0, 2]), :] # Slects the 0th and 2nd rows
x[1, 1:3] # Selects 1st row as 1-D vector and 1st through 2nd elements

x[x > 0.5] # Boolean indexing
x[:, :, np.newaxis] # 3-D vector of shape (3, 4, 1)
```

### Broadcasting
```python
x = np.random.random((3, 4)) # Random (3, 4) matrix
y = np.random.random((3, 1)) # Random (3, 1) matrix
z = np.random.random((1, 4)) # Random (1, 4) matrix
x + y # Adds y to each column of x
x * z # Multiplies z (element-wise) with each row of x
```
![[Pasted image 20240129195818.png]]

### Efficient NumPy Code
![[Pasted image 20240129195909.png]]

