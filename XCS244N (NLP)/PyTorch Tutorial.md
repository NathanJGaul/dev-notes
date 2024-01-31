[colab notebook](https://colab.research.google.com/drive/1XJhrdkcE6rlC3WwtH7Py1bYl8CEVuBSU?usp=sharing#scrollTo=zn4pW2rkmXuj)

PyTorch: author and manipulate tensors (with GPUs); and author Neural Networks

```python
import torch
import torch.nn as nn
```

Tensors in PyTorch are "equivalent" to numpy arrays

```python
data = torch.tensor([
				   [0, 1],
				   [2, 3],
				   [4, 5]
])

# typing a tensor
data = torch.tensor([
					 [0.32221111, 0.5],
					 [2, 3],
					 [4, 5]
], dtype=torch.float32)
```

```python
zeros = torch.zeros(2, 5) # a tensor of all zeros
# tensor([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])

ones = torch.ones(3, 4) # a tensor of all ones
# tensor([[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]])

rr = torch.arange(1, 10) # range from [1, 10)
# tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])

rr + 2
# tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11])

rr * 2
# tensor([ 2,  4,  6,  8, 10, 12, 14, 16, 18])
```

```python
a = torch.tensor([[1, 2], [2, 3], [4, 5]]) # (3, 2)
b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]) # (2, 4) (3, 4)

print("A is", a)
print("B is", b)
print("The product is", a.matmul(b))
print("The other product is", a @ b) # +, -, *, @

# A is tensor([[1, 2], [2, 3], [4, 5]])
# B is tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
# The product is tensor([[11, 14, 17, 20], [17, 22, 27, 32], [29, 38, 47, 56]])
# The other product is tensor([[11, 14, 17, 20], [17, 22, 27, 32], [29, 38, 47, 56]])
```

### matrix multiplication
"matrix version of the dot product. The result of matrix multiplication is a matrix, whose elements are the dot products of pairs of vectors in each matrix"
$$
\begin{bmatrix}
a_{11} \ \ a_{12} \\
a_{21} \ \ a_{22} \\
\end{bmatrix}

\begin{bmatrix}
b_{11} \ \ b_{12} \\
b_{21} \ \ b_{22} \\
\end{bmatrix}

=
\begin{bmatrix}
a_{11}b_{11} + a_{12}b_{21} \ \ \ a_{11}b_{12} + a_{12}b_{22}\\
a_{21}b_{11} + a_{22}b_{21} \ \ \ a_{21}b_{12} + a_{22}b_{22}\\
\end{bmatrix}
$$
```python
a.matmul(b)
```
![[Pasted image 20240130125814.png]]
### dot product: 
"takes two same-sized vectors and returns a single number"
$$
[a_1 \ a_2]\begin{bmatrix}b_1 \\b_2 \end{bmatrix}=a_1b_1 + a_2b_2
$$
$$
a \cdot b = \vert a \vert \vert b \vert \rm cos \theta
$$
```python
a @ b
```

![[Pasted image 20240130125750.png]]
### Batching Matrix Multiplication
```python
torch.bmm(input, mat2, out=None) # -> Tensor
```
Performs a batch matrix-matrix product of matrices stored in `input` and `mat2`.
`input` and `mat2` must be 3-D tensors each containing the same number of matrices.
if `input` is a $(b×n×m)$ tensor, mat2 is a $(b×m×p)$ tensor, `out` will be a $(b×n×p)$ tensor.
```python
input = torch.randn(10, 3, 4)
mat2 = torch.randn(10, 4, 5)
res = torch.bmm(input, mat2)
res.size()
# torch.Size([10, 3, 5])
```

### Reshaping Tensors
```python
rr = torch.arange(1, 16)
print("The shape is currently", rr.shape)
print("The contents are currently", rr)
print()

rr = rr.view(5, 3)
print("After reshaping, the shape is currently", rr.shape)
print("The contents are currently", rr)

# The shape is currently torch.Size([15]) 
# The contents are currently tensor([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) 

# After reshaping, the shape is currently torch.Size([5, 3]) 
# The contents are currently tensor([[ 1, 2, 3], [ 4, 5, 6], [ 7, 8, 9], [10, 11, 12], [13, 14, 15]])
```

`view` returns a reshaped tensor, `reshape` reshapes the tensor in place

### Convert NumPy arrays to/from PyTorch Tensors
```python
import numpy as np

# numpy.ndarray --> torch.Tensor:
arr = np.array([[1, 0, 5]])
data = torch.tensor(arr)
print("This is a torch.tensor", data)

# torch.Tensor --> numpy.ndarray:
new_arr = data.numpy()
print("This is a np.ndarray", new_arr)

# This is a torch.tensor tensor([[1, 0, 5]])
# This is a np.ndarray [[1 0 5]]
```

### Vectorized Operations
```python
data = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7)
print("Data is:", data)

# We can perform operations like *sum* over each row...
print("Taking the sum over columns:")
print(data.sum(dim=0))

# or over each column.
print("Taking thep sum over rows:")
print(data.sum(dim=1))
  
# Other operations are available:
print("Taking the stdev over rows:")
print(data.std(dim=1))

# Data is: tensor([[ 1., 2., 3., 4., 5., 6., 7.], [ 8., 9., 10., 11., 12., 13., 14.], [15., 16., 17., 18., 19., 20., 21.], [22., 23., 24., 25., 26., 27., 28.], [29., 30., 31., 32., 33., 34., 35.]]) 
# Taking the sum over columns: tensor([ 75., 80., 85., 90., 95., 100., 105.])
# Taking thep sum over rows: tensor([ 28., 77., 126., 175., 224.]) 
# Taking the stdev over rows: tensor([2.1602, 2.1602, 2.1602, 2.1602, 2.1602])
```

```python
data = torch.tensor([[1, 2.2, 9.6], [4, -7.2, 6.3]])

row_avg = data.mean(dim=1)
col_avg = data.mean(dim=0)

print(row_avg.shape)
print(row_avg)

print(col_avg.shape)
print(col_avg)

# torch.Size([2])
# tensor([4.2667, 1.0333]) 

# torch.Size([3]) 
# tensor([ 2.5000, -2.5000, 7.9500])
```

### Indexing
```python
# Initialize an example tensor
x = torch.Tensor([
[[1, 2], [3, 4]],
[[5, 6], [7, 8]],
[[9, 10], [11, 12]]
])
x
# tensor([[[ 1.,  2.],
#         [ 3.,  4.]],
#
#        [[ 5.,  6.],
#         [ 7.,  8.]],
#
#        [[ 9., 10.],
#         [11., 12.]]])

x.shape
# torch.Size([3, 2, 2])

# Access the 0th element, which is the first row
x[0] # Equivalent to x[0, :]
# tensor([[ 1.,  2.],
#        [ 5.,  6.],
#        [ 9., 10.]])

x[:, 0]
# tensor([[ 1.,  2.],
#        [ 5.,  6.],
#        [ 9., 10.]])
```

### Autograd
```python
# Create an example tensor
# requires_grad parameter tells PyTorch to store gradients
x = torch.tensor([2.], requires_grad=True)

# Print the gradient if it is calculated
# Currently None since x is a scalar
pp.pprint(x.grad)
```

```python
# Calculating the gradient of y with respect to x
y = x * x * 3 # 3x^2
y.backward()
pp.print(x.grad)
# tensor([12.])
```

```python
z = x * x * 3
z.backward()
pp.pprint(x.grad)
# tensor([24.])
```

We can see that the `x.grad` is updated to be the sum of the gradients calculated so far. When we run backprop in a neural network, we sum up all the gradients for a particular neuron before making an update. This is exactly what is happening here! This is also the reason why we need to run `zero_grad()` in every training iteration (more on this later). Otherwise our gradients would keep building up from one training iteration to the other, which would cause our updates to be wrong.

## Neural Network Module

So far we have looked into the tensors, their properties and basic operations on tensors. These are especially useful to get familiar with if we are building the layers of our network from scratch. We will utilize these in Assignment 3, but moving forward, we will use predefined blocks in the `torch.nn` module of `PyTorch`. We will then put together these blocks to create complex networks. Let's start by importing this module with an alias so that we don't have to type `torch` every time we use it.

```python
import torch.nn as nn
```

### Linear Layer

We can use `nn.Linear(H_in, H_out)` to create a a linear layer. This will take a matrix of `(N, *, H_in)` dimensions and output a matrix of `(N, *, H_out)`. The `*` denotes that there could be arbitrary number of dimensions in between. The linear layer performs the operation `Ax+b`, where `A` and `b` are initialized randomly. If we don't want the linear layer to learn the bias parameters, we can initialize our layer with `bias=False`.
```python
# Create the inputs
input = torch.ones(2,3,4)
# N* H_in -> N*H_out

# Make a linear layers transforming N,*,H_in dimensinal inputs to N,*,H_out
# dimensional outputs
linear = nn.Linear(4, 2)
linear_output = linear(input)
linear_output
# tensor([[[-1.3855,  0.3119],
#         [-1.3855,  0.3119],
#         [-1.3855,  0.3119]],
#
#        [[-1.3855,  0.3119],
#         [-1.3855,  0.3119],
#         [-1.3855,  0.3119]]], grad_fn=<ViewBackward0>)
```