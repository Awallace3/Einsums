# PyEinsums API Reference

## Table of Contents
- [Initialization](#initialization)
- [Tensor Types](#tensor-types)
- [Tensor Creation](#tensor-creation)
- [Tensor Operations](#tensor-operations)
- [Linear Algebra](#linear-algebra)
- [Einsum Plans](#einsum-plans)
- [Profiling](#profiling)

## Initialization

The library auto-initializes on import. Pass arguments via `--einsums`:
```python
import einsums as ein  # Calls ein.initialize() automatically
# ein.core.finalize() is registered via atexit
```

## Tensor Types

Type suffixes: `F`=float32, `D`=float64, `C`=complex64, `Z`=complex128

```python
# Core tensor classes in ein.core:
RuntimeTensorF   # float32 tensor
RuntimeTensorD   # float64 tensor  
RuntimeTensorC   # complex64 tensor
RuntimeTensorZ   # complex128 tensor

RuntimeTensorViewF  # View into float32 tensor
RuntimeTensorViewD  # View into float64 tensor
RuntimeTensorViewC  # View into complex64 tensor
RuntimeTensorViewZ  # View into complex128 tensor
```

## Tensor Creation

### Direct Construction
```python
# Named tensor with dimensions
A = ein.core.RuntimeTensorD("A", [3, 3])

# Unnamed tensor
B = ein.core.RuntimeTensorD([3, 3])

# From numpy array (copies data)
C = ein.core.RuntimeTensorD(np.array([[1, 2], [3, 4]]))
```

### Factory Functions (ein.utils)
```python
# Zero-initialized tensors
T = ein.utils.create_tensor("name", [dim1, dim2], dtype=float)

# Random tensors
R = ein.utils.create_random_tensor("name", [dim1, dim2], dtype=float)

# Positive definite matrices (useful for testing)
P = ein.utils.create_random_definite("name", rows, mean=1.0, dtype=float)

# Semidefinite matrices
S = ein.utils.create_random_semidefinite("name", rows, mean=1.0, force_zeros=1, dtype=float)

# Factory with method choice ("einsums" or "numpy")
T = ein.utils.tensor_factory("name", [dims], dtype=float, method="einsums")
R = ein.utils.random_tensor_factory("name", [dims], dtype=float, method="einsums")
```

Supported dtypes: `float`, `np.float32`, `np.float64`, `complex`, `np.complex64`, `np.complex128`

## Tensor Operations

### Properties
```python
A.rank()      # Number of dimensions
A.size()      # Total elements
A.dims()      # List of dimensions
A.dim(i)      # Dimension at index i
A.strides()   # List of strides
A.stride(i)   # Stride at index i
A.name        # Tensor name (get/set)
A.get_name()  # Get tensor name
A.set_name(s) # Set tensor name
```

### Indexing and Slicing
```python
# Single element access
val = A[i, j]

# Slicing returns RuntimeTensorView
view = A[0:2, :]
view = A[0]        # First row (or first slice for rank > 2)

# Assignment
A[i, j] = value
A[0:2, :] = np.array(...)
A[:, :] = 0.0      # Fill with scalar
```

### In-place Operations
```python
A.zero()           # Set all elements to 0
A.set_all(value)   # Set all elements to value
A += B             # Element-wise add
A -= B             # Element-wise subtract
A *= B             # Element-wise multiply
A /= B             # Element-wise divide
A += scalar        # Add scalar to all
```

### Iteration
```python
for val in A:      # Iterate over elements
    print(val)

# Index iteration via ein.utils.TensorIndices
for idx in ein.utils.TensorIndices(A):
    print(idx, A[idx])
```

### NumPy Interoperability
```python
# Tensor to numpy (creates view when possible)
arr = np.array(A)

# Numpy to tensor (copies)
T = ein.core.RuntimeTensorD(numpy_array)
```

## Linear Algebra

All functions in `ein.core`:

### Matrix Multiplication
```python
# C = alpha * op(A) @ op(B) + beta * C
ein.core.gemm(trans_a, trans_b, alpha, A, B, beta, C)
# trans_a/trans_b: "N" (no transpose), "T" (transpose), "C" (conjugate transpose)

# y = alpha * op(A) @ x + beta * y  
ein.core.gemv(trans_a, alpha, A, x, beta, y)
```

### Eigendecomposition
```python
# Symmetric/Hermitian eigendecomposition
# A is overwritten with eigenvectors, eigenvalues in w
ein.core.syev(jobz, A, w)  
ein.core.heev(jobz, A, w)  # Alias for complex
# jobz: "V" (compute vectors), "N" (eigenvalues only)

# General eigendecomposition
ein.core.geev(A, w, lvecs, rvecs)
```

### Matrix Operations
```python
ein.core.scale(alpha, A)              # A *= alpha
ein.core.scale_row(row, alpha, A)     # Scale row
ein.core.scale_column(col, alpha, A)  # Scale column
ein.core.axpy(alpha, X, Y)            # Y += alpha * X
ein.core.axpby(alpha, X, beta, Y)     # Y = alpha*X + beta*Y
ein.core.ger(alpha, x, y, A)          # A += alpha * outer(x, y)
ein.core.dot(A, B)                    # Sum of element products
ein.core.true_dot(A, B)               # Dot with conjugate of A
```

### Factorizations
```python
ein.core.getrf(A, pivot)              # LU factorization
ein.core.getri(A, pivot)              # Inverse from LU
ein.core.invert(A)                    # Direct matrix inverse
ein.core.svd(A)                       # Returns (U, S, Vt)
ein.core.svd_dd(A, job=Vectors.All)   # SVD via divide-and-conquer
ein.core.truncated_svd(A, k)          # Truncated SVD
ein.core.qr(A)                        # QR factorization
ein.core.q(qr_result, tau)            # Extract Q from QR
ein.core.r(qr_result, tau)            # Extract R from QR
```

### Solvers
```python
ein.core.gesv(A, B)                   # Solve AX = B
ein.core.pseudoinverse(A)             # Moore-Penrose pseudoinverse
ein.core.solve_continuous_lyapunov(A, Q)  # Solve AX + XA^T = Q
```

### Norms
```python
ein.core.norm(norm_type, A)           # Matrix norm
ein.core.vec_norm(A)                  # Vector 2-norm
ein.core.sum_square(A)                # Returns (sumsq, scale)
ein.core.det(A)                       # Matrix determinant

# Norm types (ein.core.Norm enum):
Norm.MAXABS, Norm.ONE, Norm.INFINITY, Norm.FROBENIUS
```

### Miscellaneous
```python
ein.core.direct_product(alpha, A, B, beta, C)  # C = alpha*A*B + beta*C
ein.core.truncated_syev(A, k)                   # Truncated eigendecomposition
```

## Einsum Plans

Compile-time optimized contraction plans:

```python
# Compile a plan from index notation
plan = ein.core.compile_plan("ij,jk->ik", A, B, C)

# Execute the plan: C = C_prefactor*C + AB_prefactor*A@B
plan.execute(C_prefactor, C, AB_prefactor, A, B)
```

Plan types (auto-selected based on contraction pattern):
- `EinsumGenericPlan` - General fallback
- `EinsumDotPlan` - Optimized dot product
- `EinsumGemmPlan` - Matrix multiplication
- `EinsumGemvPlan` - Matrix-vector product
- `EinsumGerPlan` - Outer product
- `EinsumDirectProductPlan` - Element-wise product

## Profiling

```python
# Manual section timing
section = ein.core.Section("label")
# ... code to profile ...
section.end()  # Or let destructor handle it

# Decorator for function profiling
@ein.utils.labeled_section
def my_function():
    pass

@ein.utils.labeled_section("custom label")
def my_function_with_label():
    pass
```

## Configuration

```python
config = ein.core.GlobalConfigMap.get_singleton()
config.get_str("key")
config.get_int("key")
config.get_float("key")
config.get_bool("key")
config.set_str("key", "value")
config.set_int("key", 42)
config.set_float("key", 3.14)
config.set_bool("key", True)
```

## Logging

```python
ein.core.log(level, "message")
ein.core.log_trace("message")
ein.core.log_debug("message")
ein.core.log_info("message")
ein.core.log_warn("message")
ein.core.log_error("message")
ein.core.log_critical("message")
```

## GPU Support

Check availability:
```python
if ein.core.gpu_enabled():
    # GPU operations available
    pass
```

GPU tensor views are available as `ein.core.PyGPUView` when compiled with GPU support.
