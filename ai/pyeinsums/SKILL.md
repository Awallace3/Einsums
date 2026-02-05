---
name: pyeinsums
description: "Guide for Python applications using the einsums tensor library. Use when writing Python code that imports einsums, creating tensor operations, using einsum contractions, performing linear algebra with einsums tensors, debugging pyeinsums code, or optimizing tensor computations. Triggers on import einsums, ein.core, RuntimeTensor, einsum plans, or tensor algebra in Python."
---

# PyEinsums

Python bindings for the Einsums C++ tensor algebra library. Provides high-performance tensor operations with BLAS optimization.

## Quick Start

```python
import einsums as ein
import numpy as np

# Create tensors
A = ein.core.RuntimeTensorD("A", [3, 4])  # 3x4 double tensor
B = ein.utils.create_random_tensor("B", [4, 5], dtype=float)

# Matrix multiplication
C = ein.utils.create_tensor("C", [3, 5], dtype=float)
ein.core.gemm("N", "N", 1.0, A, B, 0.0, C)

# NumPy interop
arr = np.array(C)
T = ein.core.RuntimeTensorD(numpy_array)
```

## Tensor Types

Type suffixes: `D`=float64 (default), `F`=float32, `Z`=complex128, `C`=complex64

```python
ein.core.RuntimeTensorD  # double precision
ein.core.RuntimeTensorF  # single precision  
ein.core.RuntimeTensorZ  # complex double
ein.core.RuntimeTensorC  # complex single
```

Use `ein.utils.create_tensor()` for dtype-based creation:
```python
T = ein.utils.create_tensor("name", [dims], dtype=np.float64)
R = ein.utils.create_random_tensor("name", [dims], dtype=complex)
```

## Common Patterns

### Tensor Creation
```python
# Direct construction
A = ein.core.RuntimeTensorD("A", [m, n])
B = ein.core.RuntimeTensorD([m, n])  # unnamed
C = ein.core.RuntimeTensorD(numpy_array)  # from numpy

# Factory functions (recommended)
T = ein.utils.create_tensor("T", [m, n], dtype=float)
R = ein.utils.create_random_tensor("R", [m, n], dtype=float)
P = ein.utils.create_random_definite("P", n, dtype=float)  # positive definite
```

### Indexing and Slicing
```python
val = A[i, j]           # Get element
A[i, j] = 1.0           # Set element
row = A[0, :]           # Slice → RuntimeTensorView
block = A[0:2, 1:3]     # Sub-block
A[:, :] = 0.0           # Fill with scalar
```

### Linear Algebra (Preferred)
```python
# Matrix multiply: C = A @ B
ein.core.gemm("N", "N", 1.0, A, B, 0.0, C)
# "N"=no transpose, "T"=transpose, "C"=conjugate transpose

# Eigendecomposition (symmetric)
ein.core.syev("V", A, eigenvalues)  # "V"=vectors, "N"=values only

# Other operations
ein.core.scale(alpha, A)            # A *= alpha
ein.core.axpy(alpha, X, Y)          # Y += alpha*X
ein.core.dot(A, B)                  # Element-wise dot
ein.core.invert(A)                  # In-place inverse
```

### Einsum Plans (Advanced, Harder Contractions)
```python
# Compile optimized contraction plan
plan = ein.core.compile_plan("ij,jk->ik", A, B, C)

# Execute: C = C_pre*C + AB_pre*(A contracted with B)
plan.execute(0.0, C, 1.0, A, B)  # C_prefactor=0 zeroes C first
```

## Performance Tips

1. **Reuse einsum plans** for repeated contractions with same shapes
2. **Use appropriate dtype** - don't use complex when real suffices
3. **Prefer BLAS operations** (gemm, gemv) over element loops
4. **Profile with Section**:
   ```python
   section = ein.core.Section("computation")
   # ... code ...
   section.end()
   ```
5. **Use decorator for function profiling**:
   ```python
   @ein.utils.labeled_section
   def my_function():
       pass
   ```

## Debugging

### Common Issues

**Import errors**: Ensure einsums is built with Python bindings (`EINSUMS_BUILD_PYTHON=ON`)

**Type mismatches**: Match tensor type suffix to numpy dtype:
- `RuntimeTensorD` ↔ `np.float64`
- `RuntimeTensorF` ↔ `np.float32`
- `RuntimeTensorZ` ↔ `np.complex128`
- `RuntimeTensorC` ↔ `np.complex64`

**Dimension errors**: Check `tensor.dims()` and `tensor.rank()`

**NaN/Inf values**: Call `tensor.zero()` after creation if needed

## API Reference

For complete API documentation, see [references/api_reference.md](references/api_reference.md).

Key sections:
- Tensor creation and operations
- Linear algebra functions (gemm, syev, svd, etc.)
- Einsum plans and optimization
- Profiling utilities
