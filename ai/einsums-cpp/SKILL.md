---
name: einsums-cpp
description: "Guide for C++ projects linking against the Einsums tensor algebra library. Use when writing C++ code with Einsums tensors, implementing tensor contractions via einsum(), using BLAS-backed linear algebra, debugging Einsums C++ code, or optimizing tensor operations. Triggers on Einsums includes, Tensor templates, einsum(), TensorView, or einsums namespace usage."
---

# Einsums C++

C++20 tensor algebra library with compile-time contraction optimization to BLAS operations.

## Quick Start

```cpp
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/LinearAlgebra.hpp>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::index;

int main() {
    // Create tensors
    auto A = create_random_tensor<double>("A", 3, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto C = create_tensor<double>("C", 3, 5);
    
    // Einstein summation: C_ij = A_ik * B_kj
    einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
    
    return 0;
}
```

## Core Concepts

### Tensor Types
```cpp
Tensor<T, Rank>      // Owning tensor (T=type, Rank=dimensions)
TensorView<T, Rank>  // Non-owning view into tensor
```

### Index Notation
Use `einsums::index` namespace for index labels: `i, j, k, l, m, n, ...`

```cpp
using namespace einsums::index;
// Now i, j, k, etc. are available as index labels
```

## Common Patterns

### Tensor Creation
```cpp
// Factory functions
auto A = create_tensor<double>("A", m, n);       // Zero-initialized
auto B = create_random_tensor<double>("B", m, n); // Random values
auto C = create_ones_tensor<double>("C", m, n);   // All ones

// Direct construction
Tensor<double, 2> D("D", m, n);
```

### Element Access
```cpp
A(i, j) = 1.0;        // Set element
double val = A(i, j); // Get element
A.zero();             // Zero all elements
A.set_all(value);     // Fill with value
```

### Views and Slicing
```cpp
using einsums::All;
using einsums::Range;

auto row = A(Range{0, 1}, All);           // First row
auto col = A(All, Range{0, 1});           // First column  
auto block = A(Range{0, 2}, Range{0, 2}); // 2x2 subblock
```

### Einstein Summation
```cpp
// Basic: C = A * B (matrix multiply)
einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

// With prefactors: C = beta*C + alpha*A*B
einsum(beta, Indices{i, j}, &C, alpha, Indices{i, k}, A, Indices{k, j}, B);

// Trace
double trace;
einsum(Indices{}, &trace, Indices{i, i}, A);

// Outer product
einsum(Indices{i, j}, &C, Indices{i}, x, Indices{j}, y);
```

### Linear Algebra
```cpp
using namespace einsums::linear_algebra;

// Matrix multiply: C = alpha*A*B + beta*C
gemm<false, false>(alpha, A, B, beta, &C);
// Template args: <TransA, TransB>

// Eigendecomposition (symmetric)
auto [evecs, evals] = syev(A);
// Or in-place:
syev(&A, &w);  // A overwritten with eigenvectors

// Other operations
scale(alpha, &A);            // A *= alpha
axpy(alpha, X, &Y);          // Y += alpha*X
invert(&A);                  // In-place inverse
auto d = dot(A, B);          // Element-wise dot
auto [U, S, Vt] = svd(A);    // SVD decomposition
```

## Performance Tips

1. **Use einsum()** - Automatically selects optimal BLAS routine (gemm, gemv, dot, ger)
2. **Avoid unnecessary copies** - Use views when possible
3. **Profile with LabeledSection**:
   ```cpp
   {
       LabeledSection section("computation");
       // ... code ...
   }
   ```
4. **Prefer compile-time rank** - Use `Tensor<T, 2>` over `RuntimeTensor<T>` when rank is known

## CMake Integration

```cmake
find_package(Einsums REQUIRED)
target_link_libraries(myapp PRIVATE Einsums::Einsums)
```

## Debugging Tips

### Common Issues

**Dimension mismatch**: Check tensor dimensions match contraction pattern
```cpp
// Debug: print tensor info
println(A);  // Prints dimensions and values
```

**BLAS errors**: Ensure linked against compatible BLAS (OpenBLAS, MKL, etc.)

**Concept constraint failures**: Verify template args satisfy:
- `TensorConcept` - any tensor type
- `MatrixConcept` - rank-2 tensor
- `VectorConcept` - rank-1 tensor

## API Reference

For complete API documentation, see [references/api_reference.md](references/api_reference.md).

Key sections:
- Tensor types and creation
- Einstein summation syntax
- Linear algebra functions
- C++20 concepts
- CMake linking
