# Einsums C++ API Reference

## Table of Contents
- [Includes](#includes)
- [Tensor Types](#tensor-types)
- [Tensor Creation](#tensor-creation)
- [Tensor Operations](#tensor-operations)
- [TensorView](#tensorview)
- [Einstein Summation](#einstein-summation)
- [Linear Algebra](#linear-algebra)
- [Concepts](#concepts)
- [Profiling](#profiling)
- [Linking](#linking)

## Includes

```cpp
#include <Einsums/Tensor/Tensor.hpp>        // Tensor<T, Rank>
#include <Einsums/TensorAlgebra.hpp>        // einsum()
#include <Einsums/LinearAlgebra.hpp>        // gemm, syev, etc.
#include <Einsums/Print.hpp>                // println()
#include <Einsums/Runtime.hpp>              // initialize/finalize
#include <Einsums/Profile/LabeledSection.hpp>  // Profiling
```

## Tensor Types

```cpp
namespace einsums {

// Fixed-rank tensor
template <typename T, size_t Rank>
struct Tensor;

// Non-owning view into tensor
template <typename T, size_t Rank>
struct TensorView;

// Dynamic-rank tensor (used in Python bindings)
template <typename T>
struct RuntimeTensor;

// Specialized tensor types
template <typename T, size_t Rank> struct BlockTensor;      // Block-diagonal
template <typename T, size_t Rank> struct TiledTensor;      // Tiled storage
template <typename T, size_t Rank> struct DeviceTensor;     // GPU tensor
template <typename T, size_t Rank> struct DiskTensor;       // HDF5-backed
template <typename T, size_t Rank> struct FunctionTensor;   // Computed on-access

// Compile-time properties
tensor.Rank;                    // constexpr size_t
using ValueType = T::ValueType; // Element type
}
```

## Tensor Creation

### Construction
```cpp
using namespace einsums;

// Named tensor with dimensions
auto A = Tensor<double, 2>("A", 3, 3);

// Reshape from existing tensor (moves data)
auto B = Tensor<double, 2>(std::move(flat_tensor), "B", 3, 3);

// Use -1 for automatic dimension inference
auto C = Tensor<double, 3>(std::move(data), "C", 3, -1, 3);
```

### Factory Functions
```cpp
// Zero-initialized
auto Z = create_tensor<double>("Z", 3, 3);

// Ones-initialized
auto O = create_ones_tensor<double>("O", 3, 3);

// Random values
auto R = create_random_tensor<double>("R", 3, 3);

// Incremental values (0, 1, 2, ...)
auto I = create_incremented_tensor<double>("I", 3, 3);
```

## Tensor Operations

### Properties
```cpp
A.dim(i);           // Dimension at index i
A.dims();           // std::array of all dimensions
A.stride(i);        // Stride at index i
A.strides();        // std::array of all strides
A.size();           // Total number of elements
A.data();           // Raw pointer to data
A.name();           // Tensor name
A.full_view_of_underlying();  // Check if view covers full tensor
```

### Element Access
```cpp
// Single element (variadic indices)
A(i, j) = 1.0;
double val = A(i, j);

// Vector of indices
std::vector<size_t> idx{i, j};
A(idx) = 1.0;
```

### In-place Operations
```cpp
A.zero();           // Set all to 0
A.set_all(value);   // Set all to value

A += B;             // Element-wise add
A -= B;             // Element-wise subtract
A *= B;             // Element-wise multiply
A /= B;             // Element-wise divide
A *= scalar;        // Scale all elements
```

### Direct Data Access
```cpp
// Access underlying storage
VectorData<T>& vec = A.vector_data();
vec[linear_index] = value;

// Iterate elements
for (auto& elem : A.vector_data()) {
    elem *= 2.0;
}
```

## TensorView

Non-owning views into tensor regions:

```cpp
using namespace einsums;

// Slice using Range{start, stop}
auto row = A(Range{0, 1}, All);          // First row
auto col = A(All, Range{0, 1});          // First column
auto block = A(Range{0, 2}, Range{0, 2}); // 2x2 block

// Single index creates view with reduced rank for that dimension
auto row_view = A(Range{-1, i}, All);    // Range{-1, i} selects single index i

// Assign to views
row = 1.0;                               // Fill with scalar
block = other_tensor;                    // Copy from tensor
```

## Einstein Summation

Core function for tensor contractions:

```cpp
using namespace einsums;
using namespace einsums::tensor_algebra;

// Define index labels
using namespace einsums::index;  // Provides i, j, k, l, etc.

// Basic contraction: C_ij = A_ik * B_kj
Tensor<double, 2> A("A", 3, 4);
Tensor<double, 2> B("B", 4, 5);
Tensor<double, 2> C("C", 3, 5);
einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

// With prefactors: C = C_prefactor*C + AB_prefactor*A*B
einsum(0.0, Indices{i, j}, &C, 1.0, Indices{i, k}, A, Indices{k, j}, B);
// C_prefactor=0 means C is zeroed before accumulation

// Trace
double trace;
einsum(Indices{}, &trace, Indices{i, i}, A);

// Outer product: C_ij = A_i * B_j
einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B);

// Dot product (contraction to scalar)
einsum(Indices{}, &result, Indices{i}, A, Indices{i}, B);

// Hadamard (element-wise) product
einsum(Indices{i, j}, &C, Indices{i, j}, A, Indices{i, j}, B);
```

### Supported Operations (auto-detected)
- **GEMM**: Matrix-matrix multiplication
- **GEMV**: Matrix-vector multiplication
- **DOT**: Vector dot product
- **GER**: Outer product
- **Direct product**: Element-wise multiplication
- **Generic**: Element-by-element fallback

### Permutation
```cpp
// Reorder tensor axes
sort(Indices{j, i}, &B, Indices{i, j}, A);  // Transpose
```

### Element Transform
```cpp
// Apply function to each element
element_transform(&A, [](double x) { return std::sqrt(x); });
```

## Linear Algebra

All in `einsums::linear_algebra` namespace:

### Matrix Multiplication
```cpp
// C = alpha * op(A) * op(B) + beta * C
gemm<TransA, TransB>(alpha, A, B, beta, &C);

// Returns new tensor
auto C = gemm<false, false>(1.0, A, B);

// Matrix-vector: y = alpha * op(A) * x + beta * y
gemv<TransA>(alpha, A, x, beta, &y);
```

### Eigendecomposition
```cpp
// Symmetric: A overwritten with eigenvectors, w gets eigenvalues
syev(&A, &w);
syev<false>(&A, &w);  // Eigenvalues only

// Returns tuple (eigenvectors, eigenvalues)
auto [evecs, evals] = syev(A);

// Hermitian (complex)
heev(&A, &w);

// General
geev(&A, &w, &lvecs, &rvecs);
```

### Matrix Operations
```cpp
scale(alpha, &A);               // A *= alpha
scale_row(row, alpha, &A);
scale_column(col, alpha, &A);
axpy(alpha, X, &Y);             // Y += alpha * X
axpby(alpha, X, beta, &Y);      // Y = alpha*X + beta*Y
ger(alpha, x, y, &A);           // A += alpha * outer(x, y)
auto d = dot(A, B);             // Element-wise dot product
auto d = true_dot(A, B);        // Dot with conjugate of A
```

### Factorizations & Solvers
```cpp
// LU
getrf(&A, &pivot);              // LU factorization
getri(&A, pivot);               // Inverse from LU
invert(&A);                     // Direct inverse

// SVD
auto [U, S, Vt] = svd(A);
auto [U, S, Vt] = svd_dd(A);    // Divide-and-conquer
auto [U, S, Vt] = truncated_svd(A, k);

// QR
auto [qr_result, tau] = qr(A);
auto Q = q(qr_result, tau);

// Solve AX = B
gesv(&A, &B);

// Pseudoinverse
auto pinv = pseudoinverse(A, tolerance);

// Lyapunov equation AX + XA^T = Q
auto X = solve_continuous_lyapunov(A, Q);
```

### Norms
```cpp
auto n = norm(Norm::One, A);        // 1-norm
auto n = norm(Norm::Infinity, A);   // Infinity norm
auto n = norm(Norm::Frobenius, A);  // Frobenius norm
auto n = norm(Norm::MaxAbs, A);     // Max absolute value
auto n = vec_norm(v);               // Vector 2-norm
auto d = det(A);                    // Determinant
```

### Miscellaneous
```cpp
sum_square(A, &scale, &sumsq);  // Sum of squares
auto P = pow(A, alpha, cutoff); // Matrix power
direct_product(alpha, A, B, beta, &C);
```

## Concepts

C++20 concepts for generic programming:

```cpp
#include <Einsums/Concepts/TensorConcepts.hpp>

template <TensorConcept T>      // Any tensor type
template <MatrixConcept T>      // Rank-2 tensor
template <VectorConcept T>      // Rank-1 tensor
template <RankTensorConcept T>  // Tensor with compile-time rank
template <CoreTensorConcept T>  // Dense in-memory tensor
template <BasicTensorConcept T> // Basic tensor operations

// Complex type handling
template <Complex T>            // Complex-valued
template <NotComplex T>         // Real-valued
```

## Profiling

```cpp
#include <Einsums/Profile/LabeledSection.hpp>

// RAII profiling section
{
    LabeledSection section("computation");
    // ... code to profile ...
}

// Function-level macro
void my_function() {
    LabeledSection0();  // Uses function name
    // ...
}

// With additional label
void my_function(int param) {
    LabeledSection1(fmt::format("param={}", param));
    // ...
}
```

## Linking

### CMake Integration
```cmake
find_package(Einsums REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE Einsums::Einsums)
```

### Required Dependencies
- BLAS/LAPACK implementation
- HDF5 (for DiskTensor)
- fmt (formatting)
- OpenMP (optional, for parallelization)
- HIP/CUDA (optional, for GPU support)

### Compiler Requirements
- C++20 support required
- Tested: GCC 10+, Clang 12+, MSVC 2022+
