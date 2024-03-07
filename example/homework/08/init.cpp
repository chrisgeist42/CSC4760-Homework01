#include <Kokkos_Core.hpp>
#include <cstdio>

// Create a program that does matrix multiply between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions

Kokkos::View<double*> matVecMultiply(Kokkos::View<double**> A, Kokkos::View<double*> B) {
    // Check for correct dimensions
    if (A.extent(1) != B.extent(0)) {
        printf("Error: Dimensions do not match for matrix-vector multiplication.\n");
        exit(-1);
    }

    // Make a view and add values
    Kokkos::View<double*> C("C", A.extent(0));

    // Perform matrix-vector multiplion
    Kokkos::parallel_for("MatVecMultiply", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, A.extent(0)), KOKKOS_LAMBDA (const int i) {
        double sum = 0.0;
        for (int j = 0; j < A.extent(1); ++j) {
            sum += A(i, j) * B(j);
        }
        C(i) = sum;
    });

    return C;
}

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    //Make View and add values
    Kokkos::View<double**> A("A", 3, 3);
    // Initialize the matrix with your values
    A(0,0) = 130; A(0,1) = 147; A(0,2) = 115;
    A(1,0) = 224; A(1,1) = 158; A(1,2) = 187;
    A(2,0) = 54;  A(2,1) = 158; A(2,2) = 120;

    // Create a 1D View for the vector
    Kokkos::View<double*> B("B", 3);

    // Initialize the vector with your values
    B(0) = 221; B(1) = 12; B(2) = 157;

    // Do a matrix multiply
    Kokkos::View<double*> C = matVecMultiply(A, B);

    // Output addition
    for (int i = 0; i < C.extent(0); ++i) {
        printf("%f\n", C(i));
    }

    return 0;
}