#include <Kokkos_Core.hpp>
#include <cstdio>

// Create a program that does matrix addition between a 2D View and a 1D View with at least one loop of parallelism.
// For a test case:
// a = [130, 137, 115]   b = [221]
//     [224, 158, 187]       [12]
//     [ 54, 211, 120]       [157]
// Extra credit: make a function and check for correct shape/dimensions

// Function to add a vector to each row of a matrix
void addVectorToMatrixRows(Kokkos::View<double**>& matrix, Kokkos::View<double*>& vector) {

    // Check for correct dimensions
    if (matrix.extent(1) != vector.extent(0)) {
        printf("Error: The number of columns in the matrix must be equal to the size of the vector.\n");
        return;
    }

    // Add the vector to each row of the matrix
    Kokkos::parallel_for("AddVectorToMatrixRows", matrix.extent(0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < matrix.extent(1); ++j) {
            matrix(i,j) += vector(j);
        }
    });
}
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  {
  // Make View and add values
  // Create a 2D View for the matrix

    Kokkos::View<double**> A("A", 3, 3);

    // Initialize the matrix with your values
    A(0,0) = 130; A(0,1) = 147; A(0,2) = 115;
    A(1,0) = 224; A(1,1) = 158; A(1,2) = 187;
    A(2,0) = 54;  A(2,1) = 158; A(2,2) = 120;

    // Create a 1D View for the vector
    Kokkos::View<double*> B("B", 3);

    // Initialize the vector with your values
    B(0) = 221; B(1) = 12; B(2) = 157;

    // Do a matrix add
    addVectorToMatrixRows(A, B);

    // Output addition
    Kokkos::parallel_for("PrintMatrix", A.extent(0), KOKKOS_LAMBDA (const int i) {
        for (int j = 0; j < A.extent(1); ++j) {
            printf("%f ", A(i,j));
        }
        printf("\n");
    });
  }
}