#include <Kokkos_Core.hpp>
#include <cstdio>

// Create a program that compares a parallel for loop and a standard for loop for summing rows of a View with Kokkos Timer.

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  {
  // Make View and create values
  // Assuming n is already defined
    int n = 10000;

    // Declare a 2D View
    Kokkos::View<double**> view("view", n, n);

    // Initialize the View with random values
    Kokkos::parallel_for("InitView", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {n,n}), KOKKOS_LAMBDA (const int i, const int j) {
        view(i,j) = static_cast<double>(rand()) / RAND_MAX;
    });
    //sum loops
    // Use a parallel for loop to sum the rows
    Kokkos::View<double*> rowSums("rowSums", n);
    Kokkos::Timer timer;
    Kokkos::parallel_for("RowSumParallel", n, KOKKOS_LAMBDA (const int i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += view(i,j);
        }
        rowSums(i) = sum;
    });
    double parallelTime = timer.seconds();

    // Use a standard for loop to sum the rows
    Kokkos::View<double*> rowSumsStandard("rowSumsStandard", n);
    timer.reset();
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += view(i,j);
        }
        rowSumsStandard(i) = sum;
    }
    double standardTime = timer.seconds();

    // Output times
    printf("Parrallel Time: %f\n", parallelTime);
    printf("Standard Time: %f\n", standardTime);

    return 0;
  }
}