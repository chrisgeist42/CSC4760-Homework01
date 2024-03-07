#include <Kokkos_Core.hpp>
#include <cstdio>

// Do simple parallel reduce to output the maximum element in a View

int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard kokkos(argc, argv);
  {
  // Make View and create values
  // Assuming n is already defined

    int n = 10;
    // Declare a 4D View
    Kokkos::View<double****> view("view", 5, 7, 12, n);

    // Initialize the View 
    Kokkos::parallel_for("InitView", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {5,7,12,n}), KOKKOS_LAMBDA (const int i, const int j, const int k, const int l) {
        view(i,j,k,l) = static_cast<double>(rand()) / RAND_MAX;
    });

    // Use parallel_reduce to find the maximum element
    double maxVal;
    Kokkos::parallel_reduce("MaxElement", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,0,0,0}, {5,7,12,n}), KOKKOS_LAMBDA (const int i, const int j, const int k, const int l, double& localMax) {
        if (view(i,j,k,l) > localMax) localMax = view(i,j,k,l);
    }, Kokkos::Max<double>(maxVal));

    // Print the maximum value
    printf("%f\n", maxVal);
  }
}