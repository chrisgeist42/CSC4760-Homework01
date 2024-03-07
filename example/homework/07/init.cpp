#include <Kokkos_Core.hpp>
#include <cstdio>


int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    // Assuming n is already defined
    int n = 10;

    // Declare a 1D View
    Kokkos::View<double*> view("view", n);

    // Initialize the View 
    Kokkos::parallel_for("InitView", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), KOKKOS_LAMBDA (const int i) {
        view(i) = i;
    });

    // Declare a view to hold the prefix sum
    Kokkos::View<double*> prefix_sum("prefix_sum", n);

    // Conduct prefix sum using parallel_scan
    Kokkos::Timer timer;
    Kokkos::parallel_scan("PrefixSum", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), KOKKOS_LAMBDA (const int i, double& update, const bool final) {
        update += view(i);
        if (final) {
            prefix_sum(i) = update;
        }
    });
    double time = timer.seconds();

    // Print the prefix sum and timer results
    printf("Prefix sum:\n");
    for (int i = 0; i < n; ++i) {
        printf("%f\n", prefix_sum(i));
    }
    printf("Time: %f seconds\n", time);

    return 0;
}