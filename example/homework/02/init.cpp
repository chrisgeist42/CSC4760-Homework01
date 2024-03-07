#include <Kokkos_Core.hpp>
#include <cstdio>

// Problem: Make an n ∗ m View where each index equals 1000 ∗ i ∗ j

int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard kokkos(argc, argv);
        {
                // set n and m, you can change these values
                int n = 16, m = 16;

                // Make View
                // Create a 2D Kokkos View with dimensions n × m
                Kokkos::View<double**> myView("my_view", n, m);

                // Initialize the View (optional)
                Kokkos::deep_copy(myView, 0.0);

                // Access and modify elements
                for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < m; ++j) {
                                // set values to 1000 * i * j);
                                myView(i, j) = 1000.0 * i * j;
                        }
                }

                // Print the contents of the view
                for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < m; ++j) {
                                printf("myView(%d, %d) = %f\n", i, j, myView(i, j));
                        }
                }
        return 0;
        }
}