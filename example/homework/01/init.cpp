#include <Kokkos_Core.hpp>
#include <cstdio>

// Problem: Link and run program with Kokkos where you initialize a View and print out its name with the $.label()$ method.

int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);

        // Create a View named "myView"
        Kokkos::View<double*> myView("myView", 10);

        // Print the name of the View
        Kokkos::parallel_for("printLabel", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1), KOKKOS_LAMBDA(int i) {
                printf("View label: %s\n", myView.label().c_str());
        });
        return 0;
}