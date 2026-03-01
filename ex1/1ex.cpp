#include <iostream>
#include <iomanip> // for std::scientific

int main() {
    // --- Part 1: single precision ---
    std::cout << "Single precision multiplication:\n";
    float f = 1.2e34f; // single precision
    for (int i = 0; i < 24; ++i) {
        f *= 2.0f;
        std::cout << "f = " << std::scientific << f << '\n';
    }

    // --- Part 2: double precision multiplication ---
    std::cout << "\nDouble precision multiplication:\n";
    double d = 1.2e304; // double precision
    for (int i = 0; i < 24; ++i) {
        d *= 2.0;
        std::cout << "d = " << std::scientific << d << '\n';
    }

    // --- Part 3: double precision division ---
    std::cout << "\nDouble precision division:\n";
    d = 1e-13;
    for (int i = 0; i < 24; ++i) {
        d /= 2.0;
        std::cout << "d = " << std::scientific << d 
                  << ", 1+d = " << std::scientific << (1.0 + d) << '\n';
    }

    // --- Part 4: single precision division ---
    std::cout << "\nSingle precision division:\n";
    f = 1e-13f;
    for (int i = 0; i < 24; ++i) {
        f /= 2.0f;
        std::cout << "f = " << std::scientific << f 
                  << ", 1+f = " << std::scientific << (1.0f + f) << '\n';
    }

    return 0;
}