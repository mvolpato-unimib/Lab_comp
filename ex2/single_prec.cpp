#include <iostream>
#include <numbers>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <type_traits>
#include <string>

using namespace std;

template <typename precision>
double Basel_generic(precision StartN, precision EndN)
{
    double True_Sum = std::numbers::pi_v <double>
                         * std::numbers::pi_v<double> / double(6);

    std::string nameType = "";

    if constexpr (std::is_same_v<precision, float>) {
        nameType = "float";
    }
    else if constexpr (std::is_same_v<precision, double>) {
        nameType = "double";
    }
    else {
        printf("Error, no valid type inserted (float/double)!\n");
        exit(EXIT_FAILURE);
    }

    std::string filename = "output_" + nameType + ".txt";

    FILE* file = fopen(filename.c_str(), "w");
    if (file == nullptr) {
        printf("Failed to open file\n");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "N dim        Error Normal      Error Reverse\n\n");

    size_t num_points = 100; // numero di punti desiderati
    precision log_start = std::log(StartN);
    precision log_end   = std::log(EndN);

    for (size_t i = 0; i < num_points; i++) {
        precision N_max = std::exp(log_start + (log_end - log_start) * i / (num_points - 1));
        size_t N_int = static_cast<size_t>(N_max);
        std::vector<precision> arr(N_int);

        for (size_t N = 1; N <= N_int; N++) {
            arr[N - 1] = precision(1) / (precision(N) * precision(N));
        }

        precision out_norm = std::accumulate(
            arr.begin(), arr.end(), precision(0)
        );
        precision err_norm = std::abs(True_Sum - out_norm);

        std::reverse(arr.begin(), arr.end());

        precision out_inverse = std::accumulate(
            arr.begin(), arr.end(), precision(0)
        );
        precision err_inverse = std::abs(True_Sum - out_inverse);

        fprintf(file, "%.0f       %.10e       %.10e\n",
                static_cast<double>(N_max),
                static_cast<double>(err_norm),
                static_cast<double>(err_inverse));
    }

    fclose(file);

    return True_Sum;
}

int main() {  
    double True_Sum = Basel_generic<float>(1e3, 1e8);
    printf("Single precision computing ended\n");
    // Basel_generic<double>(1e7, 1e10);
    // printf("Double precision computing ended\n");
    
    // printf("\nTrue Sum\n");
    // printf("%.10e\n", True_Sum);    
    return 0;
}