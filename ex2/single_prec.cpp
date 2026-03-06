#include <iostream>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <cmath>
#include <string>
#include <vector>
#include <numbers>


template <typename Precision>
void compute_basel(int start_n, int end_n)
{
    // True value in double precision
    double true_sum = std::numbers::pi_v<double> * std::numbers::pi_v<double> / 6.0;

    // Typename determined at compile-time
    std::string type_name = std::is_same_v<Precision, float> ? "float" : "double";
    std::string filename = "output_" + type_name + ".txt";
    
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Errore nell'apertura del file " << filename << "!\n";
        return;
    }

    file << std::left << std::setw(20) << "N dim" 
         << std::setw(20) << "Error Normal" 
         << "Error Reverse\n\n";

    int num_points = 40;

    std::vector<Precision> log_vector;
    Precision delta = static_cast<Precision>(end_n - start_n) / static_cast<Precision>((num_points - 1));
    for (int i = 0; i < num_points; ++i) {
        log_vector.push_back(std::pow(10, start_n + i * delta));
    }

    for (size_t i = 0; i < static_cast<size_t>(num_points); i++) {
        Precision n_int = log_vector[i];
        // --- Normal ordering sum (greatest to smallest) ---
        Precision out_norm = 0;
        for (size_t n = 1; n <= n_int; ++n) {
            Precision term = Precision(1.0) / (Precision(n) * Precision(n));
            out_norm += term;
        }
        Precision err_norm = std::abs(true_sum - (out_norm));

        // --- Inverse sum (smallest to greatest) ---
        Precision out_inverse = 0;
        for (size_t n = n_int; n > 0; --n) {
            Precision term = Precision(1.0) / (Precision(n) * Precision(n));
            out_inverse += term;
        }
        Precision err_inverse = std::abs(true_sum - (out_inverse));

        file << std::scientific << std::setprecision(6);
        file << std::left << std::setw(20) << (n_int)
            << std::setw(20) << err_norm
            << err_inverse << "\n";
    }
}

#include <iostream>
#include <string> // Necessario per std::stoi

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "ERROR, Needed: " << argv[0] << " <float_start> <float_end> <double_start> <double_end>" << std::endl;
        return 1;
    }

    int f_start = std::stoi(argv[1]);
    int f_end   = std::stoi(argv[2]);

    int d_start = std::stoi(argv[3]);
    int d_end   = std::stoi(argv[4]);

    std::cout << "Starts single precision computing (float) between 10^" << f_start << " and 10^" << f_end << "..." << std::endl;
    compute_basel<float>(f_start, f_end);
    std::cout << "Computing ends\n\n";
    
    std::cout << "Starts double precision computing (double) between 10^" << d_start << " and 10^" << d_end << "..." << std::endl;
    compute_basel<double>(d_start, d_end);
    std::cout << "Computing ends\n\n";

    return 0;
}