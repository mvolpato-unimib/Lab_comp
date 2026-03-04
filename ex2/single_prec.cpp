#include <iostream>
#include <fstream>
#include <iomanip>
#include <numbers>
#include <cmath>
#include <string>
#include <concepts> // Per std::floating_point

// Limitiamo il template solo ai tipi floating point (float, double)
template <std::floating_point Precision>
void compute_basel(Precision start_n, Precision end_n)
{
    // Il valore vero in double precision
    constexpr double true_sum = std::numbers::pi_v<double> * std::numbers::pi_v<double> / 6.0;

    // Determiniamo il nome del tipo a compile-time
    std::string type_name = std::is_same_v<Precision, float> ? "float" : "double";
    std::string filename = "output_" + type_name + ".txt";

    // Usiamo std::ofstream del C++ al posto di FILE* e fopen
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Errore nell'apertura del file " << filename << "!\n";
        return;
    }

    // Intestazione
    file << std::left << std::setw(15) << "N dim" 
         << std::setw(20) << "Error Normal" 
         << "Error Reverse\n\n";

    const size_t num_points = 20;
    // Calcoliamo i logaritmi in double per evitare perdite di precisione negli indici
    double log_start = std::log(static_cast<double>(start_n));
    double log_end   = std::log(static_cast<double>(end_n));

    for (size_t i = 0; i < num_points; i++) {
        double n_max_double = std::exp(log_start + (log_end - log_start) * i / (num_points - 1));
        size_t n_int = static_cast<size_t>(n_max_double);

        // --- Somma Normale (dal più grande al più piccolo) ---
        Precision out_norm = 0;
        for (size_t n = 1; n <= n_int; ++n) {
            Precision term = Precision(1.0) / (Precision(n) * Precision(n));
            out_norm += term;
        }
        double err_norm = std::abs(true_sum - static_cast<double>(out_norm));

        // --- Somma Inversa (dal più piccolo al più grande) ---
        // Eliminiamo il vector: risparmiamo GIGABYTES di RAM e iteriamo al contrario
        Precision out_inverse = 0;
        for (size_t n = n_int; n > 0; --n) {
            Precision term = Precision(1.0) / (Precision(n) * Precision(n));
            out_inverse += term;
        }
        double err_inverse = std::abs(true_sum - static_cast<double>(out_inverse));

        // Scrittura formattata su file
        file << std::scientific << std::setprecision(10);
        file << std::left << std::setw(15) << static_cast<double>(n_int)
             << std::setw(20) << err_norm
             << err_inverse << "\n";
    }

    // Il file si chiude automaticamente grazie al distruttore di std::ofstream
}

int main() {  
    std::cout << "Inizio calcolo in singola precisione (float)..." << std::endl;
    compute_basel<float>(1e3f, 1e8f);
    std::cout << "Calcolo in singola precisione terminato.\n\n";

    std::cout << "Inizio calcolo in doppia precisione (double)..." << std::endl;
    compute_basel<double>(1e7, 1e10);
    std::cout << "Calcolo in doppia precisione terminato.\n";
    
    return 0;
}