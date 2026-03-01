// OUTPUT ON FILE

// #include <cstdio>   // for FILE, fopen, fprintf, fclose

// int main() {
//     FILE* file = fopen("output.txt", "w");
//     if (file == nullptr) {
//         printf("Failed to open file");
//         return 1;
//     }

//     // Print multiple rows: integer and double
//     for (int i = 1; i <= 5; ++i) {
//         double value = i * 1.5;
//         fprintf(file, "%d %.2f\n", i, value);
//     }

//     fclose(file);

//     return 0;
// }