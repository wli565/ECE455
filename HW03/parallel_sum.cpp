#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int N = 1000000;
    std::vector<int> arr(N);

    // Initialize array (for example: all 1s)
    for (int i = 0; i < N; i++) {
        arr[i] = 1;
    }

    long long sum = 0;

    // Parallel sum with OpenMP
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }

    std::cout << "Sum = " << sum << std::endl;

    return 0;
}
