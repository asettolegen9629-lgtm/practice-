/*
 * Lab 1: The Amdahl Reality Gap
 * Student ID: 230103240 (N = 13240000)
 * Compilation: gcc -O2 -fopenmp collatz.c -o collatz -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define MOD 1000000007ULL


static inline uint32_t collatz_steps(uint64_t n) {
    uint32_t steps = 0;
    while (n > 1) {
        if ((n & 1) == 0) n >>= 1;
        else n = 3 * n + 1;
        steps++;
    }
    return steps;
}

int main(int argc, char* argv[]) {
    uint64_t N = 13240000ULL; 
    printf("==================================================\n");
    printf("   LAB 1: THE AMDAHL REALITY GAP\n");
    printf("   Student ID: 230103240 | Target Workload N: %llu\n", (unsigned long long)N);
    printf("==================================================\n\n");

   
    printf("--- Phase 2: Sequential Baseline Runs ---\n");
    double seq_times[3];
    uint32_t seq_max_steps = 0;
    uint64_t seq_checksum = 0;

    for (int run = 0; run < 3; run++) {
        double start = omp_get_wtime();
        uint32_t local_max = 0;
        uint64_t local_sum = 0;

        for (uint64_t i = 1; i <= N; i++) {
            uint32_t steps = collatz_steps(i);
            if (steps > local_max) local_max = steps;
            local_sum = (local_sum + steps) % MOD;
        }

        double end = omp_get_wtime();
        seq_times[run] = end - start;
        seq_max_steps = local_max;
        seq_checksum = local_sum;
        printf("Run %d (Cold/Warm): %.6f seconds\n", run + 1, seq_times[run]);
    }

    
    double T_seq = (seq_times[1] + seq_times[2]) / 2.0;
    printf("\n[Baseline Result] T_seq = %.6f s | Max Steps = %u | Checksum = %llu\n\n", 
           T_seq, seq_max_steps, (unsigned long long)seq_checksum);

    printf("--- Phase 4 (Exp A): False Sharing Benchmark ---\n");
    int max_threads = omp_get_max_threads();
    printf("Testing with Maximum Logical Threads: %d\n", max_threads);

    
    int *hits_naive = (int*)calloc(max_threads, sizeof(int));
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(max_threads)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (uint64_t i = 1; i <= N; i++) {
            if (collatz_steps(i) > 100) {
                hits_naive[tid]++;
            }
        }
    }
    double t_false_sharing = omp_get_wtime() - t0;
    uint64_t total_hits_1 = 0;
    for (int t = 0; t < max_threads; t++) total_hits_1 += hits_naive[t];
    free(hits_naive);

    
    uint64_t total_hits_reduction = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(max_threads) schedule(static) reduction(+:total_hits_reduction)
    for (uint64_t i = 1; i <= N; i++) {
        if (collatz_steps(i) > 100) {
            total_hits_reduction++;
        }
    }
    double t_mitigated = omp_get_wtime() - t0;

    printf("Variant 1 (Naive - False Sharing): Time = %.6f s | Hits = %llu\n", 
           t_false_sharing, (unsigned long long)total_hits_1);
    printf("Variant 2 (OpenMP Reduction):     Time = %.6f s | Hits = %llu\n", 
           t_mitigated, (unsigned long long)total_hits_reduction);
    printf("Speedup Penalty Ratio (V1 / V2):   %.2fx slower\n\n", 
           t_false_sharing / t_mitigated);

    return 0;
}                   