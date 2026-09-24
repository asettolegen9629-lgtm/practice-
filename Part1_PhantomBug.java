import java.util.concurrent.ThreadLocalRandom;

/**
 * Part 1: The Phantom Bug
 *
 * 4 threads all hammer the SAME static variable with totalHits++.
 * That statement is really three steps: read totalHits, add 1, write it back.
 * When two threads interleave those steps, one thread's update gets clobbered.
 * Result: totalHits ends up lower than it should be -> pi comes out too low.
 *
 * Run this 5 times (as 5 separate JVM runs) and record the pi value each time.
 */
public class Part1_PhantomBug {
    static final long TOTAL_POINTS = 50_000_000L;
    static long totalHits = 0; // shared, NOT synchronized -> data race on purpose

    public static void main(String[] args) throws InterruptedException {
        int numThreads = 4;
        long pointsPerThread = TOTAL_POINTS / numThreads;

        Thread[] threads = new Thread[numThreads];
        for (int t = 0; t < numThreads; t++) {
            threads[t] = new Thread(() -> {
                ThreadLocalRandom rnd = ThreadLocalRandom.current();
                for (long i = 0; i < pointsPerThread; i++) {
                    double x = rnd.nextDouble();
                    double y = rnd.nextDouble();
                    if (x * x + y * y <= 1.0) {
                        totalHits++; // <-- THE BUG: read-modify-write is not atomic
                    }
                }
            });
        }

        long start = System.nanoTime();
        for (Thread th : threads) th.start();
        for (Thread th : threads) th.join();
        long end = System.nanoTime();

        double pi = 4.0 * totalHits / TOTAL_POINTS;
        System.out.printf("totalHits = %d, pi ~= %.6f, time = %.1f ms%n",
                totalHits, pi, (end - start) / 1e6);
    }
}
