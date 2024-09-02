#include <iostream>
#include <set>
#include <random>
#include <assert.h>

#include <chrono>

#include <fstream>
#include <string>
#include <vector>

#include "zipf.h"
#include "latest-generator.h"
#include "skiplist.h"

double calculatePercentile(const std::vector<double>& data, double percentile) {
    std::vector<double> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end());

    double index = (percentile / 100.0) * (sortedData.size() - 1);
    int lowerIndex = static_cast<int>(index);
    int upperIndex = std::min(lowerIndex + 1, static_cast<int>(sortedData.size() - 1));

    double fractionalPart = index - lowerIndex;

    return sortedData[lowerIndex] + fractionalPart * (sortedData[upperIndex] - sortedData[lowerIndex]);
}

void fb(SkipList<Key>& sl) {
    // Load keys from the 'fb' dataset
    std::string file_path = "./dataset/fb.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::vector<Key> numbers;
    std::string line;
    while (std::getline(file, line)) {
        Key number = std::stoull(line);
        numbers.push_back(number);
    }
    file.close();

    // Random generator for selecting keys
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, numbers.size() - 1);

    // Insert keys from dataset
    auto w_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Search keys from dataset
    auto r_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate and display times
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;
    printf("\n[Real-fb] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void books(SkipList<Key>& sl) {
    // Load keys from the 'books' dataset
    std::string file_path = "./dataset/books.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::vector<Key> numbers;
    std::string line;
    while (std::getline(file, line)) {
        Key number = std::stoull(line);
        numbers.push_back(number);
    }
    file.close();

    // Random generator for selecting keys
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, numbers.size() - 1);

    // Insert keys from dataset
    auto w_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Search keys from dataset
    auto r_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate and display times
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;
    printf("\n[Real-books] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void wiki(SkipList<Key>& sl) {
    // Load keys from the 'wiki' dataset
    std::string file_path = "./dataset/wiki.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::vector<Key> numbers;
    std::string line;
    while (std::getline(file, line)) {
        Key number = std::stoull(line);
        numbers.push_back(number);
    }
    file.close();

    // Random generator for selecting keys
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, numbers.size() - 1);

    // Insert keys from dataset
    auto w_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Search keys from dataset
    auto r_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate and display times
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;
    printf("\n[Real-wiki] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void osm(SkipList<Key>& sl) {
    // Load keys from the 'osm' dataset
    std::string file_path = "./dataset/osm.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    std::vector<Key> numbers;
    std::string line;
    while (std::getline(file, line)) {
        Key number = std::stoull(line);
        numbers.push_back(number);
    }
    file.close();

    // Random generator for selecting keys
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, numbers.size() - 1);

    // Insert keys from dataset
    auto w_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Search keys from dataset
    auto r_start = Clock::now();
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        Key key = numbers[distr(gen)];
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate and display times
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;
    printf("\n[Real-osm] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadA(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload A: Read-heavy (50% reads, 50% writes)
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = (nextValue() % (write)) + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys
    auto r_start = Clock::now();
    for (int i = 0; i < read; i++) {
        Key Zkey = (nextValue() % (read)) + 1;

        int next_op = rand() % 100;

        if (next_op < 50) {
            sl.Contains(Zkey);
        } else {
            sl.Insert_usplit(Zkey);
        }
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-A] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadB(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload B: Read-mostly (95% reads, 5% writes)
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = (nextValue() % (write)) + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key Zkey = (nextValue() % (read)) + 1;

        int next_op = rand() % 100;

        if (next_op < 95) {
            sl.Contains(Zkey);
        } else {
            sl.Insert_usplit(Zkey);
        }
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-B] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadC(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload C: Read-only (100% reads)
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = (nextValue() % (write)) + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key Zkey = (nextValue() % (read)) + 1;
        sl.Contains(Zkey);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-C] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadD(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload D: Read-latest (95% reads, 5% inserts)
    init_latestgen(write);
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = next_value_latestgen() % write + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key Zkey = next_value_latestgen() % read + 1;

        int next_op = rand() % 100;

        if (next_op < 95) {
            sl.Contains(Zkey);
        } else {
            sl.Insert_usplit(Zkey);
        }
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-D] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadE(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload E: Short ranges (95% scans, 5% inserts)
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = (nextValue() % (write)) + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Scan keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key Zkey = (nextValue() % (read)) + 1;

        int next_op = rand() % 100;

        if (next_op < 95) {
            sl.Scan(Zkey, 100);
        } else {
            sl.Insert_usplit(Zkey);
        }
    }
    auto r_end = Clock::now();

    // Calculate scan time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-E] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void YCSBWorkloadF(const int write, const int read, SkipList<Key>& sl) {
    // YCSB Workload F: Read-modify-write (50% reads, 50% writes)
    init_zipf_generator(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = (nextValue() % (write)) + 1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Read-modify-write for keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key Zkey = (nextValue() % (read)) + 1;

        int next_op = rand() % 100;

        if (next_op < 50) {
            sl.Contains(Zkey);
        } else {
            sl.Contains(Zkey);
            sl.Insert_usplit(Zkey);
        }
    }
    auto r_end = Clock::now();

    // Calculate operation time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[YCSB-F] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Zipfian(const int write, const int read, SkipList<Key>& sl) {
    // Zipfian distribution generator
    init_zipf_generator(0, write);

    // Insert keys following Zipfian distribution
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = nextValue() % write+1;        
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys following Zipfian distribution
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = nextValue() % read+1;
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[Zipfian] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Uniform(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(1, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        sl.Insert_usplit(distr(gen)+1);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        sl.Contains(distr(gen)+1);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[Uniform] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void RevSequential(const int write, const int read, SkipList<Key>& sl) {
    // Insert keys reverse sequentially
    auto w_start = Clock::now();
    for (int i = write; i > 0; i--) {
        sl.Insert_usplit(i);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys reverse sequentially
    auto r_start = Clock::now();
    for (int i = read; i > 0; i--) {
        sl.Contains(i);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[Rev-Sequential] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Sequential(const int write, const int read, SkipList<Key>& sl) {
    // Insert keys sequentially
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        sl.Insert_usplit(i);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys sequentially
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        sl.Contains(i);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[Sequential] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Zipfian_latency(const int write, const int read, SkipList<Key>& sl) {
    // Vectors to store write and read latencies
    std::vector<double> w_lat;
    std::vector<double> r_lat;

    // Initialize Zipfian generator
    init_zipf_generator(0, write);

    // Measure write latencies
    for (int i = 1; i <= write; ++i) {
        Key key = nextValue() % write + 1;
        auto w_start = Clock::now();
        sl.Insert_usplit(key);
        auto w_end = Clock::now();
        double w_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_end - w_start).count();
        w_lat.push_back(w_time);
    }
    std::cout << "After Insert\n";

    // Measure read latencies
    for (int i = 1; i <= read; ++i) {
        Key key = nextValue() % read + 1;
        auto r_start = Clock::now();
        sl.Contains(key);
        auto r_end = Clock::now();
        double r_time = std::chrono::duration_cast<std::chrono::nanoseconds>(r_end - r_start).count();
        r_lat.push_back(r_time);
    }

    // Calculate and print write latencies percentiles
    std::vector<double> write_percentiles = {50, 99, 99.9, 99.99, 99.999};
    std::cout << "\n[Zipfian Latency] Insertion: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf", p, calculatePercentile(w_lat, p));
    }

    // Calculate and print read latencies percentiles
    std::cout << "\n[Zipfian Latency] Lookup: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf", p, calculatePercentile(r_lat, p));
    }
    std::cout << std::endl;
}

void Uniform_latency(const int write, const int read, SkipList<Key>& sl) {
    // Vectors to store write and read latencies
    std::vector<double> w_lat;
    std::vector<double> r_lat;

    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Measure write latencies
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen) + 1;
        auto w_start = Clock::now();
        sl.Insert_usplit(key);
        auto w_end = Clock::now();
        double w_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_end - w_start).count();
        w_lat.push_back(w_time);
    }
    std::cout << "After Insert\n";

    // Measure read latencies
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen) + 1;
        auto r_start = Clock::now();
        sl.Contains(key);
        auto r_end = Clock::now();
        double r_time = std::chrono::duration_cast<std::chrono::nanoseconds>(r_end - r_start).count();
        r_lat.push_back(r_time);
    }

    // Calculate and print write latencies percentiles
    std::vector<double> write_percentiles = {50, 99, 99.9, 99.99, 99.999};
    std::cout << "\n[Uniform Latency] Insertion: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf", p, calculatePercentile(w_lat, p));
    }

    // Calculate and print read latencies percentiles
    std::cout << "\n[Uniform Latency] Lookup: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf", p, calculatePercentile(r_lat, p));
    }
    std::cout << std::endl;
}

void Sequential_latency(const int write, const int read, SkipList<Key>& sl) {
    // Vectors to store write and read latencies
    std::vector<double> w_lat;
    std::vector<double> r_lat;

    // Measure write latencies
    for (int i = 1; i <= write; ++i) {
        auto w_start = Clock::now();
        sl.Insert_usplit(i);
        auto w_end = Clock::now();
        double w_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_end - w_start).count();
        w_lat.push_back(w_time);
    }
    std::cout << "After Insert\n";

    // Measure read latencies
    for (int i = 1; i <= read; ++i) {
        auto r_start = Clock::now();
        sl.Contains(i);
        auto r_end = Clock::now();
        double r_time = std::chrono::duration_cast<std::chrono::nanoseconds>(r_end - r_start).count();
        r_lat.push_back(r_time);
    }

    // Calculate and print write latencies percentiles
    std::vector<double> write_percentiles = {50, 99, 99.9, 99.99, 99.999};
    std::cout << "\n[Sequential Latency] Insertion: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf ns", p, calculatePercentile(w_lat, p));
    }

    // Calculate and print read latencies percentiles
    std::cout << "\n[Sequential Latency] Lookup: ";
    for (double p : write_percentiles) {
        printf(" %.2lf = %.lf ns", p, calculatePercentile(r_lat, p));
    }
    std::cout << std::endl;
}

void Uniform_Scan(const int write, const int read, SkipList<Key> &sl) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    auto w_start = Clock::now();
    for(int i = 1; i <= write; i++) {
        Key key = distr(gen)+1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    printf("After Insert\n");
    auto r_start = Clock::now();
    for(int i = 1; i <= read; i++) {
        Key key = distr(gen)+1;
        sl.Scan(key, 100);
    }
    auto r_end = Clock::now();

    float r_time, w_time;
    r_time = std::chrono::duration_cast<std::chrono::nanoseconds>(r_end - r_start).count() * 0.001;
    w_time = std::chrono::duration_cast<std::chrono::nanoseconds>(w_end - w_start).count() * 0.001;
    printf("\n[Uniform-Scan] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Array(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen)+1;
        sl.Insert_Array(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen)+1;
        sl.Contains_Raise(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[+Array] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Raise(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen)+1;
        sl.Insert_Raise(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen)+1;
        sl.Contains_Raise(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[+Raise] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Search(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen)+1;
        sl.Insert_Search(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen)+1;
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[+Search] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void Split(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen)+1;
        sl.Insert_usplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen)+1;
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[+Split] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void EvenSplitSequential(const int write, const int read, SkipList<Key>& sl) {
    // Insert keys sequentially
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        sl.Insert_esplit(i);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys sequentially
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        sl.Contains(i);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[EvenSplit Sequential] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void EvenSplitRevSequential(const int write, const int read, SkipList<Key>& sl) {
    // Insert keys reverse sequentially
    auto w_start = Clock::now();
    for (int i = write; i > 0; i--) {
        sl.Insert_esplit(i);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys reverse sequentially
    auto r_start = Clock::now();
    for (int i = read; i > 0; i--) {
        sl.Contains(i);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[EvenSplit - RevSequential] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void EvenSplitUniform(const int write, const int read, SkipList<Key>& sl) {
    // Uniformly distributed random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, write);

    // Insert random keys
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = distr(gen)+1;
        sl.Insert_esplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search random keys
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = distr(gen)+1;
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[EvenSplit Uniform] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void EvenSplitZipfian(const int write, const int read, SkipList<Key>& sl) {
    // Zipfian distribution generator
    init_zipf_generator(0, write);

    // Insert keys following Zipfian distribution
    auto w_start = Clock::now();
    for (int i = 1; i <= write; ++i) {
        Key key = nextValue() % write+1;        
        sl.Insert_esplit(key);
    }
    auto w_end = Clock::now();
    std::cout << "After Insert\n";

    // Calculate insertion time
    float w_time = std::chrono::duration_cast<std::chrono::microseconds>(w_end - w_start).count() * 0.001;

    // Search for keys following Zipfian distribution
    auto r_start = Clock::now();
    for (int i = 1; i <= read; ++i) {
        Key key = nextValue() % read+1;
        sl.Contains(key);
    }
    auto r_end = Clock::now();

    // Calculate search time
    float r_time = std::chrono::duration_cast<std::chrono::microseconds>(r_end - r_start).count() * 0.001;

    // Display results
    printf("\n[EvenSplit Zipfian] Insertion = %.2lf µs, Lookup = %.2lf µs\n", w_time, r_time);
}

void printUsage(const char* programName) {
    std::cerr << "\nUsage: " << programName << " [Write Count] [Read Count] [Benchmark]\n\n"
              << "Benchmark can be selected by number or name.\n\n"
              << "Synthetic Benchmarks:\n"
              << " 0 - Sequential\n"
              << " 1 - Rev-Sequential\n"
              << " 2 - Uniform\n"
              << " 3 - Zipfian\n"
              << " 4~9 - YCSB(A~F)\n\n"
              << "Real-World Benchmarks: (Fixed Dataset Size: 200M)\n"
              << " 10 - fb\n"
              << " 11 - books\n"
              << " 12 - wiki\n"
              << " 13 - osm\n\n"
              << "Latency Benchmarks:\n"
              << " 14 - Sequential_latency\n"
              << " 15 - Uniform_latency\n"
              << " 16 - Zipfian_latency\n\n"
              << "Scan Benchmarks:\n"
              << " 17 - Scan\n\n"
              << "Breakdown Benchmarks (Uniform Only):\n"
              << " 18 - +Array\n"
              << " 19 - +Raise\n"
              << " 20 - +Search\n"
              << " 21 - +Split\n\n"
              << "Even Split Benchmarks:\n"
              << " 22 - Sequential\n"
              << " 23 - Rev-Sequential\n"
              << " 24 - Uniform\n"
              << " 25 - Zipfian\n";
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printUsage(argv[0]);
        return 1;
    }

    const int W = std::atoi(argv[1]);  // Insertion count
    const int R = std::atoi(argv[2]);  // Lookup count
    const int B = std::atoi(argv[3]);  // Benchmark type

    SkipList<Key> sl;

    auto runBenchmarkType1 = [&](const std::string& name, void (*benchmarkFunc)(int, int, SkipList<Key>&)) {
        std::cout << "\n[" << name << " Benchmark in progress...]\n\n";
        benchmarkFunc(W, R, sl);
    };

    auto runBenchmarkType2 = [&](const std::string& name, void (*benchmarkFunc)(SkipList<Key>&)) {
        std::cout << "\n[" << name << " Benchmark in progress...]\n\n";
        benchmarkFunc(sl);
    };

    switch (B) {
        // Type 1:
        case 0: runBenchmarkType1("Sequential", Sequential); break;
        case 1: runBenchmarkType1("Rev-Sequential", RevSequential); break;
        case 2: runBenchmarkType1("Uniform", Uniform); break;
        case 3: runBenchmarkType1("Zipfian", Zipfian); break;
        case 4: runBenchmarkType1("YCSB-A", YCSBWorkloadA); break;
        case 5: runBenchmarkType1("YCSB-B", YCSBWorkloadB); break;
        case 6: runBenchmarkType1("YCSB-C", YCSBWorkloadC); break;
        case 7: runBenchmarkType1("YCSB-D", YCSBWorkloadD); break;
        case 8: runBenchmarkType1("YCSB-E", YCSBWorkloadE); break;
        case 9: runBenchmarkType1("YCSB-F", YCSBWorkloadF); break;
        case 14: runBenchmarkType1("Sequential Latency", Sequential_latency); break;
        case 15: runBenchmarkType1("Uniform Latency", Uniform_latency); break;
        case 16: runBenchmarkType1("Zipfian Latency", Zipfian_latency); break;
        case 17: runBenchmarkType1("Uniform Scan", Uniform_Scan); break;
        case 18: runBenchmarkType1("+Array", Array); break;
        case 19: runBenchmarkType1("+Raise", Raise); break;
        case 20: runBenchmarkType1("+Search", Search); break;
        case 21: runBenchmarkType1("+Split", Split); break;
        case 22: runBenchmarkType1("EvenSplit-Sequential", EvenSplitSequential); break;
        case 23: runBenchmarkType1("EvenSplit-RevSequential", EvenSplitRevSequential); break;
        case 24: runBenchmarkType1("EvenSplit-Uniform", EvenSplitUniform); break;
        case 25: runBenchmarkType1("EvenSplit-Zipfian", EvenSplitZipfian); break;
        
        // Type 2:
        case 10: runBenchmarkType2("Real-World Dataset (fb)", fb); break;
        case 11: runBenchmarkType2("Real-World Dataset (books)", books); break;
        case 12: runBenchmarkType2("Real-World Dataset (wiki)", wiki); break;
        case 13: runBenchmarkType2("Real-World Dataset (osm)", osm); break;

        default:
            std::cerr << "Invalid benchmark option provided.\n";
            printUsage(argv[0]);
            return 1;
    }

    return 0;
}