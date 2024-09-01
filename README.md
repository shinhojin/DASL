# DASL: Deterministic Arrayed Skip List

This repository contains the implementation of DASL (Deterministic Arrayed Skip List), a novel in-memory index that enhances tail latency, microarchitecture friendliness, and reduces restructuring overhead, as proposed in the research paper:

**DASL: An Index for Enhancing Tail Latency, Microarchitecture Friendliness, and Restructuring Overhead**  
by Hojin Shin, Bryan S. Kim, Seehwan Yoo, and Jongmoo Choi. (Submitted)

## Repository Structure

- `src/`: Contains the source code files.
  - `skiplist_test.cc`: Main test file for evaluating the DASL implementation.
  - `zipf.cc` and `latest-generator.cc`: Utilities for generating synthetic workloads (e.g., Zipfian).
- `src/`: Contains the header files.
  - `skiplist.h`: Header file defining the DASL structure and functions.
  - `zipf.h` and `latest-generator.h`: Header files for workload generation utilities.
- `Makefile`: The Makefile for compiling the code.
- `README.md`: This file.

## Usage

### Prerequisites

To compile and run the code, you need:
- `Compiler` : A C++ compiler supporting C++11 or later (e.g., `g++`).
- `make` : build system.
- `System` : Ubuntu 20.04.6 LTS
- `Hardware` : Intel. Support of AVX512 is a must.

### Building the Project

```bash
make
```

### Running the Benchmark

To run the benchmark, use the following command:

```bash
./sl_test [Write Count] [Read Count] [Benchmark]
```

- `Write Count` : Number of insertion operations to perform
- `Read Count` : Number of lookup operations to perform
- `Benchmark` : Select the benchmark type. You can use either the number of name to select it.
  - `Synthetic Benchmarks`: 0 - Sequential, 1 - Reverse Sequential 2 - Uniform, 3 - Zipfian
  - `YCSB Benchmarks`: 4~9 - YCSB(A~F)
  - `Real-World Benchmarks`: 10 - fb, 11 - books, 12 - wiki, 13 - osm
  - `Latency Benchmarks`: 14 - Sequential, 15 - Uniform, 16 - Zipfian
  - `Scan Benchmarks`: 17 - Scan
  - `Breakdown Benchmarks (Uniform Only)`: 18 - +Array, 19 - +Raise, 20 - +Search, 21 - +Split
  - `Even Split Benchmarks `: 22 - Sequential, 23 - Reverse Sequential, 24 - Uniform, 25 - Zipfian

### Example Command

```bash
./sl_test 1000 1000 0 # Run the Sequential synthetic benchmark with 1000 insertions and 1000 lookups
```

### Notes

- `Real-World Datasets`: The dataset is too large to upload. For more information, see ``/dataset/LOAD_DATASET.md``

Please download the necessary real-world datasets from the provided GitHub link above. These datasets are required for running benchmarks 9 through 12.