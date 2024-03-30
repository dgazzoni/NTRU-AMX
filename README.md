This is the source code accompanying the paper "Fast polynomial multiplication using matrix multiplication accelerators with applications to NTRU on the Apple M1 SoC". This is a list of folders and their contents:

- `amx`: [AMX macros from Peter Cawley](https://github.com/corsix/amx) (`aarch64.h`), plus helper macros of our own (`amx.h`) and implementations of routines described in our paper;
- `ct_results_M1_dit`: results and histogram plots for constant-time experiments in the M1, with the DIT (data-independent timing) bit set;
- `ct_results_M3_dit`: results and histogram plots for constant-time experiments in the M3, also with the DIT bit set;
- `googletest`: a copy of the [Google Test](https://github.com/google/googletest/) library;
- `jupyter`: Jupyter notebooks to generate some calculations and plots of the "Experimental results" section;
- `PQC_NEON`: relevant files from the paper ["Optimized Software Implementations of CRYSTALS-Kyber, NTRU, and Saber Using NEON-Based Special Instructions of ARMv8"](https://csrc.nist.gov/CSRC/media/Events/third-pqc-standardization-conference/documents/accepted-papers/nguyen-optimized-software-gmu-pqc2021.pdf), obtained from the associated [GitHub repository](https://github.com/GMUCERG/PQC_NEON), with some modifications as described in our paper;
- `reference`: relevant files from [NTRU](https://ntru.org)'s reference implementation, based on the [Round 3 submission package](https://ntru.org/release/NIST-PQ-Submission-NTRU-20201016.tar.gz) to the NIST PQC standardization effort;
- `rng_opt`: an optimized implementation of the NIST `randombytes` routines, based on AES-256 CTR-DRBG, implemented using AES instructions available in the ARMv8-A Cryptographic Extensions;
- `speed`: benchmarking harnesses for our implementation, constant-time experiments, our custom RNG, and some BLAS routines found in Apple's Accelerate framework.
- `speed_results_M1_dit`: raw benchmark results in the M1, with the DIT bit set, and an Excel spreadsheet compiling them, generated using a Python script discussed [below](#Helper-scripts-for-benchmarking-and-constant-time-experiments);
- `speed_results_M3_dit`: raw benchmark results in the M3, with the DIT bit set,and an Excel spreadsheet compiling them, generated using a Python script discussed [below](#Helper-scripts-for-benchmarking-and-constant-time-experiments);
- `test`: tests (using the [Google Test](https://github.com/google/googletest/) library) to validate various aspects of the implementation as well as our optimized RNG;
- `vector-polymul-ntru-ntrup`: relevant files from the paper ["Algorithmic Views of Vectorized Polynomial Multipliers -- NTRU"](https://eprint.iacr.org/2023/1637), obtained from the associated [GitHub repository](https://github.com/vector-polymul-ntru-ntrup/NTRU), with some modifications as described in our paper.

The root folder also includes some files of note:

- `run_benchmarks_dit.sh` and `run_benchmarks_no_dit.sh`: a helper script to run benchmarks (see instructions [below](#Helper-scripts-for-benchmarking-and-constant-time-experiments));
- `run_ct_experiments_dit.sh` and `run_ct_experiments_no_dit.sh`: a helper script to run constant-time experiments (see instructions [below](#Helper-scripts-for-benchmarking-and-constant-time-experiments));
- `run_all.sh`: a helper script to run both benchmarks and constant-time experiments.
- `consolidate_benchmarks.py`: a Python 3 script to consolidate benchmark results from different systems into a single Microsoft Excel file.

# Building the code

We use [CMake](https://cmake.org) as our build system. It can be installed using [Homebrew](https://brew.sh) with the command `brew install cmake`. A typical sequence of commands to build the code, starting from the root folder of the repository, is:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

**NOTE**: for the tested compilers, there is a register allocation issue when the optimized `randombytes` routine is compiled in Debug mode (i.e. passing `-DCMAKE_BUILD_TYPE=Debug` to CMake), and the build fails. However, in RelWithDebInfo and Release mode, there is no issue.

# Running tests

Compilation produces many test binaries in the build folder (`build/test_*` if using the directions in [Building the code](#building-the-code) above). While it is possible to run each binary directly, we recommend using the `ctest` utility from CMake to run all available tests with a single invocation. `ctest` also runs additional tests that automate the process of comparing KATs using the `PQCgenKAT_kem_*` binaries.

# Running benchmarks

Compilation produces many benchmarking binaries in the build folder (`build/speed_*` if using the directions in [Building the code](#building-the-code) above). Each binary may be run directly, or a full benchmark set can be automatically run using the helper scripts described in [Benchmarking helper scripts](#benchmarking-helper-scripts) below.

Note that binaries must be run with `sudo` to allow access the cycle counter.

# Helper scripts for benchmarking and constant-time experiments

We provide helper scripts to automatically run all available benchmarks (except for those related to the RNG), in the form of `run_benchmarks_dit.sh` and `run_benchmarks_no_dit.sh`, with the DIT (data-independent timing) bit set or unset, respectively. For the paper's results, only the DIT version was used, due to the [GoFetch microarchitectural attack](https://gofetch.fail). It must be run from the root folder of the repository, and places their results in a folder called `speed_results_Mx_dit`, where `Mx` will be replaced by the CPU name in the machine where the script is run, e.g. `M1`, `M2` or `M3`; this is obtained from `sysctl -n machdep.cpu.brand_string`. Each executable file that is run creates an associated text file containing the benchmark results, with a self-explanatory naming scheme.

A Python 3 script, `consolidate_benchmarks.py`, can be run afterwards (also from the root folder of the repository). It requires the [`pandas`](https://pandas.pydata.org) and [`xlsxwriter`](https://pypi.org/project/XlsxWriter/) packages, which can be installed using `pip`.

This script reads all results from the `speed_results_Mx_dit` folder and generates a Microsoft Excel file displaying them in a tabular form, similar to the format of the results presented in our paper.

Similarly, we provide helper scripts to run constant-time experiments, `run_ct_experiments_dit.sh` and `run_ct_experiments_no_dit.sh`. Similar considerations as above, regarding the DIT bit, apply. It must be run from the root folder of the repository, and places their results in a folder called `ct_results_Mx_dit` (see earlier discussion regarding CPU name). Text files are created with the same naming scheme as in the benchmark script. A Jupyter notebook found in `jupyter/histogram_heat_maps.ipynb` is supplied to generate histogram plots, and an R script found in `jupyter/equivalence_tests.R` performs statistical hypothesis tests for equivalence.

# License

Our work builds upon many other libraries and implementations, with different licenses for each. Any modifications that we make to an existing work is released under the same original license as that work. As for our original code, we release it under the [Creative Commons CC0 1.0 Universal (CC0 1.0)
Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
