# SYCL-Practice

by Jiajie Li, Yifan Li

# 1. Accurate Timing Measurement: Profiling vs. Chrono

- The original codebase doesn’t put the timer in a very accurate way, e.g. the start timer of device 1 is put before submitting the task2 to Queue2.
    - The code is like: Timer t1 → Submit Task 2 to Queue2 → Parameter setting for Task 1 → Submit Task 1 to Queue1 → Q1.wait() and measurement → Q2.wait() and measurement
- The start timer t1 should be placed after submitting task 2. In our experiments, it leads to much measurement difference. Therefore, our first step is to check the timing measurement separately for these two tasks

## Accurate timing measurement
- Seen in `edge_test_timing.cpp`
- To use chrono, we have: Timer t_start → Submit Task to Queue → Q.wait() → Timer t_end → Measurement

### Task 1: Blur
| inImgHeight | inImgWidth | FilterWidth | Profiling | Chrono |
| --- | --- | --- | --- | --- |
| 256 | 512 | 11 | 3.29504e+06 | 1.40129e+08 |
| 512 | 512 | 11 | 6.04592e+06 | 1.43141e+08 |
| 512 | 512 | 22 | 2.39162e+07 | 1.60424e+08 |
| 512 | 512 | 44 | 9.4153e+07 | 2.31409e+08 |
| 512 | 512 | 88 | 3.85957e+08 | 5.23417e+08 |
| **2048** | **2048** | **11** | **9.43326e+07** | **2.31628e+08** |
| 2048 | 2048 | 44 | 1.48324e+09 | 1.6248e+09 |
| 8192 | 8192 | 11 | 1.51579e+09 | 1.68816e+09 |

- Considering the algorithm’s complexity is (inImgHeight * inImgWidth * FilterWidth^2), it seems *Profiling* scales well on the kernel computation time. (It's basically x4 with FilterWidth x2; x2 with inImgHeight x2)
- Chrono time includes the other overhead, including the kernel launch overhead.

### Task 2: Compute Pi

| Profiling | Chrono |
| --- | --- |
| 1.29335e+08 | 2.83199e+08 |

## New Baseline
In the original setting for a 512x512 figure with FilterWidth=11, Blur kernel does not take more time than Compute_Pi. 

Therefore, we set a new baseline with a larger figure with 2048x2048, and FilterWidth=44. The following speedup is compared to this baseline. 

The 2048x2048 figure is `images-2048/goldfish_2048.png`, which is also a goldfish!

# 2. Split the workload of Task 1

