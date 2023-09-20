# SYCL-Practice

by Jiajie Li (jl4257), Yifan Li (yl3722)

## 1. Accurate Timing Measurement: Profiling vs. Chrono

- The original codebase doesn’t place the timer in a very accurate way, e.g. the start timer of device 1 is placed before submitting the task2 to Queue2.
    - The code is like: Timer t1 → Submit Task 2 to Queue2 → Parameter setting for Task 1 → Submit Task 1 to Queue1 → Q1.wait() and measurement → Q2.wait() and measurement
- The start timer t1 should be placed after submitting task 2. Based on our experiments, it leads to much measurement difference. Therefore, our first step is to check the timing measurement separately for these two tasks. 

### Accurate timing measurement
- Seen in `edge_test_timing.cpp`
- To use chrono, we have: Timer t_start → Submit Task to Queue → Q.wait() → Timer t_end → Measurement

#### Task 1: Blur
| inImgHeight | inImgWidth | FilterWidth | Profiling | Chrono |
| --- | --- | --- | --- | --- |
| 256 | 512 | 11 | 3.29504e+06 | 1.40129e+08 |
| 512 | 512 | 11 | 6.04592e+06 | 1.43141e+08 |
| 512 | 512 | 22 | 2.39162e+07 | 1.60424e+08 |
| 512 | 512 | 44 | 9.4153e+07 | 2.31409e+08 |
| 2048 | 2048 | 11 | 9.43326e+07 | 2.31628e+08 |
| 2048 | 2048 | 44 | 1.48324e+09 | 1.6248e+09 |

- Profiling scales well on the kernel computation time (basically x4 with FilterWidth x2; x2 with inImgHeight x2)
- Chrono time includes the other overhead, including the kernel launch overhead.

#### Task 2: Compute Pi

| Profiling | Chrono |
| --- | --- |
| 1.29335e+08 | 2.83199e+08 |

### New Baseline
In the original setting for a 512x512 figure with FilterWidth=11, Blur kernel does not take more time than Compute_Pi. 

Therefore, we set a new baseline with a larger figure with 2048x2048, and FilterWidth=44. The following speedup is compared to this baseline. 

The 2048x2048 figure is `images-2048/goldfish_2048.png`, which is also a goldfish!

## 2. Split the workload of Task 1

We split the task of picture blurring into two seperate tasks, and run the tasks on both devices.

### Spliting Strategy

The pixels in different channels are stored together by "interleaving", which makes it difficult for task separation. We take the approach by cutting the picture vertically into two parts and assigning one part for each device.

This strategy is implemented by adjusting the `inImgHeight` property. We use two variables, namely `inImgHeight_a` and `inImgHeight_b` to indicate the height of picture each device is responsible for. In current implenmation, these two variables share the same value, which may seem redundant. However, using two instead of one variable will offer us the flexibility to split the task unevenly, which may be useful in future cases. Corresponding vars, such as `ndRange` and `globalRange` are adjusted correspondingly. 

### Code Structure

- Baseline
Device 1: Task 1; Device 2: Task 2

- Splitted
Device 1: Half of Task 1; Device 2: Task 2 + Half of Task 1

### Performance Measurement

We're using a larger figure with 2048x2048, and FilterWidth=44 for picture blurring, as stated above. 

We use chrono to count end to end time, shown as below. 

| Code    | Run Time     |
| ------- | ---------- |
| Baseline | 1.91719 seconds |
| Splited  | 1.20849 second |

We got 1.59x speedup, which meets our expectation using 2 GPUs. 

# 3. Scale to 4 GPUs
We try to scale to use 4 GPUs. But it turns out that after splitting the blur kernel to 1/4, it becomes kind of small as well (comparable to Compute PI kernel). 

Therefore, our final strategy is to unevenly split the Task 1 into three tasks. 

Queue 1: 4/9 of Blur; Queue 2: 3/9 of Blur; Queue 3: 2/9 of Blur; Queue 4: Compute PI. 

And we launch these kernels following the above order, so that the kernel launch overhead is compensated by the uneven split. 

The final result is: 
| Code    | Run Time     |
| ------- | ---------- |
| Baseline | 1.91719 second |
| 2GPU  | 1.20849 second |
| 4GPU  | 0.893097 second |

We got 2.15x speedup over the baseline. 