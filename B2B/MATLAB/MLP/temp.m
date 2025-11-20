% ========================================================================
% gpu_benchmark.m
%
% Purpose:
%   - Check availability and basic specs of GPU device.
%   - Compare CPU vs GPU performance on:
%       (1) Large matrix multiplication
%       (2) 1D FFT on a large vector
%   - Report elapsed time and approximate throughput.
%
% Requirements:
%   - Parallel Computing Toolbox
%   - A supported GPU device (checked automatically)
% ========================================================================

clear; clc;
parallel.gpu.enableCUDAForwardCompatibility(true)
fprintf('================ GPU Benchmark =================\n');

% ------------------------- Device Info -------------------------------
if gpuDeviceCount("available") == 0
    fprintf('No available GPU detected by MATLAB.\n');
    fprintf('Check: "gpuDeviceCount" and your driver / CUDA setup.\n');
    return;
end

% Select default GPU
gdev = gpuDevice(1);

fprintf('\n[GPU Device Information]\n');
fprintf('Name               : %s\n', gdev.Name);
fprintf('Compute Capability : %s\n', gdev.ComputeCapability);
fprintf('Total Memory (GB)  : %.2f\n', gdev.TotalMemory / 1024^3);
fprintf('Multiprocessors    : %d\n', gdev.MultiprocessorCount);
fprintf('Supports FP64      : %d\n', gdev.SupportsDouble);
fprintf('Toolkit Version    : %s\n', gdev.ToolkitVersion);
fprintf('Driver Version     : %s\n', gdev.DriverVersion);

reset(gdev);   % Clear any previous GPU state

% ------------------------- Benchmark Config --------------------------
% Matrix size for GEMM test: C = A * B
% Complexity: ~ 2 * N^3 floating-point operations
N = 4096*1.5;          % Increase for heavier test if GPU has enough memory
dtype = 'single';  % 'single' is often faster on gaming GPUs

% FFT size
Nfft = 2^22;       % Length of 1D vector for FFT benchmark

fprintf('\n[Benchmark Configuration]\n');
fprintf('Matrix size (GEMM)       : %d x %d\n', N, N);
fprintf('Data type                : %s\n', dtype);
fprintf('FFT length (1D)          : %d\n', Nfft);

% ------------------------- Helper Functions --------------------------
% Matrix multiplication test (CPU)
matmul_cpu = @() matmul_test_cpu(N, dtype);

% Matrix multiplication test (GPU)
matmul_gpu = @() matmul_test_gpu(N, dtype);

% FFT test (CPU)
fft_cpu = @() fft_test_cpu(Nfft, dtype);

% FFT test (GPU)
fft_gpu = @() fft_test_gpu(Nfft, dtype);

% --------------------- Warm-up (avoid JIT bias) ----------------------
fprintf('\n[Warm-up]\n');
matmul_cpu();
matmul_gpu();
fft_cpu();
fft_gpu();
wait(gdev);  % Ensure all GPU work is finished

% ------------------------- Timing (GEMM) -----------------------------
fprintf('\n[Matrix Multiplication Benchmark]\n');

% CPU timing
t_cpu = timeit(matmul_cpu);
flops_gemm = 2 * N^3;  % Theoretical FLOPs for GEMM
gflops_cpu = flops_gemm / t_cpu / 1e9;

fprintf('CPU  : %.3f s, approx %.2f GFLOP/s\n', t_cpu, gflops_cpu);

% GPU timing (use gputimeit to measure kernel time only)
t_gpu = gputimeit(matmul_gpu);
gflops_gpu = flops_gemm / t_gpu / 1e9;

fprintf('GPU  : %.3f s, approx %.2f GFLOP/s\n', t_gpu, gflops_gpu);

% --------------------------- Timing (FFT) ----------------------------
fprintf('\n[1D FFT Benchmark]\n');

% Rough operation count for complex FFT ~ 5 * N * log2(N)
flops_fft = 5 * Nfft * log2(Nfft);

% CPU
t_fft_cpu = timeit(fft_cpu);
gflops_fft_cpu = flops_fft / t_fft_cpu / 1e9;
fprintf('CPU  : %.3f s, approx %.2f GFLOP/s\n', t_fft_cpu, gflops_fft_cpu);

% GPU
t_fft_gpu = gputimeit(fft_gpu);
gflops_fft_gpu = flops_fft / t_fft_gpu / 1e9;
fprintf('GPU  : %.3f s, approx %.2f GFLOP/s\n', t_fft_gpu, gflops_fft_gpu);

% -------------------------- Summary ----------------------------------
fprintf('\n[Summary]\n');
fprintf('GEMM speedup (GPU / CPU) : %.2fx\n', t_cpu / t_gpu);
fprintf('FFT  speedup (GPU / CPU) : %.2fx\n', t_fft_cpu / t_fft_gpu);

fprintf('\nBenchmark finished.\n');
fprintf('================================================\n');

% =====================================================================
% Local functions
% =====================================================================

function C = matmul_test_cpu(N, dtype)
    % CPU matrix multiplication: C = A * B
    A = rand(N, N, dtype);
    B = rand(N, N, dtype);
    C = A * B;
end

function Cg = matmul_test_gpu(N, dtype)
    % GPU matrix multiplication: C = A * B on GPU
    Ag = gpuArray.rand(N, N, dtype);
    Bg = gpuArray.rand(N, N, dtype);
    Cg = Ag * Bg;
end

function y = fft_test_cpu(Nfft, dtype)
    % CPU 1D FFT on random complex vector
    x = complex(rand(Nfft, 1, dtype), rand(Nfft, 1, dtype));
    y = fft(x);
end

function yg = fft_test_gpu(Nfft, dtype)
    % GPU 1D FFT on random complex vector
    xg = complex(gpuArray.rand(Nfft, 1, dtype), gpuArray.rand(Nfft, 1, dtype));
    yg = fft(xg);
end
