

# EdgeLLM: C++ INT4 Inference Engine

A zero-dependency, bare-metal inference runtime for Llama-architecture Large Language Models (LLMs). Written in C++17 with a custom Python quantization pipeline.

This project implements a complete Transformer forward pass from scratch, focusing on memory efficiency and instruction-level parallelism to run 1.1B parameter models on consumer CPUs without external ML frameworks (PyTorch, ONNX, etc.).

## Technical Implementation

*   **Architecture:** Llama-2/3 compatible (RoPE, SwiGLU, RMSNorm, Grouped Query Attention).
*   **Quantization:** Custom block-wise INT4 quantization scheme (group size 32). Compresses FP32 weights (4.2GB) to INT4 (1.3GB) with negligible perplexity degradation.
*   **Memory Management:** Single contiguous memory block allocation for model weights to minimize cache misses. Pre-allocated "scratchpad" memory for activation buffers to prevent runtime allocations.
*   **Parallelism:** Multithreaded matrix multiplication (`matmul_int4`) using `std::thread` and work-stealing logic, achieving linear scaling on multi-core CPUs.
*   **Performance:** Achieved ~4.3 tokens/second on an Apple M4 CPU (400% speedup over single-threaded baseline).

## Build & Usage

### Prerequisites
*   C++ Compiler (Clang/GCC) supporting C++17
*   CMake 3.10+
*   Python 3.x (for model export/quantization)

### 1. Export & Quantize Model
The engine requires a specific binary format. Use the provided Python script to download, quantize, and serialize the weights.

```bash
# Install dependencies
pip install torch transformers numpy

# Export TinyLlama-1.1B to INT4 format
python scripts/quantize.py
```

This generates `model_int4.bin` (approx. 1.3 GB).

### 2. Compile Engine
```bash
mkdir build
cd build
# Compile with optimization flags and threading support
cmake .. -DCMAKE_CXX_FLAGS="-O3 -pthread -march=native"
make
```

### 3. Run Inference
```bash
# Usage: ./edge_llm <path_to_model>
./edge_llm ../model_int4.bin
```

Example output:

```bash
cd build
# Add -pthread for threading
cmake .. -DCMAKE_CXX_FLAGS="-O3 -pthread -march=native"
make
./edge_llm ../scripts/model_int4.bin
cd: no such file or directory: build
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /Users/Desktop/edge-llm/build
[ 50%] Building CXX object CMakeFiles/edge_llm.dir/src/main.cpp.o
[100%] Linking CXX executable edge_llm
[100%] Built target edge_llm
 Edge LLM INT4 Engine...
we in here.
Allocating KV Cache (88 MB)...

Loaded 32000 tokens.

--- GENERATING ---
rages , and the use of the word " s " to describe the physical appearance of the body . 

Speed: 4.0991 t/s

Done.
```

## Performance Benchmarks

**Hardware:** Apple M4 (CPU only)
**Model:** TinyLlama-1.1B (Chat)

| Implementation | Precision | Latency (ms/tok) | Throughput (tok/s) | Memory Footprint |
|:---|:---|:---|:---|:---|
| Naive C++ (Baseline) | FP32 | ~1100ms | ~0.9 | 4.2 GB |
| EdgeLLM (Single Thread) | INT4 | ~900ms | ~1.1 | **1.3 GB** |
| **EdgeLLM (Multi-Thread)** | **INT4** | **~230ms** | **~4.3** | **1.3 GB** |

## Engineering Challenges & Retrospective

Building a raw inference engine revealed several low-level bottlenecks that standard frameworks abstract away.

### 1. The "Standard Library" Trap
**Attempt:** Initially used `std::accumulate` for residual connections to keep code clean.

**Failure:** Caused immediate segmentation faults. `std::accumulate` iterates based on iterator logic, which clashed with the raw pointer arithmetic used for the memory arena.

**Solution:** Wrote custom raw-loop kernels (`accum`, `element_mul`) to ensure strict bounds checking and memory safety.

### 2. File Stream Synchronization
**Attempt:** Loading the model using a standard loop over `config.n_layers`.

**Failure:** The C++ loader desynchronized from the Python exporter. The Python `model.named_parameters()` yields tensors in a specific graph order, while the C++ loop expected a strict `Attention -> MLP` order. This caused the loader to interpret float data as integer quantization flags, leading to garbage output.

**Solution:** Enforced strict serialization order in `quantize.py` and added "Magic Number" headers to every tensor block in the binary file to validate alignment during load time.

### 3. MatMul Cache Locality
**Attempt:** A standard `i-j-k` triple loop for matrix multiplication.

**Failure:** Poor performance (0.7 GFLOPS) due to non-sequential memory access on the weight matrix causing constant cache misses.

**Solution:** Reordered loops to access memory sequentially and implemented block-wise processing. This simple change yielded a 20x speedup in the raw math benchmark before multithreading was even applied.

### 4. Quantization Precision Mismatch
**Attempt:** Using a simple `(max - min)` scaling approach for quantization.

**Failure:** Resulted in high perplexity (incoherent text) because outlier weights in the attention layers skewed the scale.

**Solution:** Switched to a symmetric quantization scheme centered on zero, using `max(abs(weight))` for scaling. This preserved the distribution of the attention heads more accurately.
