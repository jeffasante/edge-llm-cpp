#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include "layers.h"

// Global buffer to hold the weights
// std::vector<float> weights;

struct LLMConfig
{
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len = 2048;
};

// hold quantized weights structs
struct QuantizedTensor
{
    std::vector<float> scales;   // 1 scale per 32 weights
    std::vector<uint8_t> q_data; // Packed bytes (2 weights per byte)
    int size;                    // Original float count
};

// // mimics the architecture of Llama
// struct TransformerWeights
// {
//     // token embedding
//     float *token_embedding; // vocab_size x dim

//     // layers
//     struct Layer
//     {
//         // RMSNorm weights
//         float *rms_att_weight; // dim - Input norm
//         float *rms_ffn_weight; // dim - Post Attention norm

//         QuantizedTensor wq, wk, wv, wo; // Attention weights
//         QuantizedTensor w1, w2, w3; // MLP weights

//         // Attention
//         float *wq; // dim x dim
//         float *wk; // kv_dim x dim - smaller than wq!
//         float *wv; // kv_dim x dim
//         float *wo; //  dim x dim

//         // MLP (Feed Forward Network)
//         float *w1; // hidden_dim x dim - gate
//         float *w2; // dim x hidden_dim - down
//         float *w3; // hidden_dim x dim - up
//     };
//     std::vector<Layer> layers;

//     // final output
//     float *rms_final_weight; // dim
//     float *w_cls;            // vocab_size x dim - output head

//     // float *w_cls; // vocab_size x dim
//     QuantizedTensor w_cls;            // vocab_size x dim - output head
// };

struct TransformerWeights
{
    // Embeddings & Norms stay FP32
    std::vector<float> token_embedding;
    std::vector<float> rms_final_weight;

    // Output Head is quantized to save memory
    QuantizedTensor w_cls;

    struct Layer
    {
        std::vector<float> rms_att_weight;
        std::vector<float> rms_ffn_weight;

        QuantizedTensor wq, wk, wv, wo;
        QuantizedTensor w1, w2, w3;
    };
    std::vector<Layer> layers;
};

// Global state
LLMConfig config;
TransformerWeights w;
// std::vector<float> raw_weights; // giant blob

// To run the model, we need scratchpads (RAM) to store the intermediate calculations.
// dynamoc state during inference
struct RunState
{
    std::vector<float> x, xb; // Activations
    std::vector<float> q, k, v;
    std::vector<float> att;
    std::vector<float> hb, hb2;
    std::vector<float> logits;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
};
RunState state;

// kernels

/// KERNELS
// the engine
// Naive implementaation (0(N^3))
// as (Out, In).
// We want to compute: x @ W.T
// x: (1, K)
// W: (N, K) -> Contiguous in memory!
// out: (1, N)
void matmul(float *out, float *x, float *w, int M, int N, int K)
{

    /*
        i = row of A
        |
        +-- j = element of A[i][:]
            |
            +-- load a_val
            |
            +-- Sweep across row B[j][:] → contiguous
            |
            +-- Update C[i][:] → contiguous
    */

    // intilaize output to zero
    // std::fill(out, out + M * N, 0.0f);

    // Parallelize this loop for massive speedup!
    // #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        float val = 0.0f;
        // Pointer to the start of the weight row for output neuron 'i'
        float *w_row = w + i * K;

        for (int j = 0; j < K; j++)
        {
            val += x[j] * w_row[j];
        }
        out[i] = val;
    }
}

// INT4 Matrix Multiplication optimized with Multi-threading
// out = x @ W.T
void matmul_int4(float *out, float *x, QuantizedTensor &w, int N, int K)
{
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // int block_size = 32;
    // int blocks_per_row = K / block_size;

    auto worker = [&](int start_row, int end_row)
    {
        for (int i = start_row; i < end_row; i++)
        {
            float sum = 0.0f;
            int blocks_per_row = K / 32;

            // Pointers for this row
            float *row_scales = w.scales.data() + i * blocks_per_row;
            uint8_t *row_data = w.q_data.data() + i * blocks_per_row * 16;

            for (int b = 0; b < blocks_per_row; b++)
            {
                float scale = row_scales[b];
                uint8_t *block_bytes = row_data + b * 16;
                float *x_block = x + b * 32;

                for (int j = 0; j < 16; j++)
                {
                    uint8_t packed = block_bytes[j];

                    // Unpack High Nibble (first weight)
                    // -8 shifts range from [0,15] to [-8,7]
                    // But our python script (quantized.py) used [-7,7] shifted by +8.
                    float val1 = (float)((int)(packed >> 4) - 8);
                    sum += (val1 * scale) * x_block[j * 2];

                    // Unpack Low Nibble
                    float val2 = (float)((int)(packed & 0x0F) - 8);
                    sum += (val2 * scale) * x_block[j * 2 + 1];
                }
            }
            out[i] = sum;
        }
    };

    // dispath the threads
    int rows_per_thread = N / num_threads;
    for (int t = 0; t < num_threads; t++)
    {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? N : start + rows_per_thread;
        threads.emplace_back(worker, start, end);
    }

    // join threads
    for (auto &t : threads)
    {
        t.join();
    }
}

// Standard FP32 Matmul (for norms/embeddings if needed, though we usually just copy)
void matmul_fp32(float *out, float *x, std::vector<float> &w, int N, int K)
{
    for (int i = 0; i < N; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < K; j++)
        {
            val += x[j] * w[i * K + j];
        }
        out[i] = val;
    }
}

// # loadings
void load_quantized(std::ifstream &file, QuantizedTensor &qt, int expected_size)
{

    int flag;
    file.read((char *)&flag, sizeof(int));
    if (flag != 1)
    {
        std::cerr << " CRITICAL ERROR: Expected INT4 tensor (flag=1), got " << flag << ".\n";
        exit(1);
    }

    int blocks = expected_size / 32;
    qt.size = expected_size;
    qt.scales.resize(blocks);
    qt.q_data.resize(blocks * 16);

    file.read((char *)qt.scales.data(), blocks * sizeof(float));
    file.read((char *)qt.q_data.data(), blocks * 16);
}

void load_fp32(std::ifstream &file, std::vector<float> &vec, int expected_size)
{

    int flag;
    file.read((char *)&flag, sizeof(int));
    if (flag != 0)
    {
        std::cerr << " CRITICAL ERROR: Expected FP32 tensor (flag=0), got " << flag << ".\n";
        exit(1);
    }

    vec.resize(expected_size);
    file.read((char *)vec.data(), expected_size * sizeof(float));
}

bool load_model(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return false;

    // Check Magic Number
    uint32_t magic;
    file.read((char *)&magic, sizeof(magic));
    if (magic != 0x494E5434)
    {
        std::cerr << "Invalid INT4 Magic!\n";
        return false;
    }

    // [hidden_size (2048), n_layers, n_heads, vocab_size]
    file.read((char *)&config.dim, sizeof(int)); // Read 2048 into DIM (not hidden_dim)
    file.read((char *)&config.n_layers, sizeof(int));
    file.read((char *)&config.n_heads, sizeof(int));
    file.read((char *)&config.vocab_size, sizeof(int));

    // derive dimensions
    config.hidden_dim = 5632; // Hardcode Correct TinyLlama MLP Size
    config.n_kv_heads = 4;
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

    w.layers.resize(config.n_layers);

    load_fp32(file, w.token_embedding, config.vocab_size * config.dim);

    // return true;

    for (int i = 0; i < config.n_layers; i++)
    {
        auto &l = w.layers[i];

        // Attention
        load_quantized(file, l.wq, config.dim * config.dim);
        load_quantized(file, l.wk, kv_dim * config.dim);
        load_quantized(file, l.wv, kv_dim * config.dim);
        load_quantized(file, l.wo, config.dim * config.dim);

        // MLP
        load_quantized(file, l.w1, config.hidden_dim * config.dim);
        load_quantized(file, l.w3, config.hidden_dim * config.dim);
        load_quantized(file, l.w2, config.dim * config.hidden_dim);

        // Norms (FP32)
        load_fp32(file, l.rms_att_weight, config.dim);
        load_fp32(file, l.rms_ffn_weight, config.dim);
    }

    std::cout << "we in here.\n";

    // Output Head (Quantized)
    // file.read((char *)&flag, sizeof(int));

    load_fp32(file, w.rms_final_weight, config.dim);
    load_quantized(file, w.w_cls, config.vocab_size * config.dim);

    return true;
}

void malloc_run_state()
{
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

    // resize short-term  buffers
    state.x.resize(config.dim);
    state.xb.resize(config.dim);

    state.q.resize(config.dim);
    state.k.resize(kv_dim);
    state.v.resize(kv_dim);

    state.att.resize(config.n_heads * config.seq_len);

    state.hb.resize(config.hidden_dim);
    state.hb2.resize(config.hidden_dim);

    state.logits.resize(config.vocab_size);

    // resize long-term memory (KV Cache)
    int cache_size = config.n_layers * config.seq_len * kv_dim;
    std::cout << "Allocating KV Cache (" << (cache_size * 4 * 2) / (1024 * 1024) << " MB)...\n";
    state.key_cache.resize(cache_size);
    state.value_cache.resize(cache_size);
}

// void malloc_weights()
// {
//     // point to raw_weights data
//     float *ptr = raw_weights.data();

//     // token embedding
//     w.token_embedding = ptr; // vocab_size x dim
//     ptr += config.vocab_size * config.dim;

//     // iterate through layers
//     w.layers.resize(config.n_layers);

//     // calculate dimensions for pointer mesh
//     int head_dim = config.dim / config.n_heads; // 64
//     int kv_dim = config.n_kv_heads * head_dim;  // 256

//     for (int i = 0; i < config.n_layers; i++)
//     {
//         TransformerWeights::Layer &layer = w.layers[i];

//         // Attention
//         layer.wq = ptr; // dim x dim
//         ptr += config.dim * config.dim;
//         layer.wk = ptr; // kv_dim x dim
//         ptr += kv_dim * config.dim;
//         layer.wv = ptr; // kv_dim x dim
//         ptr += kv_dim * config.dim;
//         layer.wo = ptr; // dim x dim
//         ptr += config.dim * config.dim;

//         // MLP (Feed Forward Network) -> gate, up and down
//         layer.w1 = ptr; // hidden_dim x dim - gate
//         ptr += config.hidden_dim * config.dim;
//         layer.w3 = ptr; // hidden_dim x dim - up
//         ptr += config.hidden_dim * config.dim;
//         layer.w2 = ptr; // dim x hidden_dim - down
//         ptr += config.dim * config.hidden_dim;

//         // RMSNorm weights
//         layer.rms_att_weight = ptr; // dim
//         ptr += config.dim;
//         layer.rms_ffn_weight = ptr; // dim
//         ptr += config.dim;
//     }

//     // final output
//     w.rms_final_weight = ptr; // dim
//     ptr += config.dim;
//     w.w_cls = ptr; // vocab_size x dim
//     ptr += config.vocab_size * config.dim;

//     // sanity check
//     size_t floats_read = ptr - raw_weights.data();
//     std::cout << "mapped " << floats_read << " floats for parameters.\n";
//     std::cout << "Total blob size: " << raw_weights.size() << " floats.\n";
//     if (floats_read != raw_weights.size())
//     {
//         std::cerr << "Warning: Parameter count mismatch! Check offsets.\n";
//     }
//     else
//     {
//         std::cout << "Memory layout Exact Match.\n";
//     }
// }

void forward(int token, int position)
{
    // embedding lookup
    // copy the embedding vector for this token into x
    float *embed = w.token_embedding.data() + token * config.dim;
    std::copy(embed, embed + config.dim, state.x.begin());

    // shortcuts forr dimensions
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    int head_dim = dim / config.n_heads;
    int kv_dim = (dim * config.n_kv_heads) / config.n_heads;
    int kv_mul = config.n_heads / config.n_kv_heads;

    // iterate through layers
    for (int layer_idx = 0; layer_idx < config.n_layers; layer_idx++)
    {
        auto &layer = w.layers[layer_idx];

        // RMSNorm before Attention
        rmsnorm(state.xb.data(), state.x.data(), layer.rms_att_weight.data(), dim);

        // Compute QKV (INT4 MatMuls)
        matmul_int4(state.q.data(), state.xb.data(), layer.wq, dim, dim);
        matmul_int4(state.k.data(), state.xb.data(), layer.wk, kv_dim, dim);
        matmul_int4(state.v.data(), state.xb.data(), layer.wv, kv_dim, dim);

        // RoPE positional embeddings
        apply_rope(state.q.data(), state.k.data(), position, head_dim, config.n_heads, config.n_kv_heads);

        // Save to KV cache
        int offset = layer_idx * config.seq_len * kv_dim + position * kv_dim;
        std::copy(state.k.begin(), state.k.end(), state.key_cache.data() + offset);
        std::copy(state.v.begin(), state.v.end(), state.value_cache.data() + offset);

        // multi-head attention and rest of the transformer operations
        // iterate over all the heads based on the number of heads (blocks)
        for (int head = 0; head < config.n_heads; head++)
        {
            // get pointer to this head's Q vector
            float *q = state.q.data() + head * head_dim;

            // calculate attention scores for this head
            float *att = state.att.data() + head * config.seq_len;

            // iterate over all past tokens up to current position (0 to position)
            for (int t = 0; t <= position; t++)
            {
                // get pointer to cached K vector
                float *k = state.key_cache.data() + layer_idx * config.seq_len * kv_dim + t * kv_dim + (head / kv_mul) * head_dim;

                // dot product Q.K
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++)
                {
                    score += q[d] * k[d];
                }
                // att[t] = score / sqrtf((float)head_dim); // scale
                att[t] = score / sqrtf(head_dim); // scale
            }

            // softmax (only up to pos)
            softmax(att, position + 1);

            // weighted sum: score * V
            float *xb_head = state.xb.data() + head * head_dim;

            // output buffer for this head
            std::fill(xb_head, xb_head + head_dim, 0.0f);

            for (int t = 0; t <= position; t++)
            {
                float *v = state.value_cache.data() + layer_idx * config.seq_len * kv_dim + t * kv_dim + (head / kv_mul) * head_dim;
                float a = att[t];

                for (int d = 0; d < head_dim; d++)
                {
                    xb_head[d] += a * v[d];
                }
            }
        }

        // Output (INT4)
        // Output projection (Wo)
        // we used xb as temporary buffer for attention output above
        // lets use state.q as temp buffer for the result of Wo to be safe
        // FFN Weights

        // output projection (Wo) - Result goes into state.q (as temp buffer)
        matmul_int4(state.q.data(), state.xb.data(), layer.wo, dim, dim);

        // Residual connection (add to state.x)
        accum(state.x.data(), state.q.data(), dim);

        // FEED FORWARD BLOCK
        // RMSNorm (Normalize x into xb again)
        rmsnorm(state.xb.data(), state.x.data(), layer.rms_ffn_weight.data(), dim);
        matmul_int4(state.hb.data(), state.xb.data(), layer.w1, hidden_dim, dim);
        matmul_int4(state.hb2.data(), state.xb.data(), layer.w3, hidden_dim, dim);

        // SwiGLU Activation
        silu(state.hb.data(), hidden_dim);                          // Sigmoid(Gate) * Gate
        element_mul(state.hb.data(), state.hb2.data(), hidden_dim); // Gate * Up

        //  Down Projection
        matmul_int4(state.xb.data(), state.hb.data(), layer.w2, dim, hidden_dim);

        // Residual connection
        accum(state.x.data(), state.xb.data(), dim);
    }

    // final output
    // Final RMSNorm
    rmsnorm(state.x.data(), state.x.data(), w.rms_final_weight.data(), dim);
    // clasffier (logits)
    matmul_int4(state.logits.data(), state.x.data(), w.w_cls, config.vocab_size, dim);
}

// simple text loader
struct Tokenizer
{
    std::vector<std::string> vocab;

    void load(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            std::cerr << " Error: Could not load vocab file from: " << path << "\n";
            std::cerr << "   Did you run 'python scripts/export_vocab.py'?\n";
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            size_t pos = 0;
            while ((pos = line.find("\\n", pos)) != std::string::npos)
            {
                line.replace(pos, 2, "\n");
                pos += 1;
            }
            vocab.push_back(line);
        }
        std::cout << "\nLoaded " << vocab.size() << " tokens.\n";
    }
    std::string decode(int id) {
        if (id >= vocab.size()) return "<?>";
        std::string token = vocab[id];
        size_t pos = 0;
        while ((pos = token.find("\xe2\x96\x81", pos)) != std::string::npos) {
            token.replace(pos, 3, " ");
            pos += 1;
        }
        return token;
    }
};

int main(int argc, char **argv)
{
    std::cout << " Edge LLM INT4 Engine...\n";
    if (argc < 2)
    {
        std::cout << "Usage: ./edge_llm <model_int4.bin>\n";
        return 1;
    }

    if (!load_model(argv[1]))
    {
        std::cerr << "Load Failed\n";
        return 1;
    }

    // std::cout << "\n\n\n--- RUNNING MODEL ---\n";

    malloc_run_state();

    Tokenizer tokenizer;
    tokenizer.load("../vocab.txt");
    if (tokenizer.vocab.empty())
        tokenizer.load("vocab.txt");

    int token = 1; // BOS
    int pos = 0;

    std::cout << "\n--- GENERATING ---\n";
    // Timing
    auto start = std::chrono::high_resolution_clock::now();

    for (; pos < 30; pos++)
    {
        forward(token, pos);

        int next_token = 0;
        float max_val = -1e9;
        for (int i = 0; i < config.vocab_size; i++)
        {
            if (state.logits[i] > max_val)
            {
                max_val = state.logits[i];
                next_token = i;
            }
        }

        std::cout << tokenizer.decode(next_token)  <<" " << std::flush;
        token = next_token;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = end - start;
    std::cout << "\n\nSpeed: " << 30 / dt.count() << " t/s\n";
    

    std::cout << "\n\nDone.\n";
    return 0;
}
