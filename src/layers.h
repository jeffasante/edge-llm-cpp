#pragma once
#include <vector>
#include <cmath>

// Root Mean Square Normalization -> Keeps numbers stable so they don't explode.
// x: input vector (size n)
// out: output vector (size n)
// weight: learned scale parameter (size n)
void rmsnorm(float *out, float *x, float *weight, int n)
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int i = 0; i < n; i++)
    {
        ss += x[i] * x[i];
    }

    // calculate normalization factor
    // epsilon to avoid division by zero
    float val = 1.0f / sqrt(ss / n + 1e-5f);

    // normalize and scale
    for (int i = 0; i < n; i++)
    {
        out[i] = weight[i] * (val * x[i]);
    }
}

// Sigmoid Linear unit
// x= x * sigmoid(x)
void silu(float *out, int n)
{
    for (int i = 0; i < n; i++)
    {
        float sigmoid = 1.0f / (1.0f + expf(-out[i]));
        out[i] = out[i] * sigmoid;
    }
}

// element-wise multiplication (for SwiGLU)
// out = out * other
void element_mul(float *a, float *other, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] *= other[i];
    }
}

// RoPE (Rotary Positional Embeddings)
// this rotates the vectors to encode position information
void apply_rope(float *q, float *k, int position,
                int head_dim, int n_heads, int n_kv_heads)
{
    float theta_scale = powf(10000.0f, -2.0f / head_dim);

    for (int h = 0; h < n_heads; h++)
    {
        for (int i = 0; i < head_dim; i += 2){
            // calculate rotation angle for this position and dimension
            float theta = position * powf(theta_scale, i / 2.0f);
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            // rotate q
            float* q_ptr = q + h * head_dim + i;
            float q0 = q_ptr[0];
            float q1 = q_ptr[1];
            q_ptr[0] = q0 * cos_theta - q1 * sin_theta;
            q_ptr[1] = q0 * sin_theta + q1 * cos_theta;

            // rotate K (only if within kv_heads range)
            if (h < n_kv_heads)
            {
                float* k_ptr = k + h * head_dim + i;
                float k0 = k_ptr[0];
                float k1 = k_ptr[1];
                k_ptr[0] = k0 * cos_theta - k1 * sin_theta;
                k_ptr[1] = k0 * sin_theta + k1 * cos_theta;
            }
        }
       
    }
}

// softmax -> converts logits to probabilities that sum to 1
void softmax(float *logits, int n)
{
    // find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n; i++)
    {
        if (logits[i] > max_logit)
            max_logit = logits[i];
    }

    // compute exponentials and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++)
    {
        logits[i] = expf(logits[i] - max_logit);
        sum_exp += logits[i];
    }

    // normalize to get probabilities
    for (int i = 0; i < n; i++)
    {
        logits[i] /= sum_exp;
    }
}


// Accumakte (residual connection)
// x += y
void accum(float *x, float *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        x[i] += y[i];
    }
}