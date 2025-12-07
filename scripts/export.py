
import numpy as np
import torch
import struct
import os
from transformers import AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "model_int4.bin"
BLOCK_SIZE = 32

def quantize_and_pack(data):
    # Reshape into blocks of 32
    # Pad if necessary (though TinyLlama dims are usually divisible by 32)
    padding = (-len(data)) % BLOCK_SIZE
    if padding != 0:
        data = np.pad(data, (0, padding))
        
    n_blocks = len(data) // BLOCK_SIZE
    blocks = data.reshape(n_blocks, BLOCK_SIZE)
    
    # 1. Calculate Scale (Abs Max / 7)
    max_val = np.max(np.abs(blocks), axis=1)
    max_val = np.maximum(max_val, 1e-5) 
    scales = max_val / 7.0 
    
    # 2. Quantize to signed int8 [-7, 7]
    q_blocks = np.round(blocks / scales[:, None]).astype(np.int8)
    q_blocks = np.clip(q_blocks, -7, 7)
    
    # 3. Offset to unsigned [1, 15] for storage
    q_blocks = q_blocks + 8 
    q_blocks = q_blocks.astype(np.uint8)
    
    # 4. Pack 2 numbers into 1 byte
    # pairs: [n_blocks, 16, 2]
    pairs = q_blocks.reshape(n_blocks, BLOCK_SIZE//2, 2)
    # High nibble << 4 | Low nibble
    packed = (pairs[:, :, 0] << 4) | pairs[:, :, 1]
    
    return scales.astype(np.float32), packed.flatten()

def export_int4():
    print(f"  Loading {MODEL_ID}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    except:
        print("Please run: pip install torch transformers numpy")
        return

    print(f" Quantizing to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "wb") as f:
        # Magic Header: 'INT4'
        f.write(struct.pack("I", 0x494E5434)) 
        config = model.config
        f.write(struct.pack("i", config.hidden_size))
        f.write(struct.pack("i", config.num_hidden_layers))
        f.write(struct.pack("i", config.num_attention_heads))
        f.write(struct.pack("i", config.vocab_size))

        total_size = 0
        
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().astype(np.float32)
            data_flat = data.flatten()
            
            # We quantize ONLY the big MatMul weights
            # We keep Norms and Embeddings in FP32 for accuracy
            is_linear = "weight" in name and "norm" not in name and "embed" not in name
            
            if is_linear:
                scales, packed = quantize_and_pack(data_flat)
                # Flag 1: Quantized
                f.write(struct.pack("i", 1)) 
                f.write(scales.tobytes())
                f.write(packed.tobytes())
                print(f"    {name}: {data_flat.nbytes//1024}KB -> {(scales.nbytes+packed.nbytes)//1024}KB")
                total_size += scales.nbytes + packed.nbytes
            else:
                # Flag 0: FP32
                f.write(struct.pack("i", 0))
                f.write(data_flat.tobytes())
                print(f"    {name}: FP32")
                total_size += data_flat.nbytes

    print(f" Done! File size: {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    export_int4()