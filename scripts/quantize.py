import numpy as np
import torch
import struct
from transformers import AutoModelForCausalLM
import os

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_FILE = "model_int4.bin"
BLOCK_SIZE = 32  # Group 32 weights together

def quantize_and_pack(data):
    # padding
    padding = (-len(data)) % BLOCK_SIZE
    if padding != 0:
        data = np.pad(data, (0, padding))

    # data is a 1D numpy array of float32
    n_blocks = len(data) // BLOCK_SIZE
    blocks = data.reshape(n_blocks, BLOCK_SIZE)
    
    # scale
    # calculate scale for each block
    max_val = np.max(np.abs(blocks), axis=1)
    # avoid divide by zero
    # max_val[max_val == 0] = 1e-5
    max_val = np.maximum(max_val, 1e-5)
    # scale maps range [-max, max] to [-7,7 ]
    scales = max_val / 7.0
    
    # quantize
    # BLOCKS/SCALES[:,NONE] broadcasts the scale
    q_blocks = np.round(blocks / scales[:, None]).astype(np.int8)
    # packed = quantized.tobytes()
    # clip to ensure we stay in [-7,7] range (signed 4-bit)
    q_blocks = np.clip(q_blocks, -7, 7)
    # pack into uint8
    # We pack two 4-bit numbers into one 8-bit byte.
    # High nibble | Low nibble
    # Since q_blocks are signed (-7..7), we add 8 to make them unsigned (1..15) for storage
    # This maps -7->1, 0->8, 7->15. (0 is reserved or just unused)
    q_blocks = q_blocks + 8
    # split into pairs
    q_blocks = q_blocks.astype(np.uint8)

    # pack 2
    # [n_blocks, 16,2] -> 16 bytes per block
    # we effectively compress 32 weights into 16 bytes
    pairs = q_blocks.reshape(n_blocks, BLOCK_SIZE//2, 2)
    # pack: (high << 4) | low
    packed = (pairs[:,:,0] << 4) | pairs[:, :,1]

    return  scales.astype(np.float32), packed.flatten()

def write_tensor(f, tensor, name, quantize=False):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data_flat = data.flatten()
    
    if quantize:
        # quantize
        scales, packed = quantize_and_pack(data_flat)
        
        # flag 1 = quantized
        f.write(struct.pack("i", 1))
        f.write(scales.tobytes())
        f.write(packed.tobytes())
        print(f"  {name}: INT4 ({data.nbytes//1024} KB -> {(scales.nbytes+packed.nbytes)//1024} KB)")
    else:
        # flag 0 = not quantized
        f.write(struct.pack("i", 0))
        f.write(data_flat.tobytes())
        print(f"  {name}: FP32")
        
def export_int4():
    print("f loading {MODEL_ID}...")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                            dtype=torch.float32)
    
    config = model.config
    print(f"quantizing to {OUTPUT_FILE} (BLOCK SIZE: {BLOCK_SIZE})...")
    
    with open(OUTPUT_FILE, "wb") as f:
        # HEADER (Magic 0x494E5434 = 'INT4')
        f.write(struct.pack("I", 0x494E5434)) 
        f.write(struct.pack("i", config.hidden_size))
        f.write(struct.pack("i", config.num_hidden_layers))
        f.write(struct.pack("i", config.num_attention_heads))
        f.write(struct.pack("i", config.vocab_size))
        
        write_tensor(f, model.model.embed_tokens.weight,\
            "token_emb", quantize=False)
        
        # Layers (Strict Loop)
        for i in range(config.num_hidden_layers):
            layer = model.model.layers[i]
            print(f"Processing Layer {i}...")
            
            # ATTENTION (Quantized)
            write_tensor(f, layer.self_attn.q_proj.weight, f"l{i}.wq", True)
            write_tensor(f, layer.self_attn.k_proj.weight, f"l{i}.wk", True)
            write_tensor(f, layer.self_attn.v_proj.weight, f"l{i}.wv", True)
            write_tensor(f, layer.self_attn.o_proj.weight, f"l{i}.wo", True)
            
            # MLP (Quantized)
            write_tensor(f, layer.mlp.gate_proj.weight, f"l{i}.w1", True)
            write_tensor(f, layer.mlp.up_proj.weight,   f"l{i}.w3", True)
            write_tensor(f, layer.mlp.down_proj.weight, f"l{i}.w2", True)
            
            # NORMS (FP32)
            write_tensor(f, layer.input_layernorm.weight, f"l{i}.att_norm", False)
            write_tensor(f, layer.post_attention_layernorm.weight, f"l{i}.ffn_norm", False)

            
        # 3. Final Norm & Head
        write_tensor(f, model.model.norm.weight, "final_norm", False)
        write_tensor(f, model.lm_head.weight, "lm_head", True)

    print(" Done!")
    # print(f"Done! Final size: {total_bytes / (1024*1024):.2f} MB")
  
if __name__ == "__main__":
    export_int4()