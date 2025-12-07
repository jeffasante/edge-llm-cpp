from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Exporting {tokenizer.vocab_size} tokens to vocab.txt...")

with open("../vocab.txt", "w", encoding="utf-8") as f:
    for i in range(tokenizer.vocab_size):
        # Get the raw token piece directly from SentencePiece
        token_str = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        
        # Escape newlines so the C++ loader can read each token on one line
        token_str = token_str.replace("\n", "\\n").replace("\r", "")

        f.write(token_str + "\n")

print("vocab.txt saved!")
