## Mini Inference

Implementation of an inference engine for Llama3 variant architectures.

Goal is to have a well optimized engine suitable for personal use.

### How to run
```bash
# Download weights
# Note: This will download the weights to your current directory
python3 scripts/load_weights.py --model meta-llama/Llama-3.2-1B --weights model.safetensors --out-bin llama.bin --out-index configs/model_index.json --dequantize-fp32

# Generate vocabulary
python3 scripts/tokenizer_script.py --out vocab.bin --byte-level

# Script usage
python3 scripts/load_weights.py --help

# Downlaod json.hpp
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp

# Compile code and execute binary
g++ -std=c++23 -o inference bpe.cpp model.cpp weights.cpp main.cpp && MAX_NEW_TOKENS=32 ./inference
```
### Note
1. Uses modern C++ features so compile with C++20 and above

References
1. [Llama2.c](https://github.com/karpathy/llama2.c)
2. [InferGPT](https://github.com/JINO-ROHIT/inferGPT/tree/main)