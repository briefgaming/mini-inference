## Mini Inference

An implementation of a mini inference engine for Llama3 LLM architectures.

Goal is to have a well optimized engine suitable for personal use.

### How to run
```bash
# Downlaod json.hpp
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp

# Compile code and execute binary
g++ -o inference  bpe.cpp model.cpp weights.cpp main.cpp -std=c++23 && ./inference
```
### Note
1. Uses modern C++ features so compile with C++20 and above

References
1. [Llama2.c](https://github.com/karpathy/llama2.c)
2. [InferGPT](https://github.com/JINO-ROHIT/inferGPT/tree/main)