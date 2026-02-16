from dataclasses import dataclass
import argparse
import mmap
import json
import numpy as np


@dataclass
class ModelConfig:
    model_name: str
    weights_name: str
    tensor_info_name: str
    indexes_file_name: str


class MemoryMapWeights:
    def __init__(self, model_config: ModelConfig, dequantize_to_fp32: bool = False):
        self.model_config = model_config
        self.dequantize_to_fp32 = dequantize_to_fp32

        self.checkpoint = self._pull_hf_model()
        self._file = open(self.checkpoint, "rb")
        self.mm = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
        self.header_offset = 0
        self.layer_tensor = {}
        self.size = 0
        self.tensor_file = open(self.model_config.tensor_info_name, "wb")
        self.header = self._extract_header_data()

    def _pull_hf_model(self):
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self.model_config.model_name, filename=self.model_config.weights_name
        )

    def _store_weights(self, buf: bytes):
        self.tensor_file.write(buf)

        buf_s = len(buf)
        padded_size = (buf_s + 15) & (~15)  # Round up to the nearest 16 byte boundary
        if padded_size > buf_s:
            self.tensor_file.write(
                bytearray(padded_size - buf_s)
            )  # Ensure next tensor starts at the next location that's a multiple of 16 bytes

        return padded_size

    def _save_index(self):
        file_name = self.model_config.indexes_file_name
        with open(file_name, "w") as f:
            json.dump(self.layer_tensor, f)
        print(f"Saved layer indexes to {file_name}")

    @staticmethod
    def _bf16_to_fp32(u16_array: np.ndarray) -> np.ndarray:
        # BF16 -> FP32 by placing BF16 bits in the high 16 bits of FP32.
        u32 = u16_array.astype(np.uint32) << 16
        return u32.view(np.float32)

    @staticmethod
    def _numpy_dtype(dtype: str):
        mapping = {
            "BF16": np.uint16,
            "F16": np.float16,
            "F32": np.float32,
            "F64": np.float64,
            "I8": np.int8,
            "U8": np.uint8,
            "I16": np.int16,
            "U16": np.uint16,
            "I32": np.int32,
            "U32": np.uint32,
            "I64": np.int64,
            "U64": np.uint64,
            "BOOL": np.bool_,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype in safetensors header: {dtype}")
        return mapping[dtype]

    def _extract_tensor_payload(self, key: str, transpose: bool):
        offset = self.header[key]["data_offsets"]
        shape = self.header[key]["shape"]
        dtype = self.header[key]["dtype"]

        start_idx = (
            self.header_offset + offset[0]
        )  # Starting offset excludes header size
        end_idx = start_idx + (offset[1] - offset[0])
        raw = self.mm[start_idx:end_idx]

        if self.dequantize_to_fp32:
            if dtype != "BF16":
                raise ValueError(
                    f"FP32 dequantization requested, but tensor '{key}' has dtype {dtype}."
                )
            tensor_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            if transpose:
                tensor_u16 = tensor_u16.T.copy()
            tensor_f32 = self._bf16_to_fp32(tensor_u16)
            return tensor_f32.tobytes(), list(tensor_f32.shape), "F32"

        if not transpose:
            return raw, shape, dtype

        tensor = np.frombuffer(raw, dtype=self._numpy_dtype(dtype)).reshape(shape)
        tensor_t = tensor.T.copy()
        return tensor_t.tobytes(), list(tensor_t.shape), dtype

    def _calculate_offsets(self, key: str):
        buffer, out_shape, out_dtype = self._extract_tensor_payload(
            key, transpose=False
        )
        padded_size = self._store_weights(buffer)

        self.layer_tensor[key] = {
            "offset": self.size,
            "size": len(buffer),
            "padded_size": padded_size,
            "shape": out_shape,
            "dtype": out_dtype,
            "transposed": False,
        }

        return padded_size

    def _calculate_offsets_t(self, key: str):
        buffer_t, out_shape, out_dtype = self._extract_tensor_payload(
            key, transpose=True
        )
        padded_size = self._store_weights(buffer_t)

        self.layer_tensor[key] = {
            "offset": self.size,
            "size": len(buffer_t),
            "padded_size": padded_size,
            "shape": out_shape,
            "dtype": out_dtype,
            "transposed": True,
        }

        return padded_size

    def _extract_header_data(self) -> dict:
        header = self.mm.read(8)
        n = int.from_bytes(header, byteorder="little")
        header_bytes = self.mm.read(n)
        header_data = json.loads(header_bytes)
        self.header_offset = n + 8
        return header_data

    def _extract_layer_tensors(self) -> int:
        # Embeddings
        self.size += self._calculate_offsets("model.embed_tokens.weight")

        # Llama 3.2 1B has 16 layers
        for i in range(16):
            # Input layer norm / attention block
            self.size += self._calculate_offsets(
                f"model.layers.{i}.input_layernorm.weight"
            )

            # Projections: Transpose for faster GEMV operations
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.self_attn.q_proj.weight"
            )
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.self_attn.k_proj.weight"
            )
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.self_attn.v_proj.weight"
            )
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.self_attn.o_proj.weight"
            )

            # MLP feed forward
            self.size += self._calculate_offsets(
                f"model.layers.{i}.post_attention_layernorm.weight"
            )

            # SwiGLU activation
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.mlp.down_proj.weight"
            )
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.mlp.gate_proj.weight"
            )
            self.size += self._calculate_offsets_t(
                f"model.layers.{i}.mlp.up_proj.weight"
            )

        self.size += self._calculate_offsets("model.norm.weight")

    def mmap_weights(self) -> int:
        self._extract_layer_tensors()
        self.tensor_file.close()
        self._file.close()
        print(f"Saved weights to {self.model_config.tensor_info_name}")
        return self.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract safetensors weights and optionally dequantize BF16 -> FP32."
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--weights", default="model.safetensors")
    parser.add_argument("--out-bin", default="llama.bin")
    parser.add_argument("--out-index", default="model_index.json")
    parser.add_argument(
        "--dequantize-fp32",
        action="store_true",
        help="Store output tensors as FP32 (dequantized from BF16).",
    )
    args = parser.parse_args()

    mmw = MemoryMapWeights(
        ModelConfig(
            args.model,
            args.weights,
            args.out_bin,
            args.out_index,
        ),
        dequantize_to_fp32=args.dequantize_fp32,
    )
    size = mmw.mmap_weights()
    mmw._save_index()
    print(f"{size / 1024 / 1024:.2f} MB")
