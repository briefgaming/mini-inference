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
    def __init__(
        self,
        model_config: ModelConfig,
        dequantize_to_fp32: bool = False,
        quantize_to_int8: bool = False,
    ):
        if dequantize_to_fp32 and quantize_to_int8:
            raise ValueError("Choose either --dequantize-fp32 or --quantize-int8, not both.")
        self.model_config = model_config
        self.dequantize_to_fp32 = dequantize_to_fp32
        self.quantize_to_int8 = quantize_to_int8

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

        if dtype == "BF16" and self.dequantize_to_fp32:
            tensor_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            if transpose:
                tensor_u16 = tensor_u16.T.copy()
            tensor_f32 = self._bf16_to_fp32(tensor_u16)
            return tensor_f32.tobytes(), list(tensor_f32.shape), "F32"

        if self.dequantize_to_fp32 and dtype not in ("BF16", "F32", "F16"):
            raise ValueError(
                f"FP32 dequantization requested, but tensor '{key}' has unsupported dtype {dtype}."
            )

        if not transpose:
            return raw, shape, "F32" if self.dequantize_to_fp32 and dtype in ("F32", "F16") else dtype

        tensor = np.frombuffer(raw, dtype=self._numpy_dtype(dtype)).reshape(shape)
        tensor_t = tensor.T.copy()
        
        ret_dtype = "F32" if self.dequantize_to_fp32 and dtype in ("F32", "F16") else dtype
        if self.dequantize_to_fp32 and dtype == "F16":
             tensor_t = tensor_t.astype(np.float32)
             
        return tensor_t.tobytes(), list(tensor_t.shape), ret_dtype

    def _extract_tensor_f32(self, key: str, transpose: bool):
        offset = self.header[key]["data_offsets"]
        shape = self.header[key]["shape"]
        dtype = self.header[key]["dtype"]

        start_idx = self.header_offset + offset[0]
        end_idx = start_idx + (offset[1] - offset[0])
        raw = self.mm[start_idx:end_idx]

        if dtype == "BF16":
            tensor = self._bf16_to_fp32(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
        else:
            tensor = np.frombuffer(raw, dtype=self._numpy_dtype(dtype)).reshape(shape).astype(
                np.float32, copy=False
            )

        if transpose:
            tensor = tensor.T.copy()
        else:
            tensor = np.ascontiguousarray(tensor)
        return tensor, list(tensor.shape), "F32"

    @staticmethod
    def _scale_name(key: str) -> str:
        return f"{key}.scales"

    @staticmethod
    def _should_quantize_tensor(key: str) -> bool:
        quantized_suffixes = (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        )
        return key == "model.embed_tokens.weight" or key.endswith(quantized_suffixes)

    @staticmethod
    def _quantize_int8(tensor: np.ndarray, scale_axis: int):
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32, copy=False)

        q = np.empty(tensor.shape, dtype=np.int8)
        scale_len = tensor.shape[scale_axis]
        scales = np.empty(scale_len, dtype=np.float32)
        chunk = 2048

        if scale_axis == 0:
            for start in range(0, scale_len, chunk):
                end = min(start + chunk, scale_len)
                tensor_chunk = tensor[start:end, :]
                max_abs = np.max(np.abs(tensor_chunk), axis=1)
                chunk_scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(
                    np.float32
                )
                scales[start:end] = chunk_scales
                q[start:end, :] = np.clip(
                    np.rint(tensor_chunk / chunk_scales[:, None]), -127, 127
                ).astype(np.int8)
        elif scale_axis == 1:
            for start in range(0, scale_len, chunk):
                end = min(start + chunk, scale_len)
                tensor_chunk = tensor[:, start:end]
                max_abs = np.max(np.abs(tensor_chunk), axis=0)
                chunk_scales = np.where(max_abs > 0.0, max_abs / 127.0, 1.0).astype(
                    np.float32
                )
                scales[start:end] = chunk_scales
                q[:, start:end] = np.clip(
                    np.rint(tensor_chunk / chunk_scales[None, :]), -127, 127
                ).astype(np.int8)
        else:
            raise ValueError(f"Unsupported scale axis {scale_axis}")

        return np.ascontiguousarray(q), scales

    def _store_tensor(
        self,
        key: str,
        buffer: bytes,
        out_shape: list[int],
        out_dtype: str,
        transposed: bool,
        scale_name: str = "",
    ):
        padded_size = self._store_weights(buffer)
        tensor_meta = {
            "offset": self.size,
            "size": len(buffer),
            "padded_size": padded_size,
            "shape": out_shape,
            "dtype": out_dtype,
            "transposed": transposed,
        }
        if scale_name:
            tensor_meta["scale_name"] = scale_name
        self.layer_tensor[key] = tensor_meta
        self.size += padded_size

    def _calculate_quantized_offsets(self, key: str, transpose: bool):
        tensor_f32, out_shape, _ = self._extract_tensor_f32(key, transpose=transpose)
        scale_axis = 0 if key == "model.embed_tokens.weight" else 1
        q_tensor, scales = self._quantize_int8(tensor_f32, scale_axis=scale_axis)
        scale_name = self._scale_name(key)
        self._store_tensor(
            key,
            q_tensor.tobytes(),
            list(q_tensor.shape),
            "I8",
            transpose,
            scale_name=scale_name,
        )
        self._store_tensor(
            scale_name,
            np.ascontiguousarray(scales).tobytes(),
            [int(scales.shape[0])],
            "F32",
            False,
        )

    def _calculate_offsets(self, key: str):
        if self.quantize_to_int8 and self._should_quantize_tensor(key):
            self._calculate_quantized_offsets(key, transpose=False)
            return

        if self.quantize_to_int8:
            tensor_f32, out_shape, out_dtype = self._extract_tensor_f32(
                key, transpose=False
            )
            self._store_tensor(
                key,
                tensor_f32.tobytes(),
                out_shape,
                out_dtype,
                False,
            )
            return

        buffer, out_shape, out_dtype = self._extract_tensor_payload(
            key, transpose=False
        )
        self._store_tensor(key, buffer, out_shape, out_dtype, False)

    def _calculate_offsets_t(self, key: str):
        if self.quantize_to_int8 and self._should_quantize_tensor(key):
            self._calculate_quantized_offsets(key, transpose=True)
            return

        if self.quantize_to_int8:
            tensor_f32, out_shape, out_dtype = self._extract_tensor_f32(
                key, transpose=True
            )
            self._store_tensor(
                key,
                tensor_f32.tobytes(),
                out_shape,
                out_dtype,
                True,
            )
            return

        buffer_t, out_shape, out_dtype = self._extract_tensor_payload(
            key, transpose=True
        )
        self._store_tensor(key, buffer_t, out_shape, out_dtype, True)

    def _extract_header_data(self) -> dict:
        header = self.mm.read(8)
        n = int.from_bytes(header, byteorder="little")
        header_bytes = self.mm.read(n)
        header_data = json.loads(header_bytes)
        self.header_offset = n + 8
        return header_data

    def _extract_layer_tensors(self) -> int:
        # Embeddings
        self._calculate_offsets("model.embed_tokens.weight")

        # Llama 3.2 1B has 16 layers
        for i in range(16):
            # Input layer norm / attention block
            self._calculate_offsets(f"model.layers.{i}.input_layernorm.weight")

            # Projections: Transpose for faster GEMV operations
            self._calculate_offsets_t(f"model.layers.{i}.self_attn.q_proj.weight")
            self._calculate_offsets_t(f"model.layers.{i}.self_attn.k_proj.weight")
            self._calculate_offsets_t(f"model.layers.{i}.self_attn.v_proj.weight")
            self._calculate_offsets_t(f"model.layers.{i}.self_attn.o_proj.weight")

            # MLP feed forward
            self._calculate_offsets(f"model.layers.{i}.post_attention_layernorm.weight")

            # SwiGLU activation
            self._calculate_offsets_t(f"model.layers.{i}.mlp.down_proj.weight")
            self._calculate_offsets_t(f"model.layers.{i}.mlp.gate_proj.weight")
            self._calculate_offsets_t(f"model.layers.{i}.mlp.up_proj.weight")

        self._calculate_offsets("model.norm.weight")

    def mmap_weights(self) -> int:
        self._extract_layer_tensors()
        self.tensor_file.close()
        self._file.close()
        print(f"Saved weights to {self.model_config.tensor_info_name}")
        return self.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract safetensors weights and optionally export FP32 or weight-only INT8."
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
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Store quantized INT8 weights with companion FP32 scale tensors.",
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
        quantize_to_int8=args.quantize_int8,
    )
    size = mmw.mmap_weights()
    mmw._save_index()
    print(f"{size / 1024 / 1024:.2f} MB")
