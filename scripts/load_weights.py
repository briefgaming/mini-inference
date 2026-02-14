from huggingface_hub import hf_hub_download
from dataclasses import dataclass
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
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

        self.checkpoint = self._pull_hf_model()
        self._file = open(self.checkpoint, "rb")
        self.mm = mmap.mmap(self._file.fileno(), length=0, access=mmap.ACCESS_READ)
        self.header = self._extract_header_data()

        self.header_offset = 0
        self.layer_tensor = {}
        self.size = 0
        self.tensor_file = open(self.model_config.tensor_info_name, "wb")

    def _pull_hf_model(self):
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

    def _calculate_offsets(self, key: str):
        offset = self.header[key]["data_offsets"]
        shape = self.header[key]["shape"]
        dtype = self.header[key]["dtype"]

        start_idx = (
            self.header_offset + offset[0]
        )  # Starting offset excludes header size
        end_idx = start_idx + (offset[1] - offset[0])
        buffer = self.mm[start_idx:end_idx]

        padded_size = self._store_weights(buffer)

        self.layer_tensor[key] = {
            "offset": self.size,
            "size": len(buffer),
            "padded_size": padded_size,
            "shape": shape,
            "dtype": dtype,
            "transposed": False,
        }

        return padded_size

    def _calculate_offsets_t(self, key: str):
        # print(f"Processing {self.header=}")
        offset = self.header[key]["data_offsets"]
        shape = self.header[key]["shape"]
        dtype = self.header[key]["dtype"]

        start_idx = (
            self.header_offset + offset[0]
        )  # Starting offset excludes header size
        end_idx = start_idx + (offset[1] - offset[0])

        w = np.frombuffer(
            self.mm[start_idx:end_idx], dtype=np.uint16
        )  # Zero-copy transformation
        w = w.reshape(shape)
        w = w.T.copy()

        buffer_t = w.tobytes()
        padded_size = self._store_weights(buffer_t)

        self.layer_tensor[key] = {
            "offset": self.size,
            "size": len(buffer_t),
            "padded_size": padded_size,
            "shape": list(reversed(shape)),
            "dtype": dtype,
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
    mmw = MemoryMapWeights(
        ModelConfig(
            "meta-llama/Llama-3.2-1B",
            "model.safetensors",
            "llama.bin",
            "model_index.json",
        )
    )
    size = mmw.mmap_weights()
    mmw._save_index()
    print(f"{size / 1024 / 1024:.2f} MB")
