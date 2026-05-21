import argparse
import os
import json
import logging
from typing import Optional, Dict, Any

try:
    from safetensors import safe_open
    import numpy as np
    import gguf
except ImportError:
    print("Missing dependencies. Please install: pip install safetensors numpy gguf")
    exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import struct

def read_safetensors_header(path):
    with open(path, "rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            return {}, 0
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        return json.loads(header_bytes.decode('utf-8')), header_size

def load_tensor_pure(file_path, key, header, header_size):
    tensor_info = header.get(key)
    if not tensor_info:
        raise KeyError(f"Tensor {key} not found in header")
    
    dtype = tensor_info["dtype"]
    shape = tensor_info["shape"]
    offsets = tensor_info["data_offsets"]
    
    start = 8 + header_size + offsets[0]
    length = offsets[1] - offsets[0]
    
    with open(file_path, "rb") as f:
        f.seek(start)
        raw_bytes = f.read(length)
        
    if dtype == "F32":
        return np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
    elif dtype == "F16":
        return np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
    elif dtype == "BF16":
        bf16_data = np.frombuffer(raw_bytes, dtype=np.uint16)
        f32_data = (bf16_data.astype(np.uint32) << 16).view(np.float32)
        return f32_data.reshape(shape)
    elif dtype == "I32":
        return np.frombuffer(raw_bytes, dtype=np.int32).reshape(shape)
    elif dtype == "I64":
        return np.frombuffer(raw_bytes, dtype=np.int64).reshape(shape)
    else:
        return np.frombuffer(raw_bytes, dtype=np.uint8)

def cmd_header(args):
    if args.input_file.endswith('.safetensors'):
        try:
            header, _ = read_safetensors_header(args.input_file)
            keys = [k for k in header.keys() if k != "__metadata__"]
            print(f"Safetensors Header for {args.input_file}")
            print(f"Tensor Count: {len(keys)}")
            if "__metadata__" in header and header["__metadata__"]:
                print("Metadata exists.")
        except Exception as e:
            logging.error(f"Failed to read header: {e}")
    elif args.input_file.endswith('.gguf'):
        reader = gguf.GGUFReader(args.input_file)
        print(f"GGUF Header for {args.input_file}")
        print(f"Version: {reader.version}")
        print(f"Tensor Count: {len(reader.tensors)}")
        print(f"KV Fields: {len(reader.fields)}")
    else:
        logging.error("Unsupported file format. Must be .safetensors or .gguf")

def cmd_tree(args):
    if not args.input_file.endswith('.safetensors'):
        logging.error("Tree visualization currently only supported for safetensors.")
        return

    try:
        header, _ = read_safetensors_header(args.input_file)
        tree = {}
        for key in header.keys():
            if key == "__metadata__":
                continue
            parts = key.split('.')
            curr = tree
            for part in parts[:-1]:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            curr[parts[-1]] = header[key].get("shape")
            
        print(json.dumps(tree, indent=2, default=str))
    except Exception as e:
        logging.error(f"Failed to generate tree: {e}")

def cmd_metadata(args):
    if args.input_file.endswith('.safetensors'):
        try:
            header, _ = read_safetensors_header(args.input_file)
            metadata = header.get("__metadata__", {})
            if metadata:
                for k, v in metadata.items():
                    print(f"{k}: {v}")
            else:
                print("No metadata found.")
        except Exception as e:
            logging.error(f"Failed to read metadata: {e}")
    elif args.input_file.endswith('.gguf'):
        reader = gguf.GGUFReader(args.input_file)
        for key, field in reader.fields.items():
            val = field.parts[field.data[0]] if field.data else "<empty>"
            print(f"{key}: {val}")
    else:
        logging.error("Unsupported file format.")

def cmd_template(args):
    if not args.input_file.endswith('.gguf'):
        logging.error("Template extraction is only supported from .gguf files.")
        return
        
    reader = gguf.GGUFReader(args.input_file)
    template = {"metadata": {}, "tensors": {}}
    
    for tensor in reader.tensors:
        template["tensors"][tensor.name] = {
            "type": tensor.tensor_type.name,
            "shape": list(tensor.shape)
        }
        
    out_path = "template.json"
    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)
    logging.info(f"Exported template to {out_path}")

def detect_architecture(keys):
    keys_set = set(keys)
    if any("double_blocks" in k for k in keys_set):
        return "flux"
    if any("joint_blocks" in k for k in keys_set):
        return "sd3"
    if any("label_emb" in k for k in keys_set) or any("embed_ers" in k for k in keys_set):
        return "sdxl"
    return "sd1"

def cmd_convert(args):
    path = args.input_file
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(path))
    stem = args.output_name or os.path.splitext(os.path.basename(path))[0]
    dtype_str = args.datatype.lower() if args.datatype else "f16"
    
    out_file = os.path.join(out_dir, f"{stem}-{dtype_str}.{args.filetype}")
    
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Converting {path} -> {out_file} (dtype={dtype_str})")

    # Load sensitivities if available
    sensitivities = {}
    if args.sensitivities:
        try:
            with open(args.sensitivities, "r") as f:
                sensitivities = json.load(f)
            logging.info(f"Loaded sensitivities from {args.sensitivities}")
        except Exception as e:
            logging.warning(f"Could not load sensitivities file: {e}")

    if args.filetype == 'gguf':
        try:
            header, header_size = read_safetensors_header(path)
            tensors = [k for k in header.keys() if k != "__metadata__"]
            
            arch = detect_architecture(tensors)
            logging.info(f"Detected architecture: {arch}")
            
            writer = gguf.GGUFWriter(out_file, arch)
            writer.add_quantization_version(2)
            
            metadata = header.get("__metadata__", {})
            for k, v in metadata.items():
                if isinstance(v, str):
                    writer.add_string(k, v)
                else:
                    writer.add_string(k, str(v))
            for i, key in enumerate(tensors):
                tensor = load_tensor_pure(path, key, header, header_size)
                
                # Convert names by stripping prefixes if required (basic logic)
                tensor_name = key
                if args.model_only and tensor_name.startswith("model."):
                    tensor_name = tensor_name[len("model."):]
                    
                # Target dtype logic
                target_dtype = dtype_str
                
                if dtype_str == "f16":
                    tensor = tensor.astype(np.float16)
                elif dtype_str == "bf16":
                    tensor = tensor.astype(np.float32) 
                elif dtype_str == "f32":
                    tensor = tensor.astype(np.float32)
                elif "q" in dtype_str:
                    logging.warning(f"Quantization type {dtype_str} is best handled by llama.cpp. Using F16 for pure python fallback.")
                    tensor = tensor.astype(np.float16)
                else:
                    tensor = tensor.astype(np.float16)

                writer.add_tensor(tensor_name, tensor)
                if i % 50 == 0:
                    logging.info(f"Processed {i}/{len(tensors)} tensors...")
                        
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            logging.info("Conversion to GGUF complete!")
            
        except Exception as e:
            logging.error(f"Conversion failed: {e}")
            if os.path.exists(out_file):
                os.remove(out_file)
    else:
        logging.error("Only safetensors to GGUF conversion is fully mocked in this python port.")

def main():
    parser = argparse.ArgumentParser(description="ggufy python port - A lightweight tensor format converter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert command
    parser_convert = subparsers.add_parser("convert", help="Convert model format/quantization")
    parser_convert.add_argument("input_file", help="Input safetensors file")
    parser_convert.add_argument("-d", "--datatype", help="Target quantization type (default: f16)", default="f16")
    parser_convert.add_argument("-f", "--filetype", help="Target file format", default="gguf", choices=["gguf", "safetensors"])
    parser_convert.add_argument("-o", "--output-dir", help="Output directory")
    parser_convert.add_argument("-n", "--output-name", help="Output filename without extension")
    parser_convert.add_argument("-t", "--template", help="Use a JSON template for conversion")
    parser_convert.add_argument("-j", "--threads", type=int, help="Number of threads (ignored in python port)", default=4)
    parser_convert.add_argument("-a", "--aggressiveness", type=float, help="Aggressiveness (ignored in python fallback)", default=50)
    parser_convert.add_argument("-x", "--skip-sensitivity", action="store_true", help="Skip sensitivity")
    parser_convert.add_argument("-s", "--sensitivities", help="Path to sensitivities JSON file")
    parser_convert.add_argument("-m", "--model-only", action="store_true", help="Convert only main model")
    
    # Header command
    parser_header = subparsers.add_parser("header", help="Display file header information")
    parser_header.add_argument("input_file", help="Input model file")

    # Tree command
    parser_tree = subparsers.add_parser("tree", help="Display tensor hierarchy")
    parser_tree.add_argument("input_file", help="Input model file")

    # Metadata command
    parser_metadata = subparsers.add_parser("metadata", help="Display all metadata key-value pairs")
    parser_metadata.add_argument("input_file", help="Input model file")

    # Template command
    parser_template = subparsers.add_parser("template", help="Export GGUF structure to JSON template")
    parser_template.add_argument("input_file", help="Input model file")

    args = parser.parse_args()

    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "header":
        cmd_header(args)
    elif args.command == "tree":
        cmd_tree(args)
    elif args.command == "metadata":
        cmd_metadata(args)
    elif args.command == "template":
        cmd_template(args)

if __name__ == "__main__":
    main()
