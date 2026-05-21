import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging

try:
    from safetensors import safe_open
    import numpy as np
    import gguf
except ImportError:
    messagebox.showerror("Missing Dependencies", "Please install dependencies:\npip install safetensors numpy gguf")
    sys.exit(1)

import json
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

def detect_architecture(keys):
    keys_set = set(keys)
    if any("double_blocks" in k for k in keys_set):
        return "flux"
    if any("joint_blocks" in k for k in keys_set):
        return "sd3"
    if any("label_emb" in k for k in keys_set) or any("embed_ers" in k for k in keys_set):
        return "sdxl"
    return "sd1"

# Helper function from our CLI tool
def do_convert(input_file, output_dir, output_name, datatype, filetype, model_only, progress_callback, log_callback):
    try:
        out_dir = output_dir or os.path.dirname(os.path.abspath(input_file))
        stem = output_name or os.path.splitext(os.path.basename(input_file))[0]
        dtype_str = datatype.lower() if datatype else "f16"
        
        out_file = os.path.join(out_dir, f"{stem}-{dtype_str}.{filetype}")
        os.makedirs(out_dir, exist_ok=True)
        
        log_callback(f"Starting conversion: {input_file} -> {out_file}")
        
        if filetype == 'gguf':
            header, header_size = read_safetensors_header(input_file)
            tensors = [k for k in header.keys() if k != "__metadata__"]
            
            arch = detect_architecture(tensors)
            log_callback(f"Detected architecture: {arch}")
            
            writer = gguf.GGUFWriter(out_file, arch)
            writer.add_quantization_version(2)
            
            metadata = header.get("__metadata__", {})
            for k, v in metadata.items():
                if isinstance(v, str):
                    writer.add_string(k, v)
                else:
                    writer.add_string(k, str(v))
            total = len(tensors)
            
            for i, key in enumerate(tensors):
                tensor = load_tensor_pure(input_file, key, header, header_size)
                tensor_name = key
                
                if model_only and tensor_name.startswith("model."):
                    tensor_name = tensor_name[len("model."):]
                    
                if dtype_str == "f16":
                    tensor = tensor.astype(np.float16)
                elif dtype_str == "f32":
                    tensor = tensor.astype(np.float32)
                elif "q" in dtype_str:
                    if i == 0:
                        log_callback(f"Warning: Quantization {dtype_str} not fully supported in pure Python. Falling back to F16.")
                    tensor = tensor.astype(np.float16)
                else:
                    tensor = tensor.astype(np.float16)

                writer.add_tensor(tensor_name, tensor)
                
                if i % max(1, total // 100) == 0:
                    progress_callback(int((i / total) * 100))
                    
            log_callback("Writing headers and metadata...")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            
            log_callback("Writing tensors to file...")
            writer.write_tensors_to_file()
            writer.close()
            
            progress_callback(100)
            log_callback("Conversion to GGUF complete!")
            return True
        else:
            log_callback("Only safetensors to GGUF conversion is fully mocked in this python port.")
            return False
            
    except Exception as e:
        log_callback(f"Error during conversion: {e}")
        return False


class GGUFyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUFy - Python Rewrite")
        self.root.geometry("600x650")
        self.root.configure(padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(root, text="GGUFy Converter", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        # --- File Selection ---
        file_frame = ttk.LabelFrame(root, text="Input / Output")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Input File (.safetensors):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.input_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(file_frame, text="Output Name (optional):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_name_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_name_var, width=40).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # --- Conversion Settings ---
        settings_frame = ttk.LabelFrame(root, text="Settings")
        settings_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(settings_frame, text="Target Datatype:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.dtype_var = tk.StringVar(value="f16")
        dtype_combo = ttk.Combobox(settings_frame, textvariable=self.dtype_var, values=["f16", "f32", "bf16", "q4_k", "q8_0"], state="readonly", width=15)
        dtype_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(settings_frame, text="Output Format:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.format_var = tk.StringVar(value="gguf")
        format_combo = ttk.Combobox(settings_frame, textvariable=self.format_var, values=["gguf"], state="readonly", width=15)
        format_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.model_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Convert Model Only (Strip 'model.' prefixes)", variable=self.model_only_var).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # --- Inspection Actions ---
        inspect_frame = ttk.LabelFrame(root, text="Inspect Tensor File")
        inspect_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(inspect_frame, text="View Header / Metadata", command=self.view_metadata).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(inspect_frame, text="View Tensor Tree", command=self.view_tree).pack(side=tk.LEFT, padx=10, pady=10)
        
        # --- Run Button & Progress ---
        self.convert_btn = ttk.Button(root, text="Start Conversion", command=self.start_conversion)
        self.convert_btn.pack(pady=15)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)
        
        # --- Logs ---
        ttk.Label(root, text="Logs:").pack(anchor=tk.W)
        self.log_text = tk.Text(root, height=10, width=65, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def log(self, message):
        self.root.after(0, self._append_log, message)
        
    def _append_log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def set_progress(self, val):
        self.root.after(0, self._update_progress, val)
        
    def _update_progress(self, val):
        self.progress['value'] = val
        self.root.update_idletasks()

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors"), ("GGUF", "*.gguf"), ("All Files", "*.*")])
        if filename:
            self.input_var.set(filename)
            if not self.output_dir_var.get():
                self.output_dir_var.set(os.path.dirname(filename))

    def browse_output(self):
        dirname = filedialog.askdirectory()
        if dirname:
            self.output_dir_var.set(dirname)

    def view_metadata(self):
        infile = self.input_var.get()
        if not infile or not os.path.exists(infile):
            messagebox.showerror("Error", "Please select a valid input file first.")
            return
            
        self.log("--- METADATA ---")
        if infile.endswith('.safetensors'):
            try:
                header, _ = read_safetensors_header(infile)
                keys = [k for k in header.keys() if k != "__metadata__"]
                self.log(f"Keys: {len(keys)}")
                metadata = header.get("__metadata__", {})
                if metadata:
                    for k, v in metadata.items():
                        self.log(f"{k}: {v}")
                else:
                    self.log("No metadata found.")
            except Exception as e:
                self.log(f"Error: {e}")
        elif infile.endswith('.gguf'):
            try:
                reader = gguf.GGUFReader(infile)
                for key, field in reader.fields.items():
                    val = field.parts[field.data[0]] if field.data else "<empty>"
                    self.log(f"{key}: {val}")
            except Exception as e:
                self.log(f"Error: {e}")

    def view_tree(self):
        infile = self.input_var.get()
        if not infile or not os.path.exists(infile):
            messagebox.showerror("Error", "Please select a valid input file first.")
            return
            
        if not infile.endswith('.safetensors'):
            self.log("Tree view is only supported for .safetensors files currently.")
            return
            
        self.log("--- TENSOR TREE ---")
        try:
            header, _ = read_safetensors_header(infile)
            keys = [k for k in header.keys() if k != "__metadata__"]
            self.log(f"Found {len(keys)} tensors.")
            for k in keys[:50]:
                self.log(f"{k} : {header[k].get('shape')}")
            if len(keys) > 50:
                self.log(f"... and {len(keys)-50} more.")
        except Exception as e:
            self.log(f"Error reading tree: {e}")

    def start_conversion(self):
        infile = self.input_var.get()
        if not infile or not os.path.exists(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
            
        self.convert_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        self.log("--- STARTING CONVERSION ---")
        
        # Run in background thread
        thread = threading.Thread(target=self.run_conversion_thread)
        thread.daemon = True
        thread.start()
        
    def run_conversion_thread(self):
        success = do_convert(
            input_file=self.input_var.get(),
            output_dir=self.output_dir_var.get(),
            output_name=self.output_name_var.get(),
            datatype=self.dtype_var.get(),
            filetype=self.format_var.get(),
            model_only=self.model_only_var.get(),
            progress_callback=self.set_progress,
            log_callback=self.log
        )
        
        self.root.after(0, lambda: self.convert_btn.config(state=tk.NORMAL))
        if success:
            self.root.after(0, lambda: messagebox.showinfo("Success", "Conversion finished successfully!"))
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", "Conversion encountered an error. Check logs."))

if __name__ == "__main__":
    root = tk.Tk()
    app = GGUFyGUI(root)
    root.mainloop()
