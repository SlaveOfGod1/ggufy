"""ggufy GUI - a tkinter interface mirroring the original DVUI/SDL GUI.

Supports: file selection (dialog + drag-and-drop), model inspection, and
conversion between GGUF and SafeTensors with the same options as the CLI.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback
from typing import Any, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import convert as conv
from . import image_arch as arch_mod
from . import types as types_mod
from .convert import ConvertOptions, QuantizationFamilies
from .file_loader import TensorFile
from .types import FileType

VERSION = "0.1.0"

GGUF_TARGET_TYPES = ["f32", "f16", "bf16", "q2_k", "q3_k", "q4_0", "q4_1",
                     "q4_k", "q5_0", "q5_1", "q5_k", "q6_k", "q8_0"]

ST_TARGET_TYPES = ["F32", "F16", "BF16", "F8_E4M3", "Scaled F8_E4M3",
                   "F8_E5M2", "MXFP8_E4M3", "NVFP4", "INT8", "INT8 ConvRot",
                   "INT4 ConvRot", "INT4 ConvRot SR", "W4A8 ConvRot"]

ST_DTYPE_MAP = {
    "Scaled F8_E4M3": "SCALED_F8_E4M3",
    "INT8 ConvRot": "INT8_CONVROT",
    "INT4 ConvRot": "INT4_CONVROT",
    "INT4 ConvRot SR": "INT4_CONVROT_SR",
    "W4A8 ConvRot": "ASYM_W4A8_INT8",
}


class Callbacks:
    def __init__(self, state: "GuiState"):
        self.state = state

    def is_cancelled(self):
        return self.state.cancel_requested

    def report_progress(self, done, total, name, src_type, dst_type, n_elements):
        self.state.progress_q.put((done, total, name, src_type, dst_type, n_elements))


class GuiState:
    def __init__(self):
        self.loaded_file: Optional[TensorFile] = None
        self.file_selected: Optional[str] = None
        self.load_error: Optional[str] = None

        self.target_filetype = FileType.GGUF
        self.target_dtype: Optional[str] = None
        self.target_folder = ""
        self.target_filename = ""
        self.filename_base_stem = ""
        self.prev_target_dtype = None
        self.prev_template_path_len = 0
        self.target_threads = max(1, os.cpu_count() or 4)
        self.cpu_count = max(1, os.cpu_count() or 4)
        self.target_aggressiveness = 50
        self.skip_sensitivity = False
        self.model_only = False
        self.allow_unknown_arch = False
        self.allow_upscale = False
        self.arch_override = ""
        self.sensitivity_path: Optional[str] = None
        self.template_path: Optional[str] = None
        self.use_quant_types: Optional[str] = None
        self.stochastic_rounding: Optional[int] = None

        self.convert_state = "idle"  # idle/converting/done/err
        self.convert_progress = 0
        self.convert_total = 0
        self.convert_error: Optional[str] = None
        self.convert_elapsed = 0.0
        self.convert_output_path: Optional[str] = None
        self.cancel_requested = False

        self.convert_tensor_name = ""
        self.convert_tensor_src_type = ""
        self.convert_tensor_dst_type = ""
        self.convert_tensor_elements = 0

        self.predicted_size: Optional[int] = None
        self.prev_pred_signature = None

        self.tool_status = ""
        self.tool_status_is_error = False
        self.same_file_error = False

        self.progress_q: "queue.Queue[Any]" = queue.Queue()
        self.result_q: "queue.Queue[Any]" = queue.Queue()

    def prediction_signature(self) -> str:
        return "|".join([
            self.target_filetype,
            self.target_dtype or "",
            str(self.target_aggressiveness),
            str(self.skip_sensitivity),
            str(self.model_only),
            str(self.allow_unknown_arch),
            str(self.allow_upscale),
            self.arch_override,
            self.template_path or "",
            self.sensitivity_path or "",
            self.file_selected or "",
            str(self.target_threads),
        ])


class GgufyGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = GuiState()
        self.root.title(f"ggufy {VERSION}")
        self.root.geometry("820x640")
        self.root.minsize(620, 480)
        self._build_ui()
        self._poll_queues()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # --- File section
        file_frame = ttk.LabelFrame(main, text="Model file", padding=6)
        file_frame.pack(fill=tk.X, **pad)
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).pack(side=tk.LEFT, padx=4)
        self.file_info_var = tk.StringVar(value="No file loaded.")
        ttk.Label(file_frame, textvariable=self.file_info_var, foreground="#666").pack(fill=tk.X, **pad)

        # --- Conversion options
        opt = ttk.LabelFrame(main, text="Conversion options", padding=6)
        opt.pack(fill=tk.BOTH, expand=True, **pad)

        row0 = ttk.Frame(opt)
        row0.pack(fill=tk.X, **pad)
        ttk.Label(row0, text="Output format:").pack(side=tk.LEFT)
        self.filetype_var = tk.StringVar(value="gguf")
        self.filetype_cb = ttk.Combobox(row0, textvariable=self.filetype_var, state="readonly",
                                        values=["gguf", "safetensors"], width=14)
        self.filetype_cb.pack(side=tk.LEFT, padx=4)
        self.filetype_cb.bind("<<ComboboxSelected>>", lambda e: self._on_filetype_change())
        ttk.Label(row0, text="Datatype:").pack(side=tk.LEFT, padx=(16, 0))
        self.dtype_var = tk.StringVar()
        self.dtype_cb = ttk.Combobox(row0, textvariable=self.dtype_var, state="readonly", width=22)
        self.dtype_cb.pack(side=tk.LEFT, padx=4)
        self.dtype_cb.bind("<<ComboboxSelected>>", lambda e: self._schedule_prediction())

        row1 = ttk.Frame(opt)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Output folder:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar()
        ttk.Entry(row1, textvariable=self.folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row1, text="...", width=3, command=self._browse_folder).pack(side=tk.LEFT)

        row2 = ttk.Frame(opt)
        row2.pack(fill=tk.X, **pad)
        ttk.Label(row2, text="Output name:").pack(side=tk.LEFT)
        self.name_var = tk.StringVar()
        ttk.Entry(row2, textvariable=self.name_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(row2, text="Threads:").pack(side=tk.LEFT, padx=(16, 0))
        self.threads_var = tk.StringVar(value=str(self.state.target_threads))
        ttk.Spinbox(row2, from_=1, to=128, textvariable=self.threads_var, width=5).pack(side=tk.LEFT, padx=4)

        row3 = ttk.Frame(opt)
        row3.pack(fill=tk.X, **pad)
        ttk.Label(row3, text="Aggressiveness:").pack(side=tk.LEFT)
        self.agg_var = tk.IntVar(value=50)
        self.agg_scale = ttk.Scale(row3, from_=1, to=100, variable=self.agg_var, orient=tk.HORIZONTAL)
        self.agg_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.agg_label = ttk.Label(row3, text="50", width=4)
        self.agg_label.pack(side=tk.LEFT)
        self.agg_scale.bind("<ButtonRelease-1>", lambda e: self._on_agg_change())

        row4 = ttk.Frame(opt)
        row4.pack(fill=tk.X, **pad)
        self.skip_sens_var = tk.BooleanVar()
        self.model_only_var = tk.BooleanVar()
        self.allow_unknown_var = tk.BooleanVar()
        self.allow_upscale_var = tk.BooleanVar()
        ttk.Checkbutton(row4, text="Skip sensitivity", variable=self.skip_sens_var,
                        command=self._schedule_prediction).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row4, text="Model only", variable=self.model_only_var,
                        command=self._schedule_prediction).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row4, text="Allow unknown arch", variable=self.allow_unknown_var,
                        command=self._schedule_prediction).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row4, text="Allow upscale", variable=self.allow_upscale_var,
                        command=self._schedule_prediction).pack(side=tk.LEFT)

        row5 = ttk.Frame(opt)
        row5.pack(fill=tk.X, **pad)
        ttk.Label(row5, text="Template:").pack(side=tk.LEFT)
        self.template_var = tk.StringVar()
        ttk.Entry(row5, textvariable=self.template_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row5, text="...", width=3, command=self._browse_template).pack(side=tk.LEFT)
        ttk.Button(row5, text="Export", width=6, command=self._export_template).pack(side=tk.LEFT, padx=2)
        ttk.Button(row5, text="Sens.", width=6, command=self._gen_sensitivities).pack(side=tk.LEFT, padx=2)

        row6 = ttk.Frame(opt)
        row6.pack(fill=tk.X, **pad)
        ttk.Label(row6, text="Sensitivities:").pack(side=tk.LEFT)
        self.sens_var = tk.StringVar()
        ttk.Entry(row6, textvariable=self.sens_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row6, text="...", width=3, command=self._browse_sens).pack(side=tk.LEFT)
        ttk.Label(row6, text="Quant types:").pack(side=tk.LEFT, padx=(16, 0))
        self.qtypes_var = tk.StringVar()
        ttk.Entry(row6, textvariable=self.qtypes_var, width=10).pack(side=tk.LEFT, padx=4)

        row7 = ttk.Frame(opt)
        row7.pack(fill=tk.X, **pad)
        ttk.Label(row7, text="Arch override:").pack(side=tk.LEFT)
        self.arch_var = tk.StringVar()
        ttk.Entry(row7, textvariable=self.arch_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # --- Prediction + actions
        act = ttk.Frame(main)
        act.pack(fill=tk.X, **pad)
        self.pred_var = tk.StringVar(value="")
        ttk.Label(act, textvariable=self.pred_var, foreground="#333").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.convert_btn = ttk.Button(act, text="Convert", command=self._start_convert)
        self.convert_btn.pack(side=tk.RIGHT)
        self.cancel_btn = ttk.Button(act, text="Cancel", command=self._cancel_convert, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.RIGHT, padx=4)

        # --- Progress
        prog = ttk.Frame(main)
        prog.pack(fill=tk.X, **pad)
        self.progress = ttk.Progressbar(prog, maximum=100)
        self.progress.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.progress_label = ttk.Label(prog, text="", width=50)
        self.progress_label.pack(side=tk.LEFT, padx=6)

        # --- Tool status
        self.status_var = tk.StringVar(value="")
        ttk.Label(main, textvariable=self.status_var, foreground="#666").pack(fill=tk.X, **pad)

        self._on_filetype_change()

    # ------------------------------------------------------------- helpers
    def _on_filetype_change(self):
        ft = self.filetype_var.get()
        self.state.target_filetype = FileType.GGUF if ft == "gguf" else FileType.SAFETENSORS
        if ft == "gguf":
            self.dtype_cb["values"] = GGUF_TARGET_TYPES
            if self.dtype_var.get() not in GGUF_TARGET_TYPES:
                self.dtype_var.set("")
        else:
            self.dtype_cb["values"] = ST_TARGET_TYPES
            if self.dtype_var.get() not in ST_TARGET_TYPES:
                self.dtype_var.set("")
        self._schedule_prediction()

    def _on_agg_change(self):
        self.agg_label.config(text=str(self.agg_var.get()))
        self._schedule_prediction()

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("Model files", "*.safetensors *.gguf"), ("All files", "*.*")])
        if path:
            self._set_file(path)

    def _set_file(self, path):
        self.state.file_selected = path
        self.file_var.set(path)
        self._load_file_async()

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.folder_var.set(path)
            self.state.target_folder = path

    def _browse_template(self):
        path = filedialog.askopenfilename(title="Select template",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.template_var.set(path)
            self.state.template_path = path
            self._schedule_prediction()

    def _browse_sens(self):
        path = filedialog.askopenfilename(title="Select sensitivities file",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.sens_var.set(path)
            self.state.sensitivity_path = path
            self._schedule_prediction()

    # ------------------------------------------------------------ loading
    def _load_file_async(self):
        self.file_info_var.set("Loading...")
        self.state.load_error = None
        t = threading.Thread(target=self._load_worker, daemon=True)
        t.start()

    def _load_worker(self):
        try:
            tf = TensorFile.load_file(self.state.file_selected)
            self.state.result_q.put(("loaded", tf))
        except Exception as e:
            self.state.result_q.put(("load_error", str(e)))

    # ---------------------------------------------------------- prediction
    def _schedule_prediction(self):
        if self.state.loaded_file is None or not self.state.file_selected:
            return
        try:
            self._update_from_ui()
        except Exception:
            return
        self.pred_var.set("Predicting size...")
        t = threading.Thread(target=self._predict_worker, daemon=True)
        t.start()

    def _predict_worker(self):
        try:
            self._update_from_ui()
            if self.state.template_path:
                suffix = conv.template_type_suffix(self.state.template_path, self.state.target_filetype)
                dt = self.state.target_dtype
            else:
                dt = self.state.target_dtype
            opts = self._build_opts()
            size = conv.predict_output_size(self.state.loaded_file, opts)
            self.state.result_q.put(("predicted", size))
        except Exception as e:
            self.state.result_q.put(("predict_error", str(e)))

    # ------------------------------------------------------------ convert
    def _update_from_ui(self):
        st = self.state
        st.target_filetype = FileType.GGUF if self.filetype_var.get() == "gguf" else FileType.SAFETENSORS
        d = self.dtype_var.get()
        if d:
            st.target_dtype = ST_DTYPE_MAP.get(d, d)
        else:
            st.target_dtype = None
        st.target_folder = self.folder_var.get()
        st.target_filename = self.name_var.get()
        st.target_threads = int(self.threads_var.get() or self.state.cpu_count)
        st.target_aggressiveness = int(self.agg_var.get())
        st.skip_sensitivity = self.skip_sens_var.get()
        st.model_only = self.model_only_var.get()
        st.allow_unknown_arch = self.allow_unknown_var.get()
        st.allow_upscale = self.allow_upscale_var.get()
        st.arch_override = self.arch_var.get().strip() or None
        st.template_path = self.template_var.get().strip() or None
        st.sensitivity_path = self.sens_var.get().strip() or None
        st.use_quant_types = self.qtypes_var.get().strip() or None
        st.target_filename = self.name_var.get().strip()

    def _build_opts(self) -> ConvertOptions:
        st = self.state
        allowed = None
        if st.use_quant_types:
            allowed = QuantizationFamilies.parse(st.use_quant_types)
        filetype = st.target_filetype
        datatype = None
        if st.target_dtype:
            try:
                datatype = types_mod.for_format(st.target_dtype, filetype)
            except ValueError:
                datatype = st.target_dtype
        path = st.file_selected or ""
        if not st.target_folder:
            st.target_folder = os.path.dirname(path) or "."
        return ConvertOptions(
            path=path,
            filetype=filetype,
            datatype=datatype,
            template_path=st.template_path,
            output_dir=st.target_folder,
            output_name=st.target_filename or None,
            threads=st.target_threads,
            skip_sensitivity=st.skip_sensitivity,
            quantization_aggressiveness=float(st.target_aggressiveness),
            sensitivities_path=st.sensitivity_path,
            allowed_quant_families=allowed,
            model_only=st.model_only,
            allow_unknown_arch=st.allow_unknown_arch,
            allow_upscale=st.allow_upscale,
            arch_override=st.arch_override,
            stochastic_rounding=st.stochastic_rounding,
            callbacks=Callbacks(st),
        )

    def _start_convert(self):
        st = self.state
        if st.loaded_file is None or not st.file_selected:
            messagebox.showwarning("ggufy", "No model file loaded.")
            return
        if st.convert_state == "converting":
            return
        try:
            self._update_from_ui()
            opts = self._build_opts()
        except Exception as e:
            messagebox.showerror("ggufy", str(e))
            return
        out_path = conv.compute_output_path(opts)
        if os.path.exists(out_path):
            if not messagebox.askyesno("ggufy", f'"{out_path}" already exists. Overwrite?'):
                return
        if os.path.abspath(out_path) == os.path.abspath(st.file_selected):
            messagebox.showerror("ggufy", "Input and output are the same file.")
            st.same_file_error = True
            return
        if not st.allow_upscale and conv.detect_upscaling(st.loaded_file.tensors, opts.datatype):
            if not messagebox.askyesno("ggufy",
                                       "Source contains lossy-quantized tensors; converting to a higher-precision "
                                       "format will NOT recover lost information. Continue anyway?"):
                return
            st.allow_upscale = True
            opts.allow_upscale = True

        st.convert_state = "converting"
        st.convert_error = None
        st.cancel_requested = False
        st.convert_output_path = out_path
        self.convert_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.progress_label.config(text="Starting...")
        t = threading.Thread(target=self._convert_worker, args=(opts,), daemon=True)
        t.start()

    def _convert_worker(self, opts: ConvertOptions):
        st = self.state
        start = time.time()
        try:
            conv.convert(st.loaded_file, opts)
            st.convert_elapsed = time.time() - start
            st.result_q.put(("converted", st.convert_output_path))
        except Exception as e:
            st.result_q.put(("convert_error", str(e)))
        finally:
            st.convert_state = "idle"

    def _cancel_convert(self):
        self.state.cancel_requested = True

    # ------------------------------------------------------ tool actions
    def _export_template(self):
        st = self.state
        if st.loaded_file is None:
            messagebox.showwarning("ggufy", "No model file loaded.")
            return
        path = filedialog.asksaveasfilename(title="Export template", defaultextension=".json",
                                            initialfile="template.json")
        if not path:
            return
        try:
            arch = arch_mod.detect_arch_from_tensors(st.loaded_file.tensors)
            if st.loaded_file.type == FileType.GGUF:
                with open(path, "w", encoding="utf-8") as w:
                    from .gguf import Gguf
                    g = Gguf(st.file_selected)
                    g.write_template(w)
                    g.close()
            else:
                with open(path, "w", encoding="utf-8") as w:
                    conv.write_template_from_file(st.loaded_file, arch, True, w)
            st.tool_status = f"Template exported to {path}"
            st.tool_status_is_error = False
        except Exception as e:
            st.tool_status = f"Error: {e}"
            st.tool_status_is_error = True
        self.status_var.set(st.tool_status)

    def _gen_sensitivities(self):
        st = self.state
        if st.loaded_file is None:
            messagebox.showwarning("ggufy", "No model file loaded.")
            return
        path = filedialog.asksaveasfilename(title="Generate sensitivities",
                                            defaultextension=".json",
                                            initialfile="sensitivities.json")
        if not path:
            return
        try:
            arch = arch_mod.detect_arch_from_tensors(st.loaded_file.tensors)
            threshold = arch.threshhold if (arch and arch.threshhold is not None) else conv.QUANTIZATION_THRESHOLD
            with open(path, "w", encoding="utf-8") as w:
                conv.generate_sensitivities_from_tensors(st.loaded_file.tensors, arch, threshold, w)
            st.tool_status = f"Sensitivities exported to {path}"
            st.tool_status_is_error = False
        except Exception as e:
            st.tool_status = f"Error: {e}"
            st.tool_status_is_error = True
        self.status_var.set(st.tool_status)

    # ------------------------------------------------------------ polling
    def _poll_queues(self):
        st = self.state
        try:
            while True:
                msg = st.progress_q.get_nowait()
                done, total, name, src_type, dst_type, n_elements = msg
                st.convert_progress = done
                st.convert_total = total
                st.convert_tensor_name = name
                st.convert_tensor_src_type = src_type
                st.convert_tensor_dst_type = dst_type
                st.convert_tensor_elements = n_elements
                if total > 0:
                    self.progress["value"] = done * 100.0 / total
                self.progress_label.config(text=f"{done}/{total} {name} -> {dst_type}")
        except queue.Empty:
            pass

        try:
            while True:
                msg = st.result_q.get_nowait()
                kind = msg[0]
                if kind == "loaded":
                    tf = msg[1]
                    st.loaded_file = tf
                    arch = tf.arch.name if tf.arch else "unknown"
                    self.file_info_var.set(
                        f"{os.path.basename(st.file_selected)} | arch: {arch} | "
                        f"{len(tf.tensors)} tensors | {tf.types_line}")
                    self._auto_name()
                    self._schedule_prediction()
                elif kind == "load_error":
                    st.load_error = msg[1]
                    self.file_info_var.set(f"Load failed: {msg[1]}")
                elif kind == "predicted":
                    size = msg[1]
                    st.predicted_size = size
                    self.pred_var.set(f"Estimated output size: {conv.format_bytes(size)} ({size} bytes)")
                elif kind == "predict_error":
                    self.pred_var.set("")
                elif kind == "converted":
                    self.progress_label.config(text="Done.")
                    self.convert_btn.config(state=tk.NORMAL)
                    self.cancel_btn.config(state=tk.DISABLED)
                    self.pred_var.set("")
                    messagebox.showinfo("ggufy", f"Converted to:\n{msg[1]}")
                elif kind == "convert_error":
                    self.progress_label.config(text="Conversion failed.")
                    self.convert_btn.config(state=tk.NORMAL)
                    self.cancel_btn.config(state=tk.DISABLED)
                    messagebox.showerror("ggufy", str(msg[1]))
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queues)

    def _auto_name(self):
        st = self.state
        if not st.file_selected:
            return
        stem = os.path.splitext(os.path.basename(st.file_selected))[0]
        st.filename_base_stem = stem
        if not st.target_filename or st.target_filename.startswith(stem + "-"):
            st.target_filename = f"{stem}-{st.target_dtype or 'f16'}"
            self.name_var.set(st.target_filename)

    def _on_close(self):
        self.state.cancel_requested = True
        if self.state.loaded_file is not None:
            try:
                self.state.loaded_file.close()
            except Exception:
                pass
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    GgufyGui(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
