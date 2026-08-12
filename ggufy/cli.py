"""ggufy command-line interface.

Port of src/cli/main.zig. Commands: header, tree, metadata, convert,
template, names, sensitivities, version. Options mirror the Zig CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np

from . import convert as conv
from . import image_arch as arch_mod
from . import tensor_clusters as tc
from . import types as types_mod
from .convert import ConvertOptions, QuantizationFamilies
from .gguf import Gguf
from .safetensor import Safetensors
from .types import FileType

VERSION = "0.1.0"

UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]


def format_bytes(bytes_: int) -> str:
    value = float(bytes_)
    unit = 0
    while value >= 1024.0 and unit < len(UNITS) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.2f} {UNITS[unit]}"


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"ggufy: error: {message}\n")


def _add_shared(parser):
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help and exit.")
    parser.add_argument("-d", "--datatype", type=str, default=None,
                        help="When converting, the target datatype (default fp16).")
    parser.add_argument("-f", "--filetype", type=str, default="gguf",
                        help="When converting, the target filetype: gguf (default), safetensors.")
    parser.add_argument("-t", "--template", type=str, default=None,
                        help="When converting, specify a template to use.")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Output directory (default: same as source file).")
    parser.add_argument("-n", "--output-name", type=str, default=None,
                        help="Output filename without extension (default: source name + datatype).")
    parser.add_argument("-j", "--threads", type=int, default=None,
                        help="Threads to use when quantizing. Defaults to number of cores.")
    parser.add_argument("-a", "--aggressiveness", type=int, default=50,
                        help="How aggressively to quantize layers when using sensitivity. "
                             "100 is most aggressive, 1 is least.")
    parser.add_argument("-x", "--skip-sensitivity", action="store_true",
                        help="Pass this to not use a built-in layer sensitivity file and "
                             "just blindly quantize to target type.")
    parser.add_argument("-s", "--sensitivities", type=str, default=None,
                        help="Path to a sensitivities JSON file to use (overrides built-in "
                             "sensitivities) Sensitivities are only used for GGUF model output.")
    parser.add_argument("-q", "--use-quant-types", type=str, default=None,
                        help="Quantization families to use with sensitivity (e.g. \"k\", "
                             "\"0,k\", \"0,1,k\"). Default: match datatype.")
    parser.add_argument("-m", "--model-only", action="store_true",
                        help="When output is safetensors, convert only the main model "
                             "(UNet/transformer). Ignored for GGUF output.")
    parser.add_argument("-u", "--allow-unknown-arch", action="store_true",
                        help="Allow converting files with unrecognized architectures. "
                             "Results may be suboptimal.")
    parser.add_argument("-U", "--allow-upscale", action="store_true",
                        help="Allow converting from a lower-precision (quantized/FP8) source "
                             "to a higher-precision target. The extra bits are fill-in; no "
                             "quality is recovered.")
    parser.add_argument("-A", "--arch", type=str, default=None,
                        help="Set the architecture name written to the GGUF metadata "
                             "(GGUF output only). Free-form; does not affect conversion behaviour.")
    parser.add_argument("-R", "--stochastic-rounding", type=int, default=None,
                        help="Seed for INT4_CONVROT_SR stochastic rounding. Omit for the "
                             "built-in default seed; pass 0 to disable (deterministic, for "
                             "comparison). Ignored by other types.")
    parser.add_argument("-c", "--calculate-size", action="store_true",
                        help="With convert: compute and print the exact final output size "
                             "without writing any file.")
    parser.add_argument("-S", "--shapes", action="store_true",
                        help="With names: emit {\"name\":...,\"shape\":[...]} objects instead "
                             "of bare names, for architectures detected by shape.")


# Options that consume a following value. Used by _extract_positionals to
# avoid mistaking option values for positional arguments.
_VALUE_OPTIONS = {
    "-d", "--datatype", "-f", "--filetype", "-t", "--template", "-o",
    "--output-dir", "-n", "--output-name", "-j", "--threads", "-a",
    "--aggressiveness", "-s", "--sensitivities", "-q", "--use-quant-types",
    "-A", "--arch", "-R", "--stochastic-rounding",
}


def _extract_positionals(argv):
    """Split argv into (command, [remaining args]).

    Walks argv, skipping the values consumed by the value-taking options, so
    `convert -d q4_k file.safetensors` yields command=convert and the file.
    """
    command = None
    filename = None
    rest = []
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("-"):
            rest.append(arg)
            if arg in _VALUE_OPTIONS and i + 1 < n:
                rest.append(argv[i + 1])
                i += 1
        else:
            if command is None:
                command = arg
            elif filename is None:
                filename = arg
        i += 1
    return command, filename, rest


def build_parser() -> _Parser:
    parser = _Parser(prog="ggufy",
                     description="ggufy is a tool for LLM model files, "
                                 "particularly for converting between file types.",
                     add_help=False)
    _add_shared(parser)
    parser.add_argument("positionals", nargs="*",
                        help="Command then input file: header, tree, metadata, "
                             "convert, template, names, sensitivities, version, "
                             "followed by FILENAME.")
    return parser


def print_help():
    print("ggufy is a tool for LLM model files, particularly for converting between file types.\n")
    print("Usage: ggufy <COMMAND> <FILENAME> [options]\n")
    print("Possible commands:")
    print("  header         Shows header information for the specified file")
    print("  tree           Output tensor data in a tree format (SafeTensors only)")
    print("  metadata       Shows metadata information for the specified file")
    print("  convert        Convert the specified file into a different format or datatype")
    print("  template       Creates a json template from the specified file")
    print("  names          Dump tensor names as a JSON array (for test fixtures; -S to include shapes)")
    print("  sensitivities  Generate a sensitivities JSON template from the specified file")
    print("  version        Print version information\n")
    print("Options:")
    build_parser().print_help()


def report_predicted_size(f, opts: ConvertOptions):
    try:
        size = conv.predict_output_size(f, opts)
    except ValueError as e:
        if str(e) == "UnknownArchitecture":
            print("ERROR: Architecture not recognized. Pass --allow-unknown-arch (-u) "
                  "to calculate size anyway. Results may be suboptimal.")
            return
        raise
    print(f"Estimated output size: {format_bytes(size)} ({size} bytes)")


def dump_names(tensors, with_shapes: bool):
    if with_shapes:
        entries = [{"name": t.name, "shape": t.dims} for t in tensors]
        print(json.dumps(entries, indent=1))
    else:
        names = [t.name for t in tensors]
        print(json.dumps(names, indent=2))


def _make_convert_opts(args, path: str) -> ConvertOptions:
    filetype = FileType.parse_from_string(args.filetype) if args.filetype else FileType.GGUF
    datatype = None
    if args.datatype:
        dt_lower = args.datatype.lower()
        if filetype == FileType.SAFETENSORS:
            datatype = types_mod.for_format(dt_lower, FileType.SAFETENSORS)
        else:
            datatype = types_mod.for_format(dt_lower, FileType.GGUF)

    allowed_quant_families = None
    if args.use_quant_types:
        allowed_quant_families = QuantizationFamilies.parse(args.use_quant_types)

    threads = args.threads if args.threads else max(1, os.cpu_count() or 1)

    return ConvertOptions(
        path=path,
        filetype=filetype,
        datatype=datatype,
        template_path=args.template,
        output_dir=args.output_dir,
        output_name=args.output_name,
        threads=threads,
        skip_sensitivity=args.skip_sensitivity,
        quantization_aggressiveness=float(args.aggressiveness),
        sensitivities_path=args.sensitivities,
        allowed_quant_families=allowed_quant_families,
        model_only=args.model_only,
        allow_unknown_arch=args.allow_unknown_arch,
        allow_upscale=args.allow_upscale,
        arch_override=args.arch,
        stochastic_rounding=args.stochastic_rounding,
    )


def main(argv=None):
    start = time.time()
    parser = build_parser()
    command, filename, option_argv = _extract_positionals(
        argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(option_argv)

    if args.help:
        print_help()
        return 0

    if command is None:
        print("ERROR: No command given. Use --help to get more information.", file=sys.stderr)
        return 1

    if command == "version":
        print(f"ggufy {VERSION}")
        return 0

    if filename is None:
        print("ERROR: No model file specified.", file=sys.stderr)
        return 1

    filetype = FileType.parse_from_string(args.filetype) if args.filetype else FileType.GGUF
    datatype = None
    if args.datatype:
        try:
            datatype = types_mod.from_string(args.datatype)
        except ValueError:
            datatype = args.datatype
    try:
        conv.validate_datatype_for_filetype(datatype, filetype)
    except ValueError:
        return 1

    path = filename
    convert_opts = _make_convert_opts(args, path)

    with open(path, "rb") as f:
        try:
            file_type = FileType.detect_from_file(f)
        except ValueError:
            file_type = FileType.SAFETENSORS

    try:
        if file_type == FileType.SAFETENSORS:
            return _run_safetensors_command(args, convert_opts, file_type, command, filename)
        else:
            return _run_gguf_command(args, convert_opts, file_type, command, filename)
    except ValueError as e:
        if str(e) == "UnknownArchitecture":
            print("ERROR: Architecture not recognized. Pass --allow-unknown-arch (-u) "
                  "to convert anyway. Results may be suboptimal.", file=sys.stderr)
            return 1
        if str(e) == "UpscalingNotAllowed":
            return 1
        raise
    except NotImplementedError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def _run_safetensors_command(args, convert_opts: ConvertOptions, file_type, command, filename):
    f = Safetensors(filename)
    try:
        if command == "header":
            f.print_header()
        elif command == "tree":
            f.print_tensor_tree()
        elif command == "metadata":
            f.print_metadata()
        elif command == "convert":
            if args.calculate_size:
                report_predicted_size(f, convert_opts)
            else:
                conv.convert(f, convert_opts)
        elif command == "template":
            out_path = f"{args.output_name}.json" if args.output_name else "template.json"
            arch_ptr = arch_mod.detect_arch_from_tensors(f.tensors)
            with open(out_path, "w", encoding="utf-8") as w:
                conv.write_template_from_file(f, arch_ptr, True, w)
            print(f"Template exported to {out_path}")
        elif command == "sensitivities":
            out_path = f"{args.output_name}.json" if args.output_name else "sensitivities.json"
            arch_ptr = arch_mod.detect_arch_from_tensors(f.tensors)
            threshold = arch_ptr.threshhold if (arch_ptr and arch_ptr.threshhold is not None) else conv.QUANTIZATION_THRESHOLD
            with open(out_path, "w", encoding="utf-8") as w:
                conv.generate_sensitivities_from_tensors(f.tensors, arch_ptr, threshold, w)
            print(f"Sensitivities exported to {out_path}")
        elif command == "names":
            dump_names(f.tensors, args.shapes)
        else:
            raise ValueError(f"Unknown command: {command}")
    finally:
        f.close()
    return 0


def _run_gguf_command(args, convert_opts: ConvertOptions, file_type, command, filename):
    f = Gguf(filename)
    try:
        print(f"GGUF format version {f.version}")
        if command == "header":
            f.read_gguf_tensor_header()
        elif command == "tree":
            raise NotImplementedError("tree is not implemented for GGUF files")
        elif command == "metadata":
            f.read_gguf_metadata()
        elif command == "convert":
            if args.calculate_size:
                report_predicted_size(f, convert_opts)
            else:
                conv.convert(f, convert_opts)
        elif command == "names":
            dump_names(f.tensors, args.shapes)
        elif command == "template":
            out_path = f"{args.output_name}.json" if args.output_name else "template.json"
            with open(out_path, "w", encoding="utf-8") as w:
                f.write_template(w)
            print(f"Template exported to {out_path}")
        elif command == "sensitivities":
            out_path = f"{args.output_name}.json" if args.output_name else "sensitivities.json"
            arch_ptr = arch_mod.detect_arch_from_tensors(f.tensors)
            threshold = arch_ptr.threshhold if (arch_ptr and arch_ptr.threshhold is not None) else conv.QUANTIZATION_THRESHOLD
            with open(out_path, "w", encoding="utf-8") as w:
                conv.generate_sensitivities_from_tensors(f.tensors, arch_ptr, threshold, w)
            print(f"Sensitivities exported to {out_path}")
        else:
            raise ValueError(f"Unknown command: {command}")
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
