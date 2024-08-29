import argparse
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    Code,
    OpenMP,
    Cxx,
    extract_directive_signature,
    extract_directive_code,
    generate_directive_function,
    extract_directive_data,
    extract_preprocessor,
)


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arraysize", help="Size of the arrays", type=int, default=2**25
    )
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()


arguments = command_line()
size = np.int32(arguments.arraysize)
scalar = 0.4
real_type = "double"
real_bytes = 8
if arguments.float:
    real_type = "float"
    real_bytes = 4

with open("OMPStream.cpp") as file:
    source = file.read()
user_dimensions = {"array_size": size}
compiler_options = [
    "-fast",
    "-mp=gpu",
    "-I.",
    "-I..",
]

app = Code(OpenMP(), Cxx())
preprocessor = extract_preprocessor(source)
preprocessor.append(f"#define T {real_type}\n")
preprocessor.append("#define OMP_TARGET_GPU\n")
if arguments.float:
    preprocessor.append(f"#define scalar {scalar}f\n")
else:
    preprocessor.append(f"#define scalar {scalar}\n")
signatures = extract_directive_signature(source, app)
functions = extract_directive_code(source, app)
data = extract_directive_data(source, app)

# Copy
print("Tuning copy")
code = generate_directive_function(
    preprocessor,
    signatures["copy"],
    functions["copy"],
    app,
    data=data["copy"],
    user_dimensions=user_dimensions,
)
if arguments.float:
    a = np.random.randn(size).astype(np.float32)
    c = np.zeros(size).astype(np.float32)
else:
    a = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)
args = [a, c]
answer = [None, a]

tune_params = dict()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
tune_params["slength"] = [2**i for i in range(0, 5)]
metrics = dict()
metrics["GB/s"] = lambda p: (2 * real_bytes * size / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "copy",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    metrics=metrics,
    compiler_options=compiler_options,
    compiler="nvc++",
)

# Mul
print("Tuning mul")
code = generate_directive_function(
    preprocessor,
    signatures["mul"],
    functions["mul"],
    app,
    data=data["mul"],
    user_dimensions=user_dimensions,
)
if arguments.float:
    c = np.random.randn(size).astype(np.float32)
    b = np.zeros(size).astype(np.float32)
else:
    c = np.random.randn(size).astype(np.float64)
    b = np.zeros(size).astype(np.float64)
args = [b, c]
answer = [c * scalar, None]

tune_params.clear()
tune_params["nthreads"] = [32 * i for i in range(1, 33)]
tune_params["slength"] = [2**i for i in range(0, 5)]
metrics.clear()
metrics["GFLOP/s"] = lambda p: (size / 10**9) / (p["time"] / 10**3)
metrics["GB/s"] = lambda p: (2 * real_bytes * size / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "mul",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    metrics=metrics,
    compiler_options=compiler_options,
    compiler="nvc++",
)

# Add
print("Tuning add")
code = generate_directive_function(
    preprocessor,
    signatures["add"],
    functions["add"],
    app,
    data=data["add"],
    user_dimensions=user_dimensions,
)
if arguments.float:
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros(size).astype(np.float32)
else:
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)
args = [a, b, c]
answer = [None, None, a + b]

tune_params.clear()
tune_params["vlength"] = [32 * i for i in range(1, 33)]
metrics.clear()
metrics["GFLOP/s"] = lambda p: (size / 10**9) / (p["time"] / 10**3)
metrics["GB/s"] = lambda p: (3 * real_bytes * size / 10**9) / (p["time"] / 10**3)


tune_kernel(
    "add",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    metrics=metrics,
    compiler_options=compiler_options,
    compiler="nvc++",
)

# Triad
print("Tuning triad")
code = generate_directive_function(
    preprocessor,
    signatures["triad"],
    functions["triad"],
    app,
    data=data["triad"],
    user_dimensions=user_dimensions,
)
if arguments.float:
    a = np.zeros(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.random.randn(size).astype(np.float32)
else:
    a = np.zeros(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
    c = np.random.randn(size).astype(np.float64)
args = [a, b, c]
answer = [b + (scalar * c), None, None]

tune_params.clear()
tune_params["vlength"] = [32 * i for i in range(1, 33)]
metrics.clear()
metrics["GFLOP/s"] = lambda p: (2 * size / 10**9) / (p["time"] / 10**3)
metrics["GB/s"] = lambda p: (3 * real_bytes * size / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "triad",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    metrics=metrics,
    compiler_options=compiler_options,
    compiler="nvc++",
)

# Dot
print("Tuning dot")
code = generate_directive_function(
    preprocessor,
    signatures["dot"],
    functions["dot"],
    app,
    data=data["dot"],
    user_dimensions=user_dimensions,
)
if arguments.float:
    dotsum = np.float32(0)
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
else:
    dotsum = np.float64(0)
    a = np.random.randn(size).astype(np.float64)
    b = np.random.randn(size).astype(np.float64)
args = [dotsum, a, b]

tune_params.clear()
tune_params["vlength"] = [32 * i for i in range(1, 33)]
metrics.clear()
metrics["GFLOP/s"] = lambda p: (2 * size / 10**9) / (p["time"] / 10**3)
metrics["GB/s"] = lambda p: (2 * real_bytes * size / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "dot",
    code,
    0,
    args,
    tune_params,
    metrics=metrics,
    compiler_options=compiler_options,
    compiler="nvc++",
)