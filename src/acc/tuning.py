import argparse
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    Code,
    OpenACC,
    Cxx,
    extract_directive_signature,
    extract_directive_code,
    generate_directive_function,
    extract_directive_data,
    extract_preprocessor
)


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", help="Size of the arrays", type=int, default=2**25)
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()


arguments = command_line()
size = np.int32(arguments.arraysize)
real_type = "double"
real_bytes = 8
if arguments.float:
    real_type = "float"
    real_bytes = 4

with open("ACCStream.cpp") as file:
    source = file.read()
user_dimensions = {"array_size": size}
compiler_options = [
    "-fast",
    "-acc=gpu",
    "-I.",
    "-I..",
]
metrics = dict()

app = Code(OpenACC(), Cxx())
preprocessor = extract_preprocessor(source)
preprocessor.append(f"#define T {real_type}\n")
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
    user_dimensions=user_dimensions
)
if arguments.float:
    a = np.random.randn(size).astype(np.float32)
    c = np.zeros(size).astype(np.float32)
else:
    a = np.random.randn(size).astype(np.float64)
    c = np.zeros(size).astype(np.float64)
args = [a, c, size]
answer = [None, a, None]

tune_params = dict()
tune_params["vlength"] = [32*i for i in range(1, 33)]
metrics["GB/s"] = lambda p: (2 * real_bytes * size / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "copy",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    compiler_options=compiler_options,
    compiler="nvfortran",
)
