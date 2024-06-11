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
    allocate_signature_memory,
)


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arraysize", help="Size of the arrays", type=int, default=2**25)
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()


arguments = command_line()
size = np.int32(arguments.arraysize)
real_type = "double"
if arguments.float:
    real_type = "float"

with open("ACCStream.cpp") as file:
    source = file.read()
preprocessor = [f"#define T {real_type}\n"]
user_dimensions = {"array_size": size}

app = Code(OpenACC(), Cxx())
signatures = extract_directive_signature(source, app)
functions = extract_directive_code(source, app)
data = extract_directive_data(source, app)

print("Tuning copy")
code = generate_directive_function(
    preprocessor,
    signatures["copy"],
    functions["copy"],
    app,
    data=data["copy"],
    user_dimensions=user_dimensions
)
a = np.random.randn(size).astype(np.float64)
c = np.zeros(size).astype(np.float64)
args = [a, c, size]
answer = [None, a, None]

tune_params = dict()
tune_params["vlength"] = [32*i for i in range(1, 33)]

tune_kernel(
    "copy",
    code,
    0,
    args,
    tune_params,
    answer=answer,
    compiler_options=["-fast", "-acc=gpu"],
    compiler="nvfortran",
)
