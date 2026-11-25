"""Readers for the three kinds of input of the replay: ONNX networks, VNNLIB properties and
counterexample files of VNN-COMP. Everything here only translates text and tensors into the
plain-text formats that the Lean programs read; no floating-point arithmetic is done on the
network in this module.

Formats written
---------------
Network (`*.net`), whitespace-separated tokens, every float32 as the 8 hex digits of its word:

    net NIN NOUT
    sub N  w*N                       y[i] = x[i] - c[i]
    mulc N  w*N                      y[i] = x[i] * c[i]
    addc N  w*N                      y[i] = x[i] + c[i]
    dense NIN NOUT HASBIAS           then NOUT rows of NIN words, then NOUT bias words if HASBIAS=1
                                     y[j] = (sum_k x[k] * W[j][k]) + b[j]
    relu                             y[i] = if x[i] < 0 then 0 else x[i]
    conv CIN H W COUT KH KW SH SW PT PL PB PR HASBIAS
                                     then COUT*CIN*KH*KW words (row major), then COUT bias words
    end

Formula (prefix): `and K f1..fK`, `or K f1..fK`, `le t t`, `ge t t`; a term is `X i`, `Y i` or
`C decimal`.
"""
from __future__ import annotations

import gzip
import re

import numpy as np
import onnx
from onnx import numpy_helper


def read_text(path) -> str:
    path = str(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return stream.read().decode('utf-8')
    with open(path, 'r', encoding='utf-8') as stream:
        return stream.read()


def read_bytes(path) -> bytes:
    path = str(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            return stream.read()
    with open(path, 'rb') as stream:
        return stream.read()


def relative(path: str, folder: str, suffix: str) -> str:
    """`onnx/a/b.onnx` or `.../benchmarks/FAMILY/onnx/a/b.onnx` -> `a/b`."""
    path = path.strip()
    marker = f'{folder}/'
    position = path.rfind('/' + marker)
    if position >= 0:
        path = path[position + 1 + len(marker):]
    else:
        while path.startswith('./'):
            path = path[2:]
        assert path.startswith(marker), path
        path = path[len(marker):]
    assert path.endswith(suffix), path
    return path[: -len(suffix)]


def find(root, folder: str, rel: str, suffix: str):
    """The file `root/folder/rel+suffix`, compressed or not; None when it does not exist."""
    for name in (f'{rel}{suffix}.gz', f'{rel}{suffix}'):
        if (root / folder / name).exists():
            return root / folder / name
    return None


def net_files(root) -> dict:
    """Network name (the path under `onnx/` with `_` for `/`) -> ONNX file, for a family folder."""
    import csv
    files = {}
    with (root / 'instances.csv').open(newline='') as stream:
        for row in csv.reader(stream):
            if row:
                rel = relative(row[0], 'onnx', '.onnx')
                files[rel.replace('/', '_')] = find(root, 'onnx', rel, '.onnx')
    return files


# ---------------------------------------------------------------------------------------------
# VNNLIB

NUMBER = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$')


def _tokens(text: str):
    text = re.sub(r';[^\n]*', '', text)
    return re.findall(r'\(|\)|[^\s()]+', text)


def _sexprs(tokens):
    """All top-level s-expressions of a token list, as nested lists."""
    position = 0

    def read():
        nonlocal position
        token = tokens[position]
        position += 1
        if token == '(':
            items = []
            while tokens[position] != ')':
                items.append(read())
            position += 1
            return items
        if token == ')':
            raise ValueError('unbalanced parenthesis')
        return token

    result = []
    while position < len(tokens):
        result.append(read())
    return result


def _term(item):
    if not isinstance(item, str):
        raise ValueError(f'unsupported term {item!r}')
    match = re.fullmatch(r'([XY])_(\d+)', item)
    if match:
        return (match.group(1), int(match.group(2)))
    if NUMBER.match(item):
        return ('C', item)
    raise ValueError(f'unsupported term {item!r}')


def _formula(item):
    if not isinstance(item, list) or not item:
        raise ValueError(f'unsupported formula {item!r}')
    head = item[0]
    if head in ('and', 'or'):
        return (head, [_formula(child) for child in item[1:]])
    if head in ('<=', '>='):
        if len(item) != 3:
            raise ValueError(f'unsupported atom {item!r}')
        return ('le' if head == '<=' else 'ge', _term(item[1]), _term(item[2]))
    raise ValueError(f'unsupported formula head {head!r}')


def parse_vnnlib(text: str):
    """Returns (number of inputs, number of outputs, formula). The formula is the conjunction of
    all assertions; a counterexample is an assignment that makes it true."""
    inputs, outputs, asserts = set(), set(), []
    for statement in _sexprs(_tokens(text)):
        if statement[0] == 'declare-const':
            name, kind = statement[1], statement[2]
            if kind != 'Real':
                raise ValueError(f'unsupported sort {kind}')
            match = re.fullmatch(r'([XY])_(\d+)', name)
            if not match:
                raise ValueError(f'unsupported constant {name}')
            (inputs if match.group(1) == 'X' else outputs).add(int(match.group(2)))
        elif statement[0] == 'assert':
            if len(statement) != 2:
                raise ValueError('assert with several arguments')
            asserts.append(_formula(statement[1]))
        else:
            raise ValueError(f'unsupported statement {statement[0]!r}')
    if inputs != set(range(len(inputs))) or outputs != set(range(len(outputs))):
        raise ValueError('variables are not numbered from zero')
    return len(inputs), len(outputs), ('and', asserts)


def formula_tokens(formula) -> list[str]:
    head = formula[0]
    if head in ('and', 'or'):
        tokens = [head, str(len(formula[1]))]
        for child in formula[1]:
            tokens += formula_tokens(child)
        return tokens
    tokens = [head]
    for term in formula[1:]:
        tokens += [term[0], str(term[1])]
    return tokens


def formula_atoms(formula):
    if formula[0] in ('and', 'or'):
        for child in formula[1]:
            yield from formula_atoms(child)
    else:
        yield formula


# ---------------------------------------------------------------------------------------------
# Counterexample files

ASSIGNMENT = re.compile(r'\(\s*([XY])_(\d+)\s+([^\s()]+)\s*\)')


def parse_counterexample(text: str):
    """Returns (X, Y) as lists of the decimal strings of the file, in index order."""
    values = {'X': {}, 'Y': {}}
    for kind, index, number in ASSIGNMENT.findall(text):
        if not NUMBER.match(number):
            raise ValueError(f'not a decimal number: {number!r}')
        if int(index) in values[kind]:
            raise ValueError(f'{kind}_{index} assigned twice')
        values[kind][int(index)] = number
    result = []
    for kind in 'XY':
        indices = sorted(values[kind])
        if indices != list(range(len(indices))):
            raise ValueError(f'{kind} indices are not 0..n-1')
        result.append([values[kind][i] for i in indices])
    leftover = ASSIGNMENT.sub('', text).replace('(', '').replace(')', '').strip()
    if leftover:
        raise ValueError(f'unparsed text in counterexample: {leftover[:60]!r}')
    return result[0], result[1]


# ---------------------------------------------------------------------------------------------
# ONNX

DRY = False        # with True no weight is written out (scripts/survey.py only asks what translates)


def _words(array) -> list[str]:
    """The binary32 words of an array as one string of 8-digit hex numbers separated by spaces."""
    flat = np.ascontiguousarray(array, dtype='<f4').reshape(-1).view('<u4')
    if DRY or flat.size == 0:
        return []
    return [flat.astype('>u4').tobytes().hex(' ', 4)]


def count_weights(data: bytes) -> int:
    """The number of stored tensor elements of an ONNX model (initializers and Constant nodes)."""
    model = onnx.load_model_from_string(data)
    total = sum(int(np.prod(init.dims)) if init.dims else 1 for init in model.graph.initializer)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attribute in node.attribute:
                if attribute.name == 'value':
                    total += int(np.prod(attribute.t.dims)) if attribute.t.dims else 1
    return total


def lower_onnx(data: bytes):
    """Lower a feed-forward ONNX graph (a chain) to the operations of the `.net` format.
    Returns (nin, nout, list of token lists, list of ONNX operator names). Raises ValueError on
    anything it does not translate."""
    model = onnx.load_model_from_string(data)
    graph = model.graph
    constants = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    real_inputs = [i for i in graph.input if i.name not in constants]
    if len(real_inputs) != 1 or len(graph.output) != 1:
        raise ValueError('expected one input and one output')
    dims = real_inputs[0].type.tensor_type.shape.dim
    shape = [d.dim_value if d.HasField('dim_value') and d.dim_value > 0 else 1 for d in dims]
    if real_inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError('input is not float32')
    shape = shape[1:] if len(shape) > 1 else shape        # drop the batch dimension
    nin = int(np.prod(shape))
    current = real_inputs[0].name
    ops: list[list[str]] = []
    names: list[str] = []
    pending_dense = None       # index in ops of a MatMul that may still take a bias

    def const(name):
        if name not in constants:
            raise ValueError(f'operand {name} is not a constant')
        return constants[name]

    def float_const(name):
        value = const(name)
        if value.dtype != np.float32:
            raise ValueError(f'constant {name} is {value.dtype}, not float32')
        return value

    def elementwise(value, size):
        """A constant broadcast against the current tensor, flattened."""
        try:
            return np.broadcast_to(value, [1] + shape if value.ndim > len(shape) else shape).reshape(-1)
        except ValueError:
            if value.size in (1, size):
                return np.broadcast_to(value.reshape(-1), (size,))
            raise ValueError('constant does not broadcast against the tensor') from None

    for node in graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        kind = node.op_type
        names.append(kind)
        if kind == 'Constant':
            constants[node.output[0]] = numpy_helper.to_array(attrs['value'])
            continue
        size = int(np.prod(shape))
        commutative = kind in ('Add', 'Mul', 'MatMul')
        if node.input[0] == current:
            other = node.input[1] if len(node.input) > 1 else None
            data_first = True
        elif commutative and len(node.input) > 1 and node.input[1] == current:
            other = node.input[0]
            data_first = False
        else:
            raise ValueError(f'{kind} does not continue the chain')
        if kind == 'Sub':
            ops.append(['sub', str(size)] + _words(elementwise(float_const(other), size)))
            pending_dense = None
        elif kind == 'Mul':
            ops.append(['mulc', str(size)] + _words(elementwise(float_const(other), size)))
            pending_dense = None
        elif kind == 'Add':
            value = float_const(other)
            if pending_dense is not None and value.size in (1, size):
                ops[pending_dense][3] = '1'
                ops[pending_dense] += _words(np.broadcast_to(value.reshape(-1), (size,)))
            else:
                ops.append(['addc', str(size)] + _words(elementwise(value, size)))
            pending_dense = None
        elif kind in ('Flatten', 'Reshape'):
            if kind == 'Reshape':
                target = [int(v) for v in const(node.input[1]).reshape(-1)][1:]
                if target.count(-1) == 1:
                    known = int(np.prod([v for v in target if v != -1]))
                    target[target.index(-1)] = size // known
                if any(v <= 0 for v in target) or int(np.prod(target)) != size:
                    raise ValueError('Reshape changes the number of elements')
                shape = target
            else:
                if attrs.get('axis', 1) != 1:
                    raise ValueError('Flatten with axis other than 1')
                shape = [size]
            # row-major data is unchanged
        elif kind in ('Dropout', 'Identity'):
            pass                                  # inference: the identity
        elif kind == 'MatMul':
            weight = float_const(other)
            if weight.ndim != 2:
                raise ValueError('MatMul shape')
            if data_first:                        # x @ W, W is [in, out]
                weight = weight.T
            if weight.shape[1] != size:           # rows are outputs
                raise ValueError('MatMul shape')
            ops.append(['dense', str(size), str(weight.shape[0]), '0'] + _words(weight))
            pending_dense = len(ops) - 1
            shape = [int(weight.shape[0])]
        elif kind == 'Gemm':
            if attrs.get('alpha', 1.0) != 1.0 or attrs.get('beta', 1.0) != 1.0 or attrs.get('transA', 0):
                raise ValueError('Gemm with alpha, beta or transA')
            weight = float_const(node.input[1])
            if not attrs.get('transB', 0):
                weight = weight.T
            if weight.ndim != 2 or weight.shape[1] != size:
                raise ValueError('Gemm shape')
            tokens = ['dense', str(size), str(weight.shape[0]), '0'] + _words(weight)
            if len(node.input) > 2:
                bias = float_const(node.input[2])
                if bias.size not in (1, weight.shape[0]):
                    raise ValueError('Gemm bias shape')
                tokens[3] = '1'
                tokens += _words(np.broadcast_to(bias.reshape(-1), (weight.shape[0],)))
            ops.append(tokens)
            pending_dense = None
            shape = [int(weight.shape[0])]
        elif kind == 'Relu':
            ops.append(['relu'])
            pending_dense = None
        elif kind == 'Conv':
            weight = float_const(node.input[1])
            if len(shape) != 3 or weight.ndim != 4 or weight.shape[1] != shape[0]:
                raise ValueError('Conv shape')
            if attrs.get('group', 1) != 1 or any(d != 1 for d in attrs.get('dilations', [1, 1])):
                raise ValueError('Conv with groups or dilations')
            if attrs.get('auto_pad', b'NOTSET') not in (b'NOTSET', 'NOTSET'):
                raise ValueError('Conv with auto_pad')
            cin, height, width = shape
            cout, _, kh, kw = weight.shape
            sh, sw = attrs.get('strides', [1, 1])
            pt, pl, pb, pr = attrs.get('pads', [0, 0, 0, 0])
            oh = (height + pt + pb - kh) // sh + 1
            ow = (width + pl + pr - kw) // sw + 1
            tokens = ['conv'] + [str(v) for v in (cin, height, width, cout, kh, kw, sh, sw, pt, pl, pb, pr)]
            tokens += ['0'] + _words(weight)
            if len(node.input) > 2:
                bias = float_const(node.input[2])
                tokens[13] = '1'
                tokens += _words(bias)
            ops.append(tokens)
            pending_dense = None
            shape = [int(cout), int(oh), int(ow)]
        else:
            raise ValueError(f'operator {kind} is not translated')
        current = node.output[0]
    if current != graph.output[0].name:
        raise ValueError('the chain does not end at the graph output')
    return nin, int(np.prod(shape)), ops, names


def write_net(path, nin: int, nout: int, ops) -> None:
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(f'net {nin} {nout}\n')
        for tokens in ops:
            stream.write(' '.join(tokens) + '\n')
        stream.write('end\n')
