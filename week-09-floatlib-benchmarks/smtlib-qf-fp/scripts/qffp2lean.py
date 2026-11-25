#!/usr/bin/env python3
"""qffp2lean.py: SMT-LIB QF_FP instances with fixed inputs, as Lean statements about FloatLib's IEEE operations.

An instance is *in reach* when every declared constant that its assertions use is fixed by the instance itself:
some assertion is an equation `(= c t)` or `(= t c)` whose other side `t` mentions only constants that are already
fixed (a literal in the simplest case). Those equations are the stated model. The other assertions are the
constraint, and the Lean statement is the constraint evaluated at the model:

* status `sat`:   the constraint is `true` at the model (so the model satisfies every assertion);
* status `unsat`: the constraint is `false` at the model (every model must agree with the fixed inputs, so there
  is none).

Where SMT-LIB leaves a function open (`fp.min` and `fp.max` on zeros of opposite signs), a model also chooses the
function's value there. The statement then says that some choice makes the constraint true (`sat`) or that every
choice makes it false (`unsat`).

The translation of a term depends only on its syntax and sorts, never on the benchmark's name, its expected
result, or the value of an operand.

Usage:
  qffp2lean.py classify ROOT OUT.jsonl            one JSON line per instance under ROOT (fixed inputs or not)
  qffp2lean.py emit ROOT MANIFEST.jsonl OUTDIR --batch N [--only-family F]
                                                  Lean batch files for the instances in reach
"""
import argparse
import json
import os
import sys

FP_ALIASES = {'Float16': (5, 11), 'Float32': (8, 24), 'Float64': (11, 53), 'Float128': (15, 113)}
ROUNDING = {
    'roundNearestTiesToEven': '.nearest', 'RNE': '.nearest',
    'roundNearestTiesToAway': '.nearestAway', 'RNA': '.nearestAway',
    'roundTowardPositive': '.up', 'RTP': '.up',
    'roundTowardNegative': '.down', 'RTN': '.down',
    'roundTowardZero': '.towardZero', 'RTZ': '.towardZero',
}
BOOL, RM = ('bool',), ('rm',)


class Unsupported(Exception):
    """The instance uses something this translator does not translate."""


class Malformed(Exception):
    """The file is not the SMT-LIB this translator expects."""


# ---------------------------------------------------------------------------------------------------------------
# S-expressions


class Str(str):
    """A string literal, as opposed to a symbol."""


def tokenize(text):
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
        elif c == ';':
            while i < n and text[i] != '\n':
                i += 1
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while True:
                if j >= n:
                    raise Malformed('unterminated string')
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            tokens.append(Str(text[i + 1:j]))
            i = j + 1
        elif c == '|':
            j = text.find('|', i + 1)
            if j < 0:
                raise Malformed('unterminated quoted symbol')
            tokens.append(text[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()";|':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def parse(text):
    """The list of top-level S-expressions of `text`: nested lists of symbols (`str`) and strings (`Str`)."""
    stack, top = [], []
    current = top
    for token in tokenize(text):
        if token == '(' and not isinstance(token, Str):
            stack.append(current)
            new = []
            current.append(new)
            current = new
        elif token == ')' and not isinstance(token, Str):
            if not stack:
                raise Malformed('unbalanced )')
            current = stack.pop()
        else:
            current.append(token)
    if stack:
        raise Malformed('unbalanced (')
    return top


# ---------------------------------------------------------------------------------------------------------------
# Instances


class Instance:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.logic = None
        self.sorts = {}      # alias -> sort S-expression
        self.decls = {}      # constant -> sort S-expression, in declaration order
        self.macros = {}     # defined function -> (parameter names, parameter sorts, result sort, body)
        self.asserts = []
        self.check_sats = 0
        with open(path, encoding='utf-8', errors='strict') as handle:
            commands = parse(handle.read())
        for command in commands:
            if not isinstance(command, list) or not command:
                raise Malformed('a top-level atom')
            head = command[0]
            if head == 'set-info':
                if len(command) >= 3 and command[1] == ':status':
                    self.status = str(command[2])
            elif head == 'set-logic':
                self.logic = command[1]
            elif head in ('set-option', 'exit', 'get-model', 'get-value', 'get-info'):
                pass
            elif head == 'define-sort':
                if command[2]:
                    raise Unsupported('a sort with parameters')
                self.sorts[command[1]] = command[3]
            elif head == 'declare-fun':
                if command[2]:
                    raise Unsupported('an uninterpreted function with arguments')
                self.decls[command[1]] = command[3]
            elif head == 'declare-const':
                self.decls[command[1]] = command[2]
            elif head == 'define-fun':
                params = command[2]
                self.macros[command[1]] = ([p[0] for p in params], [p[1] for p in params], command[3], command[4])
            elif head == 'assert':
                self.asserts.append(command[1])
            elif head == 'check-sat':
                self.check_sats += 1
            else:
                raise Unsupported(f'the command {head}')
        if self.check_sats != 1:
            raise Unsupported(f'{self.check_sats} check-sat commands')
        # The declared constants below each defined function, computed once in definition order.
        self.macro_free = {}
        for name, (params, _, _, body) in self.macros.items():
            self.macro_free[name] = self.free(body, frozenset(params))

    # -- which declared constants a term uses ------------------------------------------------------------------

    def free(self, term, bound=frozenset()):
        """The declared constants that occur in `term`, through the bodies of the defined functions it calls."""
        found = set()
        stack = [(term, bound)]
        while stack:
            term, bound = stack.pop()
            if isinstance(term, Str):
                continue
            if isinstance(term, str):
                if term in bound:
                    continue
                if term in self.macros:
                    # A function is defined before it is used, so its set is known here.
                    found |= self.macro_free.get(term, set())
                elif term in self.decls:
                    found.add(term)
                continue
            if not term:
                continue
            head = term[0]
            if head == 'let':
                inner = set(bound)
                for name, value in term[1]:
                    stack.append((value, bound))
                    inner.add(name)
                stack.append((term[2], frozenset(inner)))
            elif head == '_':
                continue
            elif head == '!':
                stack.append((term[1], bound))
            else:
                for part in term:
                    stack.append((part, bound))
        return found

    def fix_inputs(self):
        """Split the assertions into the equations that fix the constants, in order, and the constraint.

        Returns `(fixed, constraint, unfixed)`: `fixed` is a list of `(constant, term)`, `constraint` the list
        of the other assertions, and `unfixed` the constants that the assertions use and no equation fixes.
        """
        used = set()
        for assertion in self.asserts:
            used |= self.free(assertion)
        fixed, names = [], set()
        remaining = list(self.asserts)
        progress = True
        while progress:
            progress = False
            for assertion in remaining:
                if not (isinstance(assertion, list) and len(assertion) == 3 and assertion[0] == '='):
                    continue
                for constant, other in ((assertion[1], assertion[2]), (assertion[2], assertion[1])):
                    if (isinstance(constant, str) and not isinstance(constant, Str) and constant in self.decls
                            and constant not in names and self.free(other) <= names):
                        fixed.append((constant, other))
                        names.add(constant)
                        remaining = [a for a in remaining if a is not assertion]
                        progress = True
                        break
                if progress:
                    break
        return fixed, remaining, sorted(used - names)


# ---------------------------------------------------------------------------------------------------------------
# Typed translation to Lean


class Val:
    """A translated term: its Lean text, its sort, and for a bit-vector literal its value."""

    def __init__(self, lean, sort, bits=None):
        self.lean, self.sort, self.bits = lean, sort, bits


def type_name(sort):
    return f'F{sort[1]}_{sort[2]}'


class Translator:
    def __init__(self, instance, ieee_minmax=False):
        self.instance = instance
        # A variant for comparison: read fp.min and fp.max as IEEE 754-2019 minimumNumber and maximumNumber
        # everywhere, also on zeros of opposite signs, where SMT-LIB leaves them open.
        self.ieee_minmax = ieee_minmax
        self.formats = set()     # floating-point sorts used
        self.helpers = set()     # prelude definitions to unfold, as (format, name)
        self.choices = []        # the Boolean choices of the open cases of fp.min and fp.max, in order

    # -- sorts --------------------------------------------------------------------------------------------------

    def sort(self, s):
        if isinstance(s, str):
            if s in self.instance.sorts:
                return self.sort(self.instance.sorts[s])
            if s in FP_ALIASES:
                return ('fp',) + FP_ALIASES[s]
            if s == 'Bool':
                return BOOL
            if s == 'RoundingMode':
                return RM
            raise Unsupported(f'the sort {s}')
        if len(s) == 4 and s[0] == '_' and s[1] == 'FloatingPoint':
            return ('fp', int(s[2]), int(s[3]))
        if len(s) == 3 and s[0] == '_' and s[1] == 'BitVec':
            return ('bv', int(s[2]))
        raise Unsupported(f'the sort {s}')

    # -- floating-point values ----------------------------------------------------------------------------------

    def fp_literal(self, sort, bits):
        _, eb, sb = sort
        self.formats.add(sort)
        width = eb + sb
        digits = (width + 3) // 4
        return Val(f'(ExecFloat.Binary.ofNatBits 0x{bits:0{digits}x} : {type_name(sort)})', sort, bits)

    def helper(self, sort, name):
        self.formats.add(sort)
        self.helpers.add((sort, name))
        return f'{type_name(sort)}.{name}'

    def choice(self, sort, name):
        variable = f'{name}{sort[1]}_{sort[2]}'
        if variable not in self.choices:
            self.choices.append(variable)
        return variable

    # -- terms --------------------------------------------------------------------------------------------------

    def term(self, t, env):
        if isinstance(t, Str):
            raise Unsupported('a string literal')
        if isinstance(t, str):
            return self.atom(t, env)
        if not t:
            raise Malformed('an empty application')
        head = t[0]
        if head == '!':
            return self.term(t[1], env)
        if head == 'let':
            inner = dict(env)
            for name, value in t[1]:
                inner[name] = self.term(value, env)
            return self.term(t[2], inner)
        if head == '_':
            return self.indexed_constant(t)
        if head == 'fp':
            sign, exponent, fraction = (self.term(a, env) for a in t[1:])
            for part in (sign, exponent, fraction):
                if part.sort[0] != 'bv' or part.bits is None:
                    raise Unsupported('fp applied to a bit-vector that is not a literal')
            if sign.sort[1] != 1:
                raise Malformed('fp with a sign wider than one bit')
            eb, fb = exponent.sort[1], fraction.sort[1]
            return self.fp_literal(('fp', eb, fb + 1), (sign.bits << (eb + fb)) | (exponent.bits << fb) | fraction.bits)
        if isinstance(head, list):
            return self.indexed_application(head, [self.term(a, env) for a in t[1:]])
        if head in env or (head in self.instance.macros and self.instance.macros[head][0]):
            if head in env:
                raise Unsupported('a bound name applied to arguments')
            params, sorts, result, body = self.instance.macros[head]
            arguments = [self.term(a, env) for a in t[1:]]
            if len(arguments) != len(params):
                raise Malformed(f'{head} applied to {len(arguments)} arguments')
            for argument, s in zip(arguments, sorts):
                self.expect(argument, self.sort(s), head)
            value = self.term(body, dict(zip(params, arguments)))
            self.expect(value, self.sort(result), head)
            return value
        return self.application(head, [self.term(a, env) for a in t[1:]])

    def atom(self, a, env):
        if a in env:
            return env[a]
        if a in self.instance.macros:
            params, _, result, body = self.instance.macros[a]
            if params:
                raise Malformed(f'{a} without arguments')
            value = self.term(body, {})
            self.expect(value, self.sort(result), a)
            return value
        if a in self.instance.decls:
            raise Unsupported(f'the constant {a} is not fixed')
        if a == 'true':
            return Val('true', BOOL)
        if a == 'false':
            return Val('false', BOOL)
        if a in ROUNDING:
            return Val(ROUNDING[a], RM)
        if a.startswith('#b'):
            return Val(f'0b{a[2:]}', ('bv', len(a) - 2), int(a[2:], 2))
        if a.startswith('#x'):
            return Val(f'0x{a[2:]}', ('bv', 4 * (len(a) - 2)), int(a[2:], 16))
        raise Unsupported(f'the symbol {a}')

    def indexed_constant(self, t):
        kind = t[1]
        if kind in ('+oo', '-oo', '+zero', '-zero', 'NaN'):
            eb, sb = int(t[2]), int(t[3])
            fb = sb - 1
            top = ((1 << eb) - 1) << fb
            sign = 1 << (eb + fb)
            # The NaN of SMT-LIB is written here as the quiet NaN with an empty payload.
            bits = {'+oo': top, '-oo': sign | top, '+zero': 0, '-zero': sign, 'NaN': top | (1 << (fb - 1))}[kind]
            return self.fp_literal(('fp', eb, sb), bits)
        if kind.startswith('bv') and kind[2:].isdigit():
            width, value = int(t[2]), int(kind[2:])
            return Val(str(value), ('bv', width), value)
        raise Unsupported(f'the indexed constant {kind}')

    def indexed_application(self, head, arguments):
        if head[0] == '_' and head[1] == 'to_fp' and len(arguments) == 1 and arguments[0].sort[0] == 'bv':
            eb, sb = int(head[2]), int(head[3])
            argument = arguments[0]
            if argument.sort[1] != eb + sb:
                raise Malformed('to_fp of a bit-vector of another width')
            if argument.bits is None:
                raise Unsupported('to_fp of a bit-vector that is not a literal')
            return self.fp_literal(('fp', eb, sb), argument.bits)
        raise Unsupported(f'the indexed function {" ".join(str(x) for x in head[1:2])}')

    def expect(self, value, sort, where):
        if value.sort != sort:
            raise Malformed(f'{where}: sort {value.sort} where {sort} is expected')

    def same_fp(self, arguments, where):
        sort = arguments[0].sort
        if sort[0] != 'fp':
            raise Malformed(f'{where} applied to sort {sort}')
        for argument in arguments:
            self.expect(argument, sort, where)
        return sort

    def chain(self, arguments, relation):
        """SMT-LIB's chainable relations: every neighbouring pair."""
        if len(arguments) < 2:
            raise Malformed('a relation with fewer than two arguments')
        parts = [relation(a, b) for a, b in zip(arguments, arguments[1:])]
        return parts[0] if len(parts) == 1 else '(' + ' && '.join(parts) + ')'

    def equal(self, a, b):
        self.expect(b, a.sort, '=')
        if a.sort[0] == 'fp':
            return f'({self.helper(a.sort, "same")} {a.lean} {b.lean})'
        if a.sort == BOOL:
            return f'({a.lean} == {b.lean})'
        if a.sort[0] == 'bv':
            return f'(Nat.beq {a.lean} {b.lean})'
        if a.sort == RM:
            return f'(decide (({a.lean} : Model.IEEERoundingMode) = {b.lean}))'
        raise Unsupported(f'= at sort {a.sort}')

    def application(self, head, arguments):
        binary_rounded = {'fp.add': 'add', 'fp.sub': 'sub', 'fp.mul': 'mul', 'fp.div': 'div'}
        comparisons = {'fp.eq': 'compareEqual', 'fp.lt': 'compareLess', 'fp.leq': 'compareLessEqual',
                       'fp.gt': 'compareGreater', 'fp.geq': 'compareGreaterEqual'}
        classes = {'fp.isNaN': 'isNaN', 'fp.isInfinite': 'isInfinite', 'fp.isZero': 'isZero',
                   'fp.isSubnormal': 'isSubnormal', 'fp.isNormal': 'isNormal'}
        if head in binary_rounded:
            mode, x, y = arguments
            self.expect(mode, RM, head)
            sort = self.same_fp([x, y], head)
            return Val(f'(ExecFloat.Binary.{binary_rounded[head]} {x.lean} {y.lean} (mode := {mode.lean}))', sort)
        if head == 'fp.fma':
            mode, x, y, z = arguments
            self.expect(mode, RM, head)
            sort = self.same_fp([x, y, z], head)
            return Val(f'(ExecFloat.Binary.fma {x.lean} {y.lean} {z.lean} (mode := {mode.lean}))', sort)
        if head == 'fp.sqrt':
            mode, x = arguments
            self.expect(mode, RM, head)
            sort = self.same_fp([x], head)
            return Val(f'(ExecFloat.Binary.sqrt {x.lean} (mode := {mode.lean}))', sort)
        if head == 'fp.roundToIntegral':
            mode, x = arguments
            self.expect(mode, RM, head)
            sort = self.same_fp([x], head)
            return Val(f'(ExecFloat.Binary.roundToIntegral {x.lean} {mode.lean})', sort)
        if head == 'fp.rem':
            x, y = arguments
            sort = self.same_fp([x, y], head)
            return Val(f'(ExecFloat.Binary.remainder {x.lean} {y.lean})', sort)
        if head == 'fp.abs':
            (x,) = arguments
            sort = self.same_fp([x], head)
            return Val(f'(ExecFloat.Binary.abs {x.lean})', sort)
        if head == 'fp.neg':
            (x,) = arguments
            sort = self.same_fp([x], head)
            return Val(f'(-{x.lean})', sort)
        if head in ('fp.min', 'fp.max'):
            x, y = arguments
            sort = self.same_fp([x, y], head)
            name = head[3:]
            if self.ieee_minmax:
                return Val(f'(ExecFloat.Binary.{name}imumNumber {x.lean} {y.lean})', sort)
            self.helper(sort, 'oppositeZeros')
            function = self.helper(sort, name)
            first, second = self.choice(sort, name + 'PosNeg'), self.choice(sort, name + 'NegPos')
            return Val(f'({function} {first} {second} {x.lean} {y.lean})', sort)
        if head in comparisons:
            self.same_fp(arguments, head)
            return Val(self.chain(arguments, lambda a, b: f'(ExecFloat.{comparisons[head]} {a.lean} {b.lean})'), BOOL)
        if head in classes:
            (x,) = arguments
            self.same_fp([x], head)
            return Val(f'(ExecFloat.Binary.{classes[head]} {x.lean})', BOOL)
        if head in ('fp.isNegative', 'fp.isPositive'):
            (x,) = arguments
            sort = self.same_fp([x], head)
            return Val(f'({self.helper(sort, head[3:])} {x.lean})', BOOL)
        if head == '=':
            return Val(self.chain(arguments, self.equal), BOOL)
        if head == 'distinct':
            pairs = [f'(!{self.equal(a, b)})' for i, a in enumerate(arguments) for b in arguments[i + 1:]]
            return Val(pairs[0] if len(pairs) == 1 else '(' + ' && '.join(pairs) + ')', BOOL)
        if head == 'not':
            (a,) = arguments
            self.expect(a, BOOL, head)
            return Val(f'(!{a.lean})', BOOL)
        if head in ('and', 'or'):
            for a in arguments:
                self.expect(a, BOOL, head)
            if len(arguments) == 1:
                return arguments[0]
            return Val('(' + (' && ' if head == 'and' else ' || ').join(a.lean for a in arguments) + ')', BOOL)
        if head == '=>':
            for a in arguments:
                self.expect(a, BOOL, head)
            lean = arguments[-1].lean
            for a in reversed(arguments[:-1]):
                lean = f'(!{a.lean} || {lean})'
            return Val(lean, BOOL)
        if head == 'xor':
            for a in arguments:
                self.expect(a, BOOL, head)
            lean = arguments[0].lean
            for a in arguments[1:]:
                lean = f'(Bool.xor {lean} {a.lean})'
            return Val(lean, BOOL)
        if head == 'ite':
            condition, a, b = arguments
            self.expect(condition, BOOL, head)
            self.expect(b, a.sort, head)
            return Val(f'(bif {condition.lean} then {a.lean} else {b.lean})', a.sort)
        raise Unsupported(f'the function {head}')

    # -- the statement ------------------------------------------------------------------------------------------

    def statement(self, opposite=False, mutate=False):
        """The Lean statements of the instance and the prelude definitions to unfold.

        Returns `(claims, unfold)`. `claims` maps a status to the statement that proves it: the instance's own
        status, or both `sat` and `unsat` when the benchmark says `unknown` (one of the two is true). With
        `opposite`, the claim is the one of the other status, which is false when the benchmark is right: a
        control that the statements are not trivially true. With `mutate`, the last constant that the instance
        fixes to a floating-point literal gets that literal with its lowest bit flipped (a second control: the
        claim must then be false, unless the literal is a NaN, which the flip leaves a NaN).
        """
        instance = self.instance
        if instance.status not in ('sat', 'unsat', 'unknown'):
            raise Unsupported(f'status {instance.status}')
        fixed, constraint, unfixed = instance.fix_inputs()
        if unfixed:
            raise Unsupported('inputs that are not fixed: ' + ' '.join(unfixed))
        env = {}
        literal = [position for position, (_, defining) in enumerate(fixed)
                   if isinstance(defining, list) and defining and defining[0] in ('fp', '_')]
        for position, (constant, defining) in enumerate(fixed):
            value = self.term(defining, env)
            self.expect(value, self.sort(instance.decls[constant]), constant)
            if mutate and literal and position == literal[-1]:
                if value.sort[0] != 'fp' or value.bits is None:
                    raise Unsupported('nothing to mutate')
                value = self.fp_literal(value.sort, value.bits ^ 1)
            env[constant] = value
        if mutate and not literal:
            raise Unsupported('nothing to mutate')
        parts = []
        for assertion in constraint:
            value = self.term(assertion, env)
            self.expect(value, BOOL, 'assert')
            parts.append(value.lean)
        body = 'true' if not parts else parts[0] if len(parts) == 1 else '(' + ' && '.join(parts) + ')'
        claims = {}
        if instance.status == 'unknown':
            statuses = ('sat', 'unsat')
        elif opposite:
            statuses = ('unsat' if instance.status == 'sat' else 'sat',)
        else:
            statuses = (instance.status,)
        for status in statuses:
            if status == 'sat':
                quantifier, claim = '∃', f'{body} = true'
            else:
                quantifier, claim = '∀', f'{body} = false'
            if self.choices:
                claim = f'{quantifier} {" ".join(self.choices)} : Bool, {claim}'
            claims[status] = claim
        unfold = sorted(f'{type_name(sort)}.{name}' for sort, name in self.helpers)
        return claims, unfold


PRELUDE_HEAD = '''import FloatLib

/-!
SMT-LIB 2025, QF_FP: instances with fixed inputs, as statements about FloatLib's IEEE operations.
Written by `qffp2lean.py`. Do not edit.
-/

set_option Elab.async false

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

namespace QFFP

open Lean Elab Command in
/--
`#qffp_check name`: check again, in Lean's kernel, the proof of the theorem `name` and of the auxiliary lemmas
that `float_decide` made for it (the theorems of this file named `name._...`), and print one line
`QFFP|name|kernel nanoseconds|axioms|clock|parts`. The line says `FAILED` in place of the time when the kernel
rejects a proof. `clock` is the monotonic clock in nanoseconds before and after the check, and `parts` the
kernel nanoseconds of the theorem's own proof term (the rewrite of the public operations to their kernel
operations) and of its auxiliary lemmas (the decision by evaluation).
-/
elab "#qffp_check " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let env ← getEnv
  -- The theorem and its auxiliary lemmas.
  let mut pending := #[name]
  let mut own : Array Name := #[]
  let mut seen : NameSet := {}
  while !pending.isEmpty do
    let current := pending.back!
    pending := pending.pop
    if seen.contains current then continue
    seen := seen.insert current
    if (env.getModuleIdxFor? current).isSome || !name.isPrefixOf current then continue
    match env.find? current with
    | some (.thmInfo info) =>
        own := own.push current
        for used in info.value.getUsedConstants do
          pending := pending.push used
    | _ => pure ()
  let start ← IO.monoNanosNow
  let mut accepted := true
  let mut statement := 0
  let mut auxiliary := 0
  for current in own do
    let before ← IO.monoNanosNow
    match env.find? current with
    | some (.thmInfo info) =>
        match Kernel.check env {} info.value with
        | .ok type =>
            match Kernel.isDefEq env {} type info.type with
            | .ok true => pure ()
            | _ => accepted := false
        | .error _ => accepted := false
    | _ => accepted := false
    let after ← IO.monoNanosNow
    if current == name then statement := statement + (after - before)
    else auxiliary := auxiliary + (after - before)
  let stop ← IO.monoNanosNow
  let axioms ← liftCoreM <| collectAxioms name
  let time := if accepted then toString (stop - start) else "FAILED"
  let axiomNames := ",".intercalate (axioms.toList.map toString)
  IO.println s!"QFFP|{name}|{time}|{axiomNames}|{start} {stop}|{statement} {auxiliary}"

/-- `#qffp_clock`: print the monotonic clock, `QFFPCLOCK|nanoseconds`, where the theorems of the file begin. -/
elab "#qffp_clock" : command => do
  IO.println s!"QFFPCLOCK|{← IO.monoNanosNow}"

'''

PRELUDE_FORMAT = '''/-- The sort `(_ FloatingPoint {eb} {sb})`: {eb} exponent bits and {sb} significand bits, {fb} of them stored. -/
abbrev {T} := ExecFloat.Binary (exponentBits := {eb}) (fractionBits := {fb})

namespace {T}

/-- SMT-LIB's `=` at this sort: the same value. All NaNs are one value, and the two zeros are two values. -/
def same (a b : {T}) : Bool :=
  Nat.beq (ExecFloat.Binary.toNatBits a) (ExecFloat.Binary.toNatBits b) ||
    (ExecFloat.Binary.isNaN a && ExecFloat.Binary.isNaN b)

/-- Two zeros of opposite signs, where SMT-LIB leaves `fp.min` and `fp.max` open. -/
def oppositeZeros (a b : {T}) : Bool :=
  ExecFloat.Binary.isZero a && ExecFloat.Binary.isZero b &&
    Bool.xor (ExecFloat.Binary.signBit a) (ExecFloat.Binary.signBit b)

/--
SMT-LIB's `fp.min`: the other operand when one is a NaN, the lesser operand otherwise (IEEE 754-2019
`minimumNumber`), except on two zeros of opposite signs. There a model chooses the first operand or the second,
once for the operands `(+0, -0)` (`firstAtPosNeg`) and once for `(-0, +0)` (`firstAtNegPos`).
-/
def min (firstAtPosNeg firstAtNegPos : Bool) (a b : {T}) : {T} :=
  bif oppositeZeros a b then
    bif (bif ExecFloat.Binary.signBit a then firstAtNegPos else firstAtPosNeg) then a else b
  else ExecFloat.Binary.minimumNumber a b

/-- SMT-LIB's `fp.max`, as `min`: IEEE 754-2019 `maximumNumber` except on two zeros of opposite signs. -/
def max (firstAtPosNeg firstAtNegPos : Bool) (a b : {T}) : {T} :=
  bif oppositeZeros a b then
    bif (bif ExecFloat.Binary.signBit a then firstAtNegPos else firstAtPosNeg) then a else b
  else ExecFloat.Binary.maximumNumber a b

/-- SMT-LIB's `fp.isNegative`: a negative number, zero or infinity, and not a NaN. -/
def isNegative (a : {T}) : Bool := !ExecFloat.Binary.isNaN a && ExecFloat.Binary.signBit a

/-- SMT-LIB's `fp.isPositive`: a positive number, zero or infinity, and not a NaN. -/
def isPositive (a : {T}) : Bool := !ExecFloat.Binary.isNaN a && !ExecFloat.Binary.signBit a

end {T}

'''


def lean_name(relative):
    return '«' + relative.replace('«', '_').replace('»', '_') + '»'


def instance_files(root):
    for directory, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.endswith('.smt2'):
                yield os.path.join(directory, name)


def classify(root, out):
    counts = {}
    with open(out, 'w', encoding='utf-8') as handle:
        for path in instance_files(root):
            relative = os.path.relpath(path, root)
            family = relative.split(os.sep)[0]
            record = {'instance': relative, 'family': family}
            try:
                instance = Instance(path)
                record['status'] = instance.status
                fixed, constraint, unfixed = instance.fix_inputs()
                record['constants'] = len(instance.decls)
                record['fixed'] = len(fixed)
                record['unfixed'] = len(unfixed)
                record['constraint'] = len(constraint)
                if unfixed:
                    record['reach'] = 'search'
                else:
                    try:
                        translator = Translator(instance)
                        translator.statement()
                        record['reach'] = 'translated'
                        record['formats'] = sorted(f'{s[1]}/{s[2]}' for s in translator.formats)
                    except Unsupported as error:
                        record['reach'] = 'fixed-untranslated'
                        record['why'] = str(error)
            except (Unsupported, Malformed) as error:
                record['reach'] = 'unread'
                record['why'] = f'{type(error).__name__}: {error}'
            counts[(family, record['reach'])] = counts.get((family, record['reach']), 0) + 1
            handle.write(json.dumps(record, sort_keys=True) + '\n')
    for (family, reach), count in sorted(counts.items()):
        print(f'{family}\t{reach}\t{count}')


def emit(root, manifest, outdir, batch, only_family, name_prefix, listed, opposite=False, mutate=False,
         ieee_minmax=False):
    os.makedirs(outdir, exist_ok=True)
    wanted = None
    if listed is not None:
        with open(listed, encoding='utf-8') as handle:
            wanted = {line.strip() for line in handle if line.strip()}
    selected = []
    with open(manifest, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            if (record['reach'] == 'translated' and (only_family is None or record['family'] == only_family)
                    and (wanted is None or record['instance'] in wanted)):
                selected.append(record['instance'])
    index_path = os.path.join(outdir, 'index.jsonl')
    with open(index_path, 'w', encoding='utf-8') as index:
        for number in range(0, len(selected), batch):
            chunk = selected[number:number + batch]
            file_name = f'{name_prefix}{number // batch:04d}.lean'
            body, formats = [], set()
            for relative in chunk:
                instance = Instance(os.path.join(root, relative))
                translator = Translator(instance, ieee_minmax)
                claims, unfold = translator.statement(opposite, mutate)
                formats |= translator.formats
                tactic = 'float_decide' + (f' [{", ".join(unfold)}]' if unfold else '')
                for status, claim in claims.items():
                    # An instance of unknown status has two candidate theorems, and Lean accepts one.
                    suffix = '#mutated' if mutate else f'#{status}' if len(claims) > 1 or opposite else ''
                    suffix += '#ieee' if ieee_minmax else ''
                    name = lean_name(relative + suffix)
                    block = f'theorem {name} :\n    {claim} := by\n  {tactic}\n#qffp_check {name}\n'
                    body.append((relative, instance.status, status, name, block))
            text = PRELUDE_HEAD
            for sort in sorted(formats):
                text += PRELUDE_FORMAT.replace('{T}', type_name(sort)).replace('{eb}', str(sort[1])) \
                    .replace('{sb}', str(sort[2])).replace('{fb}', str(sort[2] - 1))
            text += '#qffp_clock\n\n'
            line = text.count('\n') + 1
            for relative, status, claimed, name, block in body:
                index.write(json.dumps({'instance': relative, 'status': status, 'claims': claimed,
                                        'theorem': 'QFFP.' + name, 'file': file_name, 'line': line},
                                       sort_keys=True, ensure_ascii=False) + '\n')
                text += block + '\n'
                line += block.count('\n') + 1
            text += 'end QFFP\n'
            with open(os.path.join(outdir, file_name), 'w', encoding='utf-8') as handle:
                handle.write(text)
    print(f'{len(selected)} instances in {(len(selected) + batch - 1) // batch} files under {outdir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('classify')
    c.add_argument('root')
    c.add_argument('out')
    e = commands.add_parser('emit')
    e.add_argument('root')
    e.add_argument('manifest')
    e.add_argument('outdir')
    e.add_argument('--batch', type=int, default=250)
    e.add_argument('--only-family')
    e.add_argument('--prefix', default='Batch')
    e.add_argument('--list', help='a file of instances to take, one per line; all of the manifest otherwise')
    e.add_argument('--opposite', action='store_true',
                   help='claim the other status of each instance: a control, every theorem must be rejected')
    e.add_argument('--mutate', action='store_true',
                   help='flip the lowest bit of the last fixed floating-point literal: a control')
    e.add_argument('--ieee-minmax', action='store_true',
                   help='a variant: fp.min and fp.max as IEEE 754-2019 minimumNumber and maximumNumber, with no '
                        'open choice on zeros of opposite signs')
    arguments = parser.parse_args()
    if arguments.command == 'classify':
        classify(arguments.root, arguments.out)
    else:
        emit(arguments.root, arguments.manifest, arguments.outdir, arguments.batch, arguments.only_family,
             arguments.prefix, arguments.list, arguments.opposite, arguments.mutate, arguments.ieee_minmax)


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    main()
