import FloatLib.Floats.ExecFloat
import FloatLib.Floats.Formats.BinaryInterchange.Configured

/-!
# Exact replay of counterexamples of neural-network verifiers

`replay CASES [--from I] [--to J] [--no-real] [--no-orders]` reads blocks

    case ID
    net PATH            a network in the `.net` format of `scripts/common.py`
    x d1 .. dn          the input of the counterexample, decimal strings as the verifier wrote them
    y d1 .. dm          the output it reported (may be empty)
    spec FORMULA        the VNNLIB property in prefix form; a counterexample makes it true
    end

and prints one tab-separated line per case. The network is evaluated on `Binary32`, FloatLib's
IEEE 754 binary32 type, with FloatLib's compiled `+`, `*`, `-` and comparison (round to nearest,
ties to even), in the reference order: every dot product is accumulated from `+0` left to right,
one rounded multiplication and one rounded addition per term, and the bias is added last. The
property is then decided on exact rationals: the binary32 input and output as the rationals they
denote, the constants of the property as the decimal fractions they are.

For comparison the same network is also evaluated in exact dyadic arithmetic (no rounding at all)
and, in binary32, in three other orders of the dot product (right to left, pairwise, and with a
fused multiply-add per term).
-/

open FloatLib.Floats

namespace VnnReplay

/-- IEEE 754 binary32 as FloatLib configures it. -/
abbrev F32 := Binary32

/-- An exact dyadic number `m * 2 ^ e`. -/
structure Dy where
  m : Int
  e : Int
  deriving Inhabited

namespace Dy

def zero : Dy := ⟨0, 0⟩

def add (a b : Dy) : Dy :=
  if a.m == 0 then b
  else if b.m == 0 then a
  else if a.e ≤ b.e then ⟨a.m + b.m * (2 : Int) ^ (b.e - a.e).toNat, a.e⟩
  else ⟨a.m * (2 : Int) ^ (a.e - b.e).toNat + b.m, b.e⟩

def sub (a b : Dy) : Dy := add a ⟨-b.m, b.e⟩

def mul (a b : Dy) : Dy := ⟨a.m * b.m, a.e + b.e⟩

def relu (a : Dy) : Dy := if a.m < 0 then zero else a

/-- The value of a finite binary32 word. Infinities and NaNs are rejected before this is used. -/
def ofBits (n : Nat) : Dy :=
  let sign := n / 2 ^ 31 % 2
  let ex := n / 2 ^ 23 % 256
  let fr := n % 2 ^ 23
  let m : Int := if ex == 0 then Int.ofNat fr else Int.ofNat (fr + 2 ^ 23)
  let e : Int := if ex == 0 then -149 else Int.ofNat ex - 150
  ⟨if sign == 1 then -m else m, e⟩

def toRat (a : Dy) : Rat :=
  if a.e ≥ 0 then ((a.m * (2 : Int) ^ a.e.toNat : Int) : Rat) else mkRat a.m (2 ^ (-a.e).toNat)

end Dy

/-- The arithmetic a network is evaluated in. -/
class Scalar (α : Type) where
  zero : α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  /-- `fma a b c` is `a * b + c`. -/
  fma : α → α → α → α
  relu : α → α

instance : Scalar F32 where
  zero := 0
  add a b := a + b
  sub a b := a - b
  mul a b := a * b
  fma a b c := ExecFloat.fma a b c
  relu a := if a < 0 then 0 else a

instance : Scalar Dy where
  zero := Dy.zero
  add := Dy.add
  sub := Dy.sub
  mul := Dy.mul
  fma a b c := Dy.add (Dy.mul a b) c
  relu := Dy.relu

/-- One layer of a lowered network. -/
inductive Op (α : Type) where
  | sub (c : Array α)
  | mulc (c : Array α)
  | addc (c : Array α)
  | dense (nin nout : Nat) (rows : Array (Array α)) (bias : Option (Array α))
  | relu
  | conv (cin h w cout kh kw sh sw pt pl pb pr : Nat) (weight : Array α) (bias : Option (Array α))

def Op.map {α β : Type} (f : α → β) : Op α → Op β
  | .sub c => .sub (c.map f)
  | .mulc c => .mulc (c.map f)
  | .addc c => .addc (c.map f)
  | .dense nin nout rows bias => .dense nin nout (rows.map (·.map f)) (bias.map (·.map f))
  | .relu => .relu
  | .conv cin h w cout kh kw sh sw pt pl pb pr weight bias =>
      .conv cin h w cout kh kw sh sw pt pl pb pr (weight.map f) (bias.map (·.map f))

structure Net (α : Type) where
  nin : Nat
  nout : Nat
  ops : Array (Op α)

def Net.map {α β : Type} (f : α → β) (net : Net α) : Net β :=
  { nin := net.nin, nout := net.nout, ops := net.ops.map (·.map f) }

def hexDigit (c : Char) : Nat :=
  if '0' ≤ c ∧ c ≤ '9' then c.toNat - '0'.toNat
  else if 'a' ≤ c ∧ c ≤ 'f' then c.toNat - 'a'.toNat + 10
  else 0

def hexVal (s : String) : Nat := s.foldl (fun acc c => acc * 16 + hexDigit c) 0

/-- The whitespace-separated tokens of a text. -/
def tokens (text : String) : Array String :=
  let (acc, cur) := text.foldl (fun (state : Array String × String) c =>
    if c.isWhitespace then
      (if state.2.isEmpty then state else (state.1.push state.2, ""))
    else (state.1, state.2.push c)) (#[], "")
  if cur.isEmpty then acc else acc.push cur

def natTok (toks : Array String) (i : Nat) : Except String Nat :=
  match toks[i]? with
  | some t => match t.toNat? with
    | some n => pure n
    | none => throw s!"expected a number at token {i}: {t}"
  | none => throw s!"unexpected end of file at token {i}"

def words (toks : Array String) (i n : Nat) : Except String (Array Nat) :=
  if i + n ≤ toks.size then pure ((toks.extract i (i + n)).map hexVal)
  else throw s!"expected {n} words at token {i}"

partial def parseOps (toks : Array String) (i : Nat) (acc : Array (Op Nat)) :
    Except String (Array (Op Nat)) := do
  match toks[i]? with
  | none => throw "missing end"
  | some "end" => pure acc
  | some "relu" => parseOps toks (i + 1) (acc.push .relu)
  | some "sub" =>
      let n ← natTok toks (i + 1)
      let c ← words toks (i + 2) n
      parseOps toks (i + 2 + n) (acc.push (.sub c))
  | some "mulc" =>
      let n ← natTok toks (i + 1)
      let c ← words toks (i + 2) n
      parseOps toks (i + 2 + n) (acc.push (.mulc c))
  | some "addc" =>
      let n ← natTok toks (i + 1)
      let c ← words toks (i + 2) n
      parseOps toks (i + 2 + n) (acc.push (.addc c))
  | some "dense" =>
      let nin ← natTok toks (i + 1)
      let nout ← natTok toks (i + 2)
      let hasBias ← natTok toks (i + 3)
      let mut rows : Array (Array Nat) := Array.mkEmpty nout
      for j in [0:nout] do
        rows := rows.push (← words toks (i + 4 + j * nin) nin)
      let next := i + 4 + nout * nin
      if hasBias == 1 then
        let bias ← words toks next nout
        parseOps toks (next + nout) (acc.push (.dense nin nout rows (some bias)))
      else
        parseOps toks next (acc.push (.dense nin nout rows none))
  | some "conv" =>
      let p ← (List.range 13).mapM fun k => natTok toks (i + 1 + k)
      let g := fun k => p.getD k 0
      let count := g 3 * g 0 * g 4 * g 5
      let weight ← words toks (i + 14) count
      let next := i + 14 + count
      if g 12 == 1 then
        let bias ← words toks next (g 3)
        parseOps toks (next + g 3)
          (acc.push (.conv (g 0) (g 1) (g 2) (g 3) (g 4) (g 5) (g 6) (g 7) (g 8) (g 9) (g 10) (g 11)
            weight (some bias)))
      else
        parseOps toks next
          (acc.push (.conv (g 0) (g 1) (g 2) (g 3) (g 4) (g 5) (g 6) (g 7) (g 8) (g 9) (g 10) (g 11)
            weight none))
  | some other => throw s!"unknown operation {other}"

def parseNet (text : String) : Except String (Net Nat) := do
  let toks := tokens text
  if toks[0]? ≠ some "net" then throw "not a net file"
  let nin ← natTok toks 1
  let nout ← natTok toks 2
  let ops ← parseOps toks 3 #[]
  pure { nin, nout, ops }

/-- The order in which a dot product is accumulated. -/
inductive Order where
  | seq | rev | pair | fma
  deriving BEq, Repr

partial def pairSum {α : Type} [Scalar α] [Inhabited α] (p : Array α) (lo hi : Nat) : α :=
  if hi ≤ lo then Scalar.zero
  else if hi - lo == 1 then p[lo]!
  else
    let mid := (lo + hi) / 2
    Scalar.add (pairSum p lo mid) (pairSum p mid hi)

def dot {α : Type} [Scalar α] [Inhabited α] (order : Order) (xs ws : Array α) : α :=
  match order with
  | .seq => Id.run do
      let mut acc : α := Scalar.zero
      for i in [0:xs.size] do
        acc := Scalar.add acc (Scalar.mul xs[i]! ws[i]!)
      return acc
  | .rev => Id.run do
      let mut acc : α := Scalar.zero
      for k in [0:xs.size] do
        let i := xs.size - 1 - k
        acc := Scalar.add acc (Scalar.mul xs[i]! ws[i]!)
      return acc
  | .fma => Id.run do
      let mut acc : α := Scalar.zero
      for i in [0:xs.size] do
        acc := Scalar.fma xs[i]! ws[i]! acc
      return acc
  | .pair =>
      let products := (Array.range xs.size).map fun i => Scalar.mul xs[i]! ws[i]!
      pairSum products 0 products.size

def evalOp {α : Type} [Scalar α] [Inhabited α] (order : Order) (op : Op α) (x : Array α) : Array α :=
  match op with
  | .sub c => (Array.range x.size).map fun i => Scalar.sub x[i]! c[i]!
  | .mulc c => (Array.range x.size).map fun i => Scalar.mul x[i]! c[i]!
  | .addc c => (Array.range x.size).map fun i => Scalar.add x[i]! c[i]!
  | .relu => x.map Scalar.relu
  | .dense _ nout rows bias =>
      (Array.range nout).map fun j =>
        let s := dot order x rows[j]!
        match bias with
        | some b => Scalar.add s b[j]!
        | none => s
  | .conv cin h w cout kh kw sh sw pt pl pb pr weight bias => Id.run do
      let oh := (h + pt + pb - kh) / sh + 1
      let ow := (w + pl + pr - kw) / sw + 1
      let mut out : Array α := Array.mkEmpty (cout * oh * ow)
      for co in [0:cout] do
        for i in [0:oh] do
          for j in [0:ow] do
            let mut xs : Array α := Array.mkEmpty (cin * kh * kw)
            let mut ws : Array α := Array.mkEmpty (cin * kh * kw)
            for ci in [0:cin] do
              for r in [0:kh] do
                for c in [0:kw] do
                  let ih := i * sh + r
                  let iw := j * sw + c
                  if pt ≤ ih ∧ ih - pt < h ∧ pl ≤ iw ∧ iw - pl < w then
                    xs := xs.push x[ci * h * w + (ih - pt) * w + (iw - pl)]!
                    ws := ws.push weight[((co * cin + ci) * kh + r) * kw + c]!
            let s := dot order xs ws
            out := out.push (match bias with
              | some b => Scalar.add s b[co]!
              | none => s)
      return out

def evalNet {α : Type} [Scalar α] [Inhabited α] (order : Order) (net : Net α) (x : Array α) : Array α :=
  net.ops.foldl (fun v op => evalOp order op v) x

/-- The number written by a list of decimal digits. -/
def digitsVal (cs : List Char) : Option Nat :=
  if cs.all Char.isDigit then some (cs.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0)
  else none

/-- A decimal string as the rational it denotes. -/
def parseDec (input : String) : Option Rat :=
  let cs := (input.toList.filter fun c => !c.isWhitespace).map Char.toLower
  let (mant, ex) := cs.span (· ≠ 'e')
  let exponent : Option Int :=
    match ex with
    | [] => some 0
    | _ :: rest =>
        let (neg, ds) := match rest with
          | '-' :: t => (true, t)
          | '+' :: t => (false, t)
          | t => (false, t)
        if ds.isEmpty then none
        else (digitsVal ds).map fun v => if neg then -Int.ofNat v else Int.ofNat v
  let (negative, body) := match mant with
    | '-' :: t => (true, t)
    | '+' :: t => (false, t)
    | t => (false, t)
  let (whole, fracDot) := body.span (· ≠ '.')
  let frac := fracDot.drop 1
  if whole.isEmpty && frac.isEmpty then none
  else
    match exponent, digitsVal (whole ++ frac) with
    | some e, some digits =>
        let scale : Int := e - Int.ofNat frac.length
        let magnitude : Rat :=
          if scale ≥ 0 then ((digits * 10 ^ scale.toNat : Nat) : Rat)
          else mkRat (Int.ofNat digits) (10 ^ (-scale).toNat)
        some (if negative then -magnitude else magnitude)
    | _, _ => none

/-- The absolute value of a rational. -/
def ratAbs (q : Rat) : Rat := if q < 0 then -q else q

/-- A short decimal rendering `MANTISSAeEXP` of a rational, for the logs only. -/
def sci (q : Rat) : String :=
  if q == 0 then "0"
  else
    let a := q.num.natAbs
    let b := q.den
    let e10 : Int := ((a.log2 : Int) - (b.log2 : Int)) * 30103 / 100000
    let shift : Int := 9 - e10
    let mant := if shift ≥ 0 then a * 10 ^ shift.toNat / b else a / (b * 10 ^ (-shift).toNat)
    s!"{if q < 0 then "-" else ""}{mant}e{-shift}"

/-- A term of a property: an input, an output, or a constant with its exact value and the value of
the nearest binary32 number. -/
inductive Term where
  | x (i : Nat)
  | y (i : Nat)
  | c (exact rounded : Rat)

inductive Form where
  | and (fs : List Form)
  | or (fs : List Form)
  /-- `le a b` is `a ≤ b`. -/
  | le (a b : Term)

def f32OfDec (s : String) : Option F32 :=
  match (ExecFloat.Binary.parse s : Except _ F32) with
  | .ok v => some v
  | .error _ => none

def ratOfF32 (v : F32) : Option Rat := ExecFloat.Binary.toRat? v

def parseTerm (toks : Array String) (i : Nat) : Except String (Term × Nat) := do
  match toks[i]?, toks[i + 1]? with
  | some "X", some k => match k.toNat? with
    | some n => pure (.x n, i + 2)
    | none => throw "bad X index"
  | some "Y", some k => match k.toNat? with
    | some n => pure (.y n, i + 2)
    | none => throw "bad Y index"
  | some "C", some d =>
      match parseDec d, (f32OfDec d).bind ratOfF32 with
      | some q, some r => pure (.c q r, i + 2)
      | _, _ => throw s!"bad constant {d}"
  | _, _ => throw s!"bad term at {i}"

partial def parseForm (toks : Array String) (i : Nat) : Except String (Form × Nat) := do
  match toks[i]? with
  | some "le" =>
      let (a, j) ← parseTerm toks (i + 1)
      let (b, k) ← parseTerm toks j
      pure (.le a b, k)
  | some "ge" =>
      let (a, j) ← parseTerm toks (i + 1)
      let (b, k) ← parseTerm toks j
      pure (.le b a, k)
  | some head =>
      if head == "and" ∨ head == "or" then
        let n ← natTok toks (i + 1)
        if head == "or" ∧ n == 0 then throw "empty disjunction"
        let mut fs : Array Form := #[]
        let mut pos := i + 2
        for _ in [0:n] do
          let (f, next) ← parseForm toks pos
          fs := fs.push f
          pos := next
        pure (if head == "and" then .and fs.toList else .or fs.toList, pos)
      else throw s!"bad formula head {head}"
  | none => throw "unexpected end of formula"

/-- What a property is evaluated on. -/
structure Env where
  xs : Array Rat
  ys : Array Rat
  /-- With `false` the atoms on inputs only count as true. -/
  useIn : Bool
  /-- With `false` the atoms that mention an output count as true. -/
  useOut : Bool
  /-- Round the constants of the input atoms to binary32. -/
  roundIn : Bool

def Term.isY : Term → Bool
  | .y _ => true
  | _ => false

def Term.val (env : Env) (inputAtom : Bool) : Term → Rat
  | .x i => env.xs[i]!
  | .y i => env.ys[i]!
  | .c exact rounded => if inputAtom && env.roundIn then rounded else exact

/-- The slack of a property: `none` is plus infinity, and the property holds exactly when the slack
is not negative. An atom `a ≤ b` has slack `b - a`, a conjunction the least and a disjunction the
greatest slack of its parts. -/
partial def Form.slack (env : Env) : Form → Option Rat
  | .le a b =>
      let inputAtom := !(a.isY || b.isY)
      if inputAtom && !env.useIn then none
      else if !inputAtom && !env.useOut then none
      else some (b.val env inputAtom - a.val env inputAtom)
  | .and fs => fs.foldl (fun acc f =>
      match acc, f.slack env with
      | none, m => m
      | some a, none => some a
      | some a, some b => some (min a b)) none
  | .or fs =>
      match fs with
      | [] => some (-1)
      | f :: rest => rest.foldl (fun acc g =>
          match acc, g.slack env with
          | none, _ => none
          | some _, none => none
          | some a, some b => some (max a b)) (f.slack env)

def holds : Option Rat → Bool
  | none => true
  | some q => q ≥ 0

def showSlack : Option Rat → String
  | none => "inf"
  | some q => sci q

def hex8 (n : Nat) : String :=
  let digits := (Nat.toDigits 16 n)
  (List.replicate (8 - digits.length) '0' ++ digits).foldl (fun s c => s.push c) ""

def bitsOf (v : F32) : Nat := ExecFloat.Binary.toNatBits v

structure Loaded where
  path : String
  f32 : Net F32
  dy : Net Dy
  nonfinite : Nat

def loadNet (path : String) : IO Loaded := do
  let text ← IO.FS.readFile path
  match parseNet text with
  | .error e => throw (IO.userError s!"{path}: {e}")
  | .ok net =>
      let count := net.ops.foldl (fun acc op =>
        let bad := fun (a : Array Nat) => (a.filter fun n => n / 2 ^ 23 % 256 == 255).size
        match op with
        | .sub c | .mulc c | .addc c => acc + bad c
        | .dense _ _ rows bias => acc + rows.foldl (fun s r => s + bad r) 0 + (bias.map bad).getD 0
        | .relu => acc
        | .conv _ _ _ _ _ _ _ _ _ _ _ _ weight bias => acc + bad weight + (bias.map bad).getD 0) 0
      pure { path, f32 := net.map ExecFloat.Binary.ofNatBits, dy := net.map Dy.ofBits, nonfinite := count }

structure Options where
  first : Nat := 0
  last : Nat := 1000000000
  real : Bool := true
  orders : Bool := true

def b01 (b : Bool) : String := if b then "1" else "0"

/-- The interval that the top-level input atoms of a property give each input: `(lower, upper)`,
each `none` when the property has no such bound. Atoms under a disjunction are not read. -/
partial def Form.bounds (n : Nat) (form : Form) : Array (Option Rat × Option Rat) :=
  let rec go (acc : Array (Option Rat × Option Rat)) : Form → Array (Option Rat × Option Rat)
    | .and fs => fs.foldl go acc
    | .le (.x i) (.c q _) =>
        acc.modify i fun (lo, hi) => (lo, some (match hi with | some h => min h q | none => q))
    | .le (.c q _) (.x i) =>
        acc.modify i fun (lo, hi) => (some (match lo with | some l => max l q | none => q), hi)
    | _ => acc
  go (Array.replicate n (none, none)) form

/-- The columns of one evaluation point, in the order of `pointColumns`. -/
def evalPoint (opts : Options) (loaded : Loaded) (form : Form) (x32 : Array F32) :
    IO (Option (Array String × Array Rat)) := do
  let some xRat := x32.mapM ratOfF32 | return none
  let t0 ← IO.monoMsNow
  let y32 := evalNet .seq loaded.f32 x32
  let t1 ← IO.monoMsNow
  let some yRat := y32.mapM ratOfF32 | return none
  let slackOf := fun (ys : Array Rat) (useIn useOut roundIn : Bool) =>
    form.slack { xs := xRat, ys, useIn, useOut, roundIn }
  let full := slackOf yRat true true false
  let inExact := slackOf yRat true false false
  let inRounded := slackOf yRat true false true
  let out32 := slackOf yRat false true false
  let (outReal, tReal) ← if opts.real then do
      let t2 ← IO.monoMsNow
      let yReal := (evalNet .seq loaded.dy (x32.map fun v => Dy.ofBits (bitsOf v))).map Dy.toRat
      let s := slackOf yReal false true false
      let fullReal := slackOf yReal true true false
      let worst := (Array.range yRat.size).foldl
        (fun acc i => max acc (ratAbs (yRat[i]! - yReal[i]!))) (0 : Rat)
      let t3 ← IO.monoMsNow
      pure (#[b01 (holds s), showSlack s, b01 (holds fullReal), sci worst], t3 - t2)
    else pure (#["NA", "NA", "NA", "NA"], 0)
  let mut orders : Array String := #[]
  if opts.orders then
    for order in [Order.rev, Order.pair, Order.fma] do
      let y := evalNet order loaded.f32 x32
      match y.mapM ratOfF32 with
      | some yr =>
          let s := slackOf yr false true false
          let same := (Array.range y.size).all fun i => bitsOf y[i]! == bitsOf y32[i]!
          orders := orders ++ #[b01 (holds s), showSlack s, b01 same]
      | none => orders := orders ++ #["NA", "NA", "NA"]
  else
    orders := #["NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
  let xHex := ",".intercalate (x32.map (hex8 ∘ bitsOf)).toList
  let yHex := ",".intercalate (y32.map (hex8 ∘ bitsOf)).toList
  let columns := #[b01 (holds full), showSlack full, b01 (holds inExact), showSlack inExact,
      b01 (holds inRounded), showSlack inRounded, b01 (holds out32), showSlack out32] ++ outReal ++
    orders ++ #[toString (t1 - t0), toString tReal, xHex, yHex]
  return some (columns, yRat)

def pointColumns : List String :=
  ["full", "full_slack", "in_exact", "in_exact_slack", "in_rounded", "in_rounded_slack", "out32",
   "out32_slack", "out_real", "out_real_slack", "full_real", "max_abs_32_minus_real", "rev",
   "rev_slack", "rev_same_bits", "pair", "pair_slack", "pair_same_bits", "fma", "fma_slack",
   "fma_same_bits", "ms_binary32", "ms_exact", "x_bits", "y_bits"]

def runCase (opts : Options) (cache : IO.Ref (Option Loaded)) (id : String) (netPath : String)
    (xDec yDec : Array String) (specToks : Array String) : IO Unit := do
  let loaded ← do
    match ← cache.get with
    | some l => if l.path == netPath then pure l else
        let l ← loadNet netPath
        cache.set (some l)
        pure l
    | none =>
        let l ← loadNet netPath
        cache.set (some l)
        pure l
  let fail := fun (msg : String) => IO.println s!"{id}\terror\t{msg}"
  if xDec.size ≠ loaded.f32.nin then
    return ← fail s!"{xDec.size} inputs for a network with {loaded.f32.nin}"
  let (form, _) ← match parseForm specToks 0 with
    | .ok r => pure r
    | .error e => return ← fail s!"property: {e}"
  -- the input: each decimal rounded once to binary32, to nearest even
  let some x32 := xDec.mapM f32OfDec | return ← fail "an input is not a decimal number"
  let some xDecRat := xDec.mapM parseDec | return ← fail "an input is not a decimal number"
  let some xRat := x32.mapM ratOfF32 | return ← fail "an input is not finite"
  -- cross-checks on the reading of the decimals
  let viaRat : Array F32 := xDecRat.map fun q => (ExecFloat.Binary.ofRat q : F32)
  let parseMismatch :=
    ((Array.range x32.size).filter fun i => bitsOf x32[i]! ≠ bitsOf viaRat[i]!).size
  let inexact := ((Array.range x32.size).filter fun i => xRat[i]! ≠ xDecRat[i]!).size
  let viaDouble : Array Nat := xDec.map fun s =>
    match (ExecFloat.Binary.parse s : Except _ Binary64) with
    | .ok d => match ExecFloat.Binary.toRat? d with
      | some q => bitsOf (ExecFloat.Binary.ofRat q : F32)
      | none => 0xffffffff
    | .error _ => 0xffffffff
  let doubleDiffers :=
    ((Array.range x32.size).filter fun i => bitsOf x32[i]! ≠ viaDouble[i]!).size
  -- the reported point
  let some (reportedPoint, yRat) ← evalPoint opts loaded form x32
    | return ← fail "an output is not finite in binary32"
  let inDecimal := form.slack { xs := xDecRat, ys := yRat, useIn := true, useOut := false, roundIn := false }
  -- the reported output
  let reported ← match yDec.mapM parseDec with
    | some yRep =>
        if yRep.size == yRat.size ∧ yRep.size > 0 then
          let s := form.slack { xs := xDecRat, ys := yRep, useIn := false, useOut := true, roundIn := false }
          let worst := (Array.range yRat.size).foldl
            (fun acc i => max acc (ratAbs (yRat[i]! - yRep[i]!))) (0 : Rat)
          let norm := yRep.foldl (fun acc v => max acc (ratAbs v)) (0 : Rat)
          pure #[b01 (holds s), showSlack s, sci worst, sci norm]
        else pure #["NA", "NA", "NA", "NA"]
    | none => pure #["NA", "NA", "NA", "NA"]
  -- The reported point moved into the box. In a coordinate where the box holds a binary32
  -- number, a value below the lower bound becomes the least binary32 number not below it and a
  -- value above the upper bound the greatest one not above it. In a coordinate where the box
  -- holds no binary32 number (both bounds lie strictly between two neighbours, for example an
  -- input fixed to a decimal fraction), the value is moved into the interval between the two
  -- bounds rounded to nearest; `proj_in_exact` then stays 0 and `proj_in_rounded` is 1.
  let bounds := form.bounds x32.size
  let up := fun (q : Rat) => (ExecFloat.Binary.ofRat q (mode := .up) : F32)
  let down := fun (q : Rat) => (ExecFloat.Binary.ofRat q (mode := .down) : F32)
  let near := fun (q : Rat) => (ExecFloat.Binary.ofRat q : F32)
  let clamp := fun (i : Nat) (a b : F32) =>
    match ratOfF32 a, ratOfF32 b with
    | some ar, some br => if xRat[i]! < ar then a else if xRat[i]! > br then b else x32[i]!
    | _, _ => x32[i]!
  let projected : Array F32 := (Array.range x32.size).map fun i =>
    let v := xRat[i]!
    match bounds[i]! with
    | (some lo, some hi) =>
        match ratOfF32 (up lo), ratOfF32 (down hi) with
        | some ar, some br =>
            if ar ≤ br then clamp i (up lo) (down hi) else clamp i (near lo) (near hi)
        | _, _ => x32[i]!
    | (some lo, none) => if v < lo then up lo else x32[i]!
    | (none, some hi) => if v > hi then down hi else x32[i]!
    | (none, none) => x32[i]!
  let moved := ((Array.range x32.size).filter fun i => bitsOf projected[i]! ≠ bitsOf x32[i]!).size
  let some projRat := projected.mapM ratOfF32 | return ← fail "a projected input is not finite"
  let maxMove := (Array.range x32.size).foldl
    (fun acc i => max acc (ratAbs (projRat[i]! - xRat[i]!))) (0 : Rat)
  let projectedPoint ← if moved == 0 then pure reportedPoint else do
    let some (columns, _) ← evalPoint opts loaded form projected
      | return ← fail "an output is not finite in binary32 at the projected point"
    pure columns
  IO.println <| "\t".intercalate <|
    [id, "ok", toString x32.size, toString yRat.size, toString inexact, toString doubleDiffers,
     toString parseMismatch, toString loaded.nonfinite, b01 (holds inDecimal), showSlack inDecimal] ++
    reported.toList ++ reportedPoint.toList ++ [toString moved, sci maxMove] ++ projectedPoint.toList
  (← IO.getStdout).flush

def header : String := "\t".intercalate <|
  ["id", "status", "nin", "nout", "x_inexact", "x_double_differs", "x_parse_mismatch",
   "net_nonfinite", "in_decimal", "in_decimal_slack", "out_reported", "out_reported_slack",
   "max_abs_32_minus_reported", "reported_norm"] ++ pointColumns ++ ["moved", "max_move"] ++
  pointColumns.map ("proj_" ++ ·)

partial def parseArgs (args : List String) (opts : Options) : Options :=
  match args with
  | "--from" :: n :: rest => parseArgs rest { opts with first := n.toNat?.getD 0 }
  | "--to" :: n :: rest => parseArgs rest { opts with last := n.toNat?.getD 0 }
  | "--no-real" :: rest => parseArgs rest { opts with real := false }
  | "--no-orders" :: rest => parseArgs rest { opts with orders := false }
  | _ :: rest => parseArgs rest opts
  | [] => opts

def toksOf (line : String) : Array String := tokens line

end VnnReplay

open VnnReplay in
def main (args : List String) : IO UInt32 := do
  match args with
  | [] => IO.eprintln "usage: replay CASES [--from I] [--to J] [--no-real] [--no-orders]"; return 2
  | casesPath :: rest =>
      let opts := parseArgs rest {}
      let lines ← IO.FS.lines casesPath
      let cache ← IO.mkRef (none : Option Loaded)
      IO.println header
      let mut index := 0
      let mut id := ""
      let mut net := ""
      let mut xs : Array String := #[]
      let mut ys : Array String := #[]
      let mut spec : Array String := #[]
      for line in lines do
        let toks := toksOf line
        match toks[0]? with
        | some "case" => id := toks.getD 1 ""
        | some "net" => net := toks.getD 1 ""
        | some "x" => xs := toks.extract 1 toks.size
        | some "y" => ys := toks.extract 1 toks.size
        | some "spec" => spec := toks.extract 1 toks.size
        | some "end" =>
            if opts.first ≤ index ∧ index < opts.last then
              runCase opts cache id net xs ys spec
            index := index + 1
        | _ => pure ()
      return 0
