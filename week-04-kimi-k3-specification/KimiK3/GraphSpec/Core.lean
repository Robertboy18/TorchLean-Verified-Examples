/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program
public import TorchLean.Tensor.Pack

/-!
# Composing Kimi's forward equations

A term evaluates a shaped tensor in an environment of shaped inputs. A block
can return several tensors, such as the new recurrent state and its output.
Both use TorchLean tensor packs and ordinary tensor operations.
-/

@[expose] public section

namespace KimiK3.GraphSpec

open TorchLean

/-- A de Bruijn variable whose result shape is part of its type.

Unlike a bare `Fin Γ.length`, `Var Γ s` records that the selected entry of `Γ` has shape `s`.
Environment lookup reduces structurally without casts through `List.get`.
-/
inductive Var : (Γ : List Shape) → Shape → Type where
  /-- The first value in a nonempty environment. -/
  | head {s : Shape} {Γ : List Shape} : Var (s :: Γ) s
  /-- A variable inherited from the tail of an environment. -/
  | tail {Γ : List Shape} {s t : Shape} : Var Γ t → Var (s :: Γ) t

namespace Var

/-- Preserve a variable when one value is appended to its environment. -/
def weakenRight {Γ : List Shape} {s t : Shape} : Var Γ s → Var (Γ ++ [t]) s
  | .head => .head
  | .tail v => .tail (weakenRight v)

/-- Embed a variable from the left side of an appended environment. -/
def inLeft {Γ : List Shape} (right : List Shape) {s : Shape} : Var Γ s → Var (Γ ++ right) s
  | .head => .head
  | .tail v => .tail (inLeft right v)

/-- Embed a variable from the right side of an appended environment. -/
def inRight (left : List Shape) {Γ : List Shape} {s : Shape} : Var Γ s → Var (left ++ Γ) s :=
  match left with
  | .nil => fun v => v
  | .cons _ rest => fun v => .tail (inRight rest v)

/-- The final variable in an environment extended by one value. -/
def last (Γ : List Shape) {s : Shape} : Var (Γ ++ [s]) s :=
  match Γ with
  | .nil => .head
  | .cons _ rest => .tail (last rest)

end Var

namespace Env

/-- Typed environment lookup for pure tensors. -/
def tget {α : Type} [TorchLean.Storage α] :
    {Γ : List Shape} → {s : Shape} → TorchLean.TensorPack α Γ → Var Γ s →
      TorchLean.Tensor α s
  | _ :: _, _, .cons x _, .head => x
  | _ :: _, _, .cons _ xs, .tail i => tget xs i

/-- Appending a value does not change the meaning of an existing variable. -/
@[simp] theorem tget_append_weakenRight {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {s t : Shape}
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t) (i : Var Γ s) :
    tget (Pack.append env (.cons value .nil))
        (Var.weakenRight i) = tget env i := by
  induction i with
  | head => cases env; rfl
  | tail i ih =>
      cases env with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the left reads the original left environment. -/
@[simp] theorem tget_append_inLeft {α : Type} [TorchLean.Storage α]
    {Γ Δ : List Shape} {s : Shape}
    (left : TorchLean.TensorPack α Γ) (right : TorchLean.TensorPack α Δ) (v : Var Γ s) :
    tget (Pack.append left right) (Var.inLeft Δ v) = tget left v := by
  induction v with
  | head => cases left; rfl
  | tail v ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up a variable embedded from the right reads the appended right environment. -/
@[simp] theorem tget_append_inRight {α : Type} [TorchLean.Storage α]
    {Γ Δ : List Shape} {s : Shape}
    (left : TorchLean.TensorPack α Γ) (right : TorchLean.TensorPack α Δ) (v : Var Δ s) :
    tget (Pack.append left right) (Var.inRight Γ v) = tget right v := by
  induction Γ with
  | nil => cases left; rfl
  | cons shape Γ ih =>
      cases left with
      | cons _ rest => exact ih rest

/-- Looking up the final variable returns the value most recently appended to an environment. -/
@[simp] theorem tget_append_last {α : Type} [TorchLean.Storage α]
    {Γ : List Shape} {s : Shape}
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α s) :
    tget (Pack.append env (.cons value .nil)) (Var.last Γ) = value := by
  induction Γ with
  | nil => cases env; rfl
  | cons _ Γ ih =>
      cases env with
      | cons _ rest => exact ih rest

/-- Form an environment by reading each of its positions. -/
def ofLookup {α : Type} [Storage α] : (shapes : List Shape) →
    ({s : Shape} → Var shapes s → Tensor α s) → TensorPack α shapes
  | [], _ => .nil
  | _ :: rest, read => .cons (read .head) (ofLookup rest (fun position => read (.tail position)))

@[simp] theorem ofLookup_tget {α : Type} [Storage α] {shapes : List Shape}
    (env : TensorPack α shapes) : ofLookup shapes (tget env) = env := by
  induction env with
  | nil => rfl
  | cons value rest ih =>
    change Pack.cons value (ofLookup _ (tget rest)) = _
    rw [ih]

@[simp] theorem tget_ofLookup {α : Type} [Storage α] {shapes : List Shape} {s : Shape}
    (read : {t : Shape} → Var shapes t → Tensor α t) (position : Var shapes s) :
    tget (ofLookup shapes read) position = read position := by
  induction position with
  | head => rfl
  | tail position ih => exact ih (fun v => read (.tail v))

theorem ofLookup_congr {α : Type} [Storage α] {shapes : List Shape}
    {read other : {s : Shape} → Var shapes s → Tensor α s}
    (h : ∀ {s} (position : Var shapes s), read position = other position) :
    ofLookup shapes read = ofLookup shapes other := by
  congr 1
  funext s position
  exact h position

end Env

/-- A shape-preserving tensor operation in a forward equation. -/
structure PrimOp (ins : List Shape) (out : Shape) where
  specFwd : ∀ {α : Type}, [Storage α] → [Context α] → TensorPack α ins → Tensor α out

/-- A tensor-valued expression in a typed input environment. -/
structure Term (Γ : List Shape) (s : Shape) where
  run : ∀ {α : Type}, [Storage α] → [Context α] → TensorPack α Γ → Tensor α s

/-- One expression of each shape, all using the same input environment. -/
inductive Args (Γ : List Shape) : List Shape → Type 1 where
  | nil : Args Γ []
  | cons {s : Shape} {ss : List Shape} : Term Γ s → Args Γ ss → Args Γ (s :: ss)

namespace Term

def eval {Γ : List Shape} {s : Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (term : Term Γ s) : Tensor α s :=
  term.run env

@[simp] theorem eval_mk {Γ : List Shape} {s : Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ)
    (f : ∀ {β : Type}, [Storage β] → [Context β] → TensorPack β Γ → Tensor β s) :
    eval env ⟨f⟩ = f env := rfl

def var {Γ : List Shape} {s : Shape} (i : Var Γ s) : Term Γ s :=
  ⟨fun env => Env.tget env i⟩

def cast {Γ : List Shape} {s t : Shape} (term : Term Γ s) (h : s = t) : Term Γ t :=
  h ▸ term

def evalArgs {Γ : List Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) : {ss : List Shape} → Args Γ ss → TensorPack α ss
  | [], .nil => .nil
  | _ :: _, .cons term rest => .cons (eval env term) (evalArgs env rest)

def op {Γ ins : List Shape} {s : Shape} (primitive : PrimOp ins s) (args : Args Γ ins) :
    Term Γ s :=
  ⟨fun env => primitive.specFwd (evalArgs env args)⟩

def let1 {Γ : List Shape} {s t : Shape} (value : Term Γ s) (body : Term (Γ ++ [s]) t) :
    Term Γ t :=
  ⟨fun env => body.run (Pack.append env (.cons (value.run env) .nil))⟩

def rename {Γ Δ : List Shape} {s : Shape}
    (ρ : {t : Shape} → Var Γ t → Var Δ t) (term : Term Γ s) : Term Δ s :=
  ⟨fun env => term.run (Env.ofLookup Γ (fun position => Env.tget env (ρ position)))⟩

@[simp] theorem rename_var {Γ Δ : List Shape} {s : Shape}
    (ρ : {t : Shape} → Var Γ t → Var Δ t) (position : Var Γ s) :
    rename ρ (var position) = var (ρ position) := by
  apply congrArg Term.mk
  funext α storage context env
  exact Env.tget_ofLookup _ position

def weakenLeft {Γ : List Shape} {s t : Shape} (term : Term Γ s) : Term (t :: Γ) s :=
  term.rename (fun v => .tail v)

def weakenAppend {Γ : List Shape} (right : List Shape) {s : Shape} (term : Term Γ s) :
    Term (Γ ++ right) s :=
  term.rename (Var.inLeft right)

def weakenRight {Γ : List Shape} {s t : Shape} (term : Term Γ s) : Term (Γ ++ [t]) s :=
  term.rename Var.weakenRight

@[simp] theorem eval_op {Γ ins : List Shape} {s : Shape} {α : Type}
    [Storage α] [Context α] (env : TensorPack α Γ) (primitive : PrimOp ins s)
    (args : Args Γ ins) : eval env (op primitive args) = primitive.specFwd (evalArgs env args) :=
  rfl

@[simp] theorem eval_cast {Γ : List Shape} {s t : Shape} {α : Type}
    [Storage α] [Context α] (env : TensorPack α Γ) (term : Term Γ s) (h : s = t) :
    eval env (cast term h) = Tensor.castShape (eval env term) h := by
  cases h
  rfl

end Term

namespace Args

def rename {Γ Δ : List Shape} (ρ : {t : Shape} → Var Γ t → Var Δ t) :
    {ss : List Shape} → Args Γ ss → Args Δ ss
  | [], .nil => .nil
  | _ :: _, .cons term rest => .cons (Term.rename ρ term) (rename ρ rest)

def weakenLeft {Γ ss : List Shape} {t : Shape} (args : Args Γ ss) : Args (t :: Γ) ss :=
  args.rename (fun v => .tail v)

/-- Select the term assigned to a shape-indexed variable. -/
def get {Γ Δ : List Shape} {s : Shape} : Args Δ Γ → Var Γ s → Term Δ s
  | .cons term _, .head => term
  | .cons _ rest, .tail v => get rest v

/-- Concatenate two typed argument lists living in the same graph environment. -/
def append {Γ left right : List Shape} :
    Args Γ left → Args Γ right → Args Γ (left ++ right)
  | .nil, rightArgs => rightArgs
  | .cons term rest, rightArgs => .cons term (append rest rightArgs)

/-- View every entry of a typed environment as a term in that same environment.

The result preserves the order and shape indices of `Γ`.  Large graph definitions can therefore
pattern-match once on `vars Γ` instead of selecting every input by a numeric index and separately
proving that the selected position has the expected shape. -/
def vars : (Γ : List Shape) → Args Γ Γ
  | .nil => .nil
  | .cons _ rest => .cons (.var .head) (Args.weakenLeft (vars rest))

/-- Selecting from renamed arguments renames the selected term. -/
@[simp] theorem get_rename {Γ Δ ss : List Shape} {s : Shape}
    (ρ : {t : Shape} → Var Γ t → Var Δ t) (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.rename ρ args) position = Term.rename ρ (Args.get args position) := by
  induction position with
  | head => cases args; rfl
  | tail position ih =>
      cases args with
      | cons _ rest => exact ih rest

/-- Lookup commutes with embedding an argument list below one new graph variable. -/
@[simp] theorem get_weakenLeft {Γ ss : List Shape} {s t : Shape}
    (args : Args Γ ss) (position : Var ss s) :
    Args.get (Args.weakenLeft (t := t) args) position =
      Term.weakenLeft (t := t) (Args.get args position) := by
  exact get_rename (fun v => .tail v) args position

/-- Selecting a variable from the complete environment returns that variable as a term. -/
@[simp] theorem get_vars {Γ : List Shape} {s : Shape} (position : Var Γ s) :
    Args.get (vars Γ) position = Term.var position := by
  induction position with
  | head => rfl
  | tail position ih =>
      rw [vars, get, get_weakenLeft, ih]
      exact Term.rename_var (fun v => .tail v) position

end Args

namespace Term

@[simp] theorem eval_var {α : Type} [Storage α] [Context α] {Γ : List Shape}
    {s : Shape} (env : TensorPack α Γ) (v : Var Γ s) : eval env (var v) = Env.tget env v :=
  rfl

@[simp] theorem eval_let1 {α : Type} [Storage α] [Context α] {Γ : List Shape}
    {s t : Shape} (env : TensorPack α Γ) (value : Term Γ s) (body : Term (Γ ++ [s]) t) :
    eval env (let1 value body) = eval (Pack.append env (.cons (Term.eval env value) .nil)) body :=
  rfl

theorem eval_rename {α : Type} [Storage α] [Context α] {Γ Δ : List Shape} {s : Shape}
    (envΓ : TensorPack α Γ) (envΔ : TensorPack α Δ)
    (ρ : {t : Shape} → Var Γ t → Var Δ t)
    (h : ∀ {t} (position : Var Γ t), Env.tget envΔ (ρ position) = Env.tget envΓ position)
    (term : Term Γ s) : eval envΔ (rename ρ term) = eval envΓ term := by
  change term.run (Env.ofLookup Γ (fun position => Env.tget envΔ (ρ position))) = term.run envΓ
  rw [Env.ofLookup_congr h, Env.ofLookup_tget]

theorem evalArgs_rename {α : Type} [Storage α] [Context α] {Γ Δ : List Shape}
    (envΓ : TensorPack α Γ) (envΔ : TensorPack α Δ)
    (ρ : {t : Shape} → Var Γ t → Var Δ t)
    (h : ∀ {t} (position : Var Γ t), Env.tget envΔ (ρ position) = Env.tget envΓ position) :
    {ss : List Shape} → (args : Args Γ ss) →
      evalArgs envΔ (Args.rename ρ args) = evalArgs envΓ args
  | [], .nil => rfl
  | _ :: _, .cons term rest => by
    simp only [Args.rename, evalArgs, eval_rename envΓ envΔ ρ h, evalArgs_rename envΓ envΔ ρ h]

@[simp] theorem eval_get {Γ Δ : List Shape} {s : Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Δ) (args : Args Δ Γ) (position : Var Γ s) :
    eval env (Args.get args position) = Env.tget (evalArgs env args) position := by
  induction position with
  | head => cases args; rfl
  | tail position ih => cases args; exact ih _

def instantiate {Γ Δ : List Shape} {s : Shape} (arguments : Args Δ Γ) (term : Term Γ s) :
    Term Δ s :=
  ⟨fun env => term.run (evalArgs env arguments)⟩

end Term

/-- A forward computation producing a tensor pack. -/
structure Block (Γ outs : List Shape) where
  run : ∀ {α : Type}, [Storage α] → [Context α] → TensorPack α Γ → TensorPack α outs

namespace Block

def eval {Γ outs : List Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (block : Block Γ outs) : TensorPack α outs :=
  block.run env

def ret {Γ outs : List Shape} (results : Args Γ outs) : Block Γ outs :=
  ⟨fun env => Term.evalArgs env results⟩

def let1 {Γ outs : List Shape} {s : Shape} (value : Term Γ s) (body : Block (Γ ++ [s]) outs) :
    Block Γ outs :=
  ⟨fun env => body.run (Pack.append env (.cons (value.run env) .nil))⟩

def instantiate {Γ Δ outs : List Shape} (arguments : Args Δ Γ) (block : Block Γ outs) :
    Block Δ outs :=
  ⟨fun env => block.run (Term.evalArgs env arguments)⟩

/-- Supply the first computation's outputs together with the original inputs. -/
def andThen {Γ middle outs : List Shape} (first : Block Γ middle)
    (second : Block (Γ ++ middle) outs) : Block Γ outs :=
  ⟨fun env => second.run (Pack.append env (first.run env))⟩

@[simp] theorem eval_ret {Γ outs : List Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (results : Args Γ outs) :
    eval env (ret results) = Term.evalArgs env results := rfl

@[simp] theorem eval_let1 {Γ outs : List Shape} {s : Shape} {α : Type}
    [Storage α] [Context α] (env : TensorPack α Γ) (value : Term Γ s)
    (body : Block (Γ ++ [s]) outs) :
    eval env (let1 value body) = eval (Pack.append env (.cons (Term.eval env value) .nil)) body :=
  rfl

@[simp] theorem eval_cast {Γ outs outs' : List Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (h : outs = outs') (block : Block Γ outs) :
    eval env (h ▸ block) = h ▸ eval env block := by
  cases h
  rfl

theorem eval_andThen {Γ middle outs : List Shape} {α : Type} [Storage α] [Context α]
    (env : TensorPack α Γ) (first : Block Γ middle) (second : Block (Γ ++ middle) outs) :
    eval env (first.andThen second) = eval (Pack.append env (eval env first)) second := rfl

end Block

namespace Term

/-- Renaming arguments into the left side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inLeft
    {Γ Δ ins : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (args : Args Γ ins) :
    evalArgs (Pack.append left right)
        (Args.rename (Var.inLeft Δ) args) = evalArgs left args := by
  apply evalArgs_rename left (Pack.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Renaming arguments into the right side of an appended environment preserves their values. -/
@[simp] theorem evalArgs_rename_inRight
    {Γ Δ ins : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (args : Args Δ ins) :
    evalArgs (Pack.append left right)
        (Args.rename (Var.inRight Γ) args) = evalArgs right args := by
  apply evalArgs_rename right (Pack.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Renaming a term into the left side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inLeft
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Γ s) :
    eval (Pack.append left right)
        (Term.rename (Var.inLeft Δ) term) = eval left term := by
  apply eval_rename left (Pack.append left right)
  intro shape position
  exact Env.tget_append_inLeft left right position

/-- Appending an arbitrary typed environment does not change a weakened term's value. -/
@[simp] theorem eval_weakenAppend
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Γ s) :
    eval (Pack.append left right)
        (Term.weakenAppend Δ term) = eval left term := by
  exact eval_rename_inLeft left right term

/-- Renaming a term into the right side of an appended environment preserves its value. -/
@[simp] theorem eval_rename_inRight
    {Γ Δ : List Shape} {s : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (left : TorchLean.TensorPack α Γ)
    (right : TorchLean.TensorPack α Δ) (term : Term Δ s) :
    eval (Pack.append left right)
        (Term.rename (Var.inRight Γ) term) = eval right term := by
  apply eval_rename right (Pack.append left right)
  intro shape position
  exact Env.tget_append_inRight left right position

/-- Appending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenRight
    {Γ : List Shape} {s t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t)
    (term : Term Γ s) :
    eval (Pack.append env (.cons value .nil))
      (Term.weakenRight term) = eval env term := by
  apply eval_rename env
    (Pack.append env (.cons value .nil)) Var.weakenRight
  intro shape v
  exact Env.tget_append_weakenRight env value v

/-- The final variable in an extended environment denotes the value that was just appended. -/
@[simp] theorem eval_var_last_append
    {Γ : List Shape} {t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t) :
    eval (Pack.append env (.cons value .nil))
      (Term.var (Var.last Γ)) = value := by
  exact Env.tget_append_last env value


/-- Evaluating concatenated graph arguments concatenates their tensor values in the same order. -/
theorem evalArgs_append
    {Γ left right : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) :
    (leftArgs : Args Γ left) → (rightArgs : Args Γ right) →
      evalArgs env (Args.append leftArgs rightArgs) =
        Pack.append
          (evalArgs env leftArgs) (evalArgs env rightArgs)
  | .nil, _ => rfl
  | .cons term rest, rightArgs => by
      simp only [Args.append, evalArgs, Pack.append]
      rw [evalArgs_append env rest rightArgs]

/-- Prepending an unrelated value does not change a term's pure meaning. -/
@[simp] theorem eval_weakenLeft
    {Γ : List Shape} {s t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t)
    (term : Term Γ s) :
    eval (.cons value env) (Term.weakenLeft term) = eval env term := by
  apply eval_rename env (.cons value env) (fun v => .tail v)
  intro shape v
  rfl

/-- Prepending an unrelated value does not change a typed argument list's pure meaning. -/
@[simp] theorem evalArgs_weakenLeft
    {Γ ins : List Shape} {t : Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (value : TorchLean.Tensor α t)
    (args : Args Γ ins) :
    evalArgs (.cons value env) (Args.weakenLeft args) = evalArgs env args := by
  apply evalArgs_rename env (.cons value env) (fun v => .tail v)
  intro shape v
  rfl

/-- Evaluating all variables of an environment returns that environment in order. -/
@[simp] theorem evalArgs_vars
    {Γ : List Shape} {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) :
    evalArgs env (Args.vars Γ) = env := by
  induction Γ with
  | nil => cases env; rfl
  | cons shape Γ ih =>
      cases env with
      | cons value rest =>
          simp only [Args.vars, evalArgs, eval_var, Env.tget, evalArgs_weakenLeft]
          rw [ih rest]


end Term

/-! ## Model wrapper -/

/-- A forward expression whose environment begins with its parameters. -/
structure Model (ps ins : List Shape) (τ : Shape) where
  /-- Initial parameter values, one tensor per shape in `ps`. Shipping the initialization with the
  model means a `Model` is runnable on its own, with no separate setup step. -/
  initParams : TorchLean.TensorPack Float ps
  /-- The computation itself, as a term over the environment `ps ++ ins`: parameters first, then
  data inputs. That fixed ordering is what lets `initParams` be typed by `ps` alone. -/
  body : Term (ps ++ ins) τ

namespace Model

/-- Supply a model with parameter and input expressions from a surrounding computation. -/
def inline {Γ ps ins : List Shape} {τ : Shape} (model : Model ps ins τ)
    (params : Args Γ ps) (inputs : Args Γ ins) : Term Γ τ :=
  model.body.instantiate (Args.append params inputs)

/-- Evaluate the forward expression with the given parameters and inputs. -/
def specFwd {ps ins : List Shape} {τ : Shape} (m : Model ps ins τ)
    {α : Type 0} [TorchLean.Storage α] [Context α]
    (params : TorchLean.TensorPack α ps) (xs : TorchLean.TensorPack α ins) : TorchLean.Tensor α τ :=
  let env := Pack.append (ss₁ := ps) (ss₂ := ins) params xs
  Term.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Composing a model with input expressions evaluates those inputs before the model. -/
theorem eval_inline {Γ ps ins : List Shape} {τ : Shape}
    (model : Model ps ins τ) {α : Type 0} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (params : Args Γ ps) (inputs : Args Γ ins) :
    Term.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  simp only [inline, specFwd, Term.instantiate, Term.eval, Term.evalArgs_append]


end Model

/-! ## Models with several outputs -/

/-- A forward model that returns several tensors, such as an updated recurrent state and its readout. -/
structure MultiModel (ps ins outs : List Shape) where
  /-- Default parameters, in the same ABI order used by `body`. -/
  initParams : TorchLean.TensorPack Float ps
  /-- Shared computation ending in one well-typed term for every output shape. -/
  body : Block (ps ++ ins) outs

namespace MultiModel

/-- Supply parameter and input expressions to a model with several outputs. -/
def inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    (params : Args Γ ps) (inputs : Args Γ ins) : Block Γ outs :=
  model.body.instantiate (Args.append params inputs)

/-- Evaluate all outputs with the given parameters and inputs. -/
def specFwd {ps ins outs : List Shape} (m : MultiModel ps ins outs)
    {α : Type 0} [TorchLean.Storage α] [Context α] (params : TorchLean.TensorPack α ps)
    (xs : TorchLean.TensorPack α ins) : TorchLean.TensorPack α outs :=
  let env := Pack.append
    (ss₁ := ps) (ss₂ := ins) params xs
  Block.eval (Γ := ps ++ ins) (α := α) env m.body

/-- Composing a multi-output model preserves its forward evaluation. -/
theorem eval_inline {Γ ps ins outs : List Shape} (model : MultiModel ps ins outs)
    {α : Type 0} [TorchLean.Storage α] [Context α] (env : TorchLean.TensorPack α Γ)
    (params : Args Γ ps) (inputs : Args Γ ins) :
    Block.eval env (model.inline params inputs) =
      model.specFwd (Term.evalArgs env params) (Term.evalArgs env inputs) := by
  simp only [inline, specFwd, Block.instantiate, Block.eval, Term.evalArgs_append]


end MultiModel

end KimiK3.GraphSpec
