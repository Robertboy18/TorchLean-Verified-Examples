/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.Common
public import Mathlib.Data.List.OfFn

/-!
# Kimi K3 chat-template structure

Appendix F of the Kimi K3 report specifies the XTML message layout, placement of global and
one-shot options, assistant channels, and indexed parallel tool calls.  This module models that
grammar as typed syntax.  It deliberately stops before tokenizer IDs and natural-language option
wording, which are not fixed mathematically by the report.
-/

@[expose] public section

namespace KimiK3
namespace ChatTemplate

/-- Structural XTML tokens. Text and attribute payloads remain ordinary strings. -/
inductive Token where
  | openTag (name : String) (attributes : List (String × String))
  | text (value : String)
  | closeTag (name : String)
  | endOfMessage
  deriving DecidableEq

/-- An XTML element has explicit opening and closing boundaries. -/
structure Element where
  /-- Tag name shared by the opening and closing tokens. -/
  name : String
  /-- Attributes emitted on the opening tag. -/
  attributes : List (String × String)
  /-- Tokens enclosed by the element boundaries. -/
  body : List Token
  deriving DecidableEq

namespace Element

/-- Serialize one element with explicit opening and closing tag tokens. -/
def tokens (element : Element) : List Token :=
  .openTag element.name element.attributes :: element.body ++ [.closeTag element.name]

end Element

/-- Input-message roles supported by the request `messages` field. -/
inductive InputRole where
  | system
  | user
  | assistant
  | tool
  deriving DecidableEq

/-- Scope determines where an option message is placed in the context. -/
inductive OptionScope where
  | global
  | input
  | oneShot
  deriving DecidableEq

/-- Four effort names reserved by the template schema. -/
inductive ReasoningEffort where
  | low
  | medium
  | high
  | max
  deriving DecidableEq

/-- Options translated into natural-language context instructions. -/
inductive Option where
  | toolDeclaration (description : String)
  | thinkingEffort (effort : ReasoningEffort) (instruction : String)
  | toolChoice (instruction : String)
  | responseFormat (instruction : String)
  | extension (name instruction : String)
  deriving DecidableEq

/-- Input messages and dynamically inserted input-scope options share one chronological stream. -/
inductive InputItem where
  | message (role : InputRole) (body : List Token)
  | option (value : Option)
  deriving DecidableEq

/-- A complete context keeps stable options before history and one-shot options after history. -/
structure Context where
  /-- Options that remain stable across requests sharing the same prefix. -/
  globalOptions : List Option
  /-- Chronological message history and input-scoped options. -/
  inputItems : List InputItem
  /-- Options that apply only to the next generation. -/
  oneShotOptions : List Option
  deriving DecidableEq

namespace Context

/-- Zone-tagged context items, before concrete XTML rendering. -/
inductive Item where
  | option (scope : OptionScope) (value : Option)
  | input (value : InputItem)
  deriving DecidableEq

/-- Flatten the three context zones in cache-preserving order. -/
def layout (context : Context) : List Item :=
  context.globalOptions.map (Item.option .global) ++
    context.inputItems.map Item.input ++
      context.oneShotOptions.map (Item.option .oneShot)

/-- Changing only one-shot options preserves the complete global/history prefix. -/
theorem layout_prefix_invariant (context : Context) (newOneShot : List Option) :
    context.globalOptions.map (Item.option .global) ++ context.inputItems.map Item.input <+:
      { context with oneShotOptions := newOneShot }.layout := by
  refine ⟨newOneShot.map (Item.option .oneShot), ?_⟩
  simp [layout, List.append_assoc]

end Context

/-- The two generation prefixes described by the report. -/
inductive GenerationMode where
  | thinking
  | instruct
  deriving DecidableEq

/-- Assistant-body channel names. -/
inductive Channel where
  | think
  | response
  | tools
  deriving DecidableEq

/-- Typed argument blocks keep strings distinct from compactly serialized non-string JSON. -/
inductive ArgumentValue where
  | string (value : String)
  | compactJson (value : String)
  deriving DecidableEq

/-- One named argument supplied to a tool call. -/
structure Argument where
  /-- Parameter name expected by the tool. -/
  name : String
  /-- Typed serialized value assigned to the parameter. -/
  value : ArgumentValue
  deriving DecidableEq

/-- One tool call before its parallel-call index is assigned. -/
structure ToolCall where
  /-- Name of the tool to invoke. -/
  tool : String
  /-- Arguments passed to the tool in template order. -/
  arguments : List Argument
  deriving DecidableEq

/-- A batch indexed by `Fin count` makes parallel call/result association unambiguous. -/
abbrev ToolCallBatch (count : ℕ) := Fin count → ToolCall

/-- Tool-result bodies indexed by the corresponding finite call slot. -/
abbrev ToolResultBatch (count : ℕ) := Fin count → List Token

/-- Emit calls in increasing index order. -/
def indexedCalls {count : ℕ} (calls : ToolCallBatch count) :
    List (Fin count × ToolCall) :=
  List.ofFn fun index => (index, calls index)

/-- Emit results in call order while copying each call's tool/index pair. -/
def indexedResults {count : ℕ} (calls : ToolCallBatch count)
    (results : ToolResultBatch count) : List (Fin count × String × List Token) :=
  List.ofFn fun index => (index, (calls index).tool, results index)

/-- Calls and results enumerate every finite slot once, in the same increasing order. -/
theorem indexed_call_result_indices {count : ℕ} (calls : ToolCallBatch count)
    (results : ToolResultBatch count) :
    (indexedCalls calls).map Prod.fst = List.finRange count ∧
      (indexedResults calls results).map Prod.fst = List.finRange count := by
  constructor
  · rw [indexedCalls, List.map_ofFn]
    change List.ofFn id = List.finRange count
    exact List.ofFn_id count
  · rw [indexedResults, List.map_ofFn]
    change List.ofFn id = List.finRange count
    exact List.ofFn_id count

/-- Looking up a result by its position recovers the matching call's index and tool name. -/
theorem indexedResults_get {count : ℕ} (calls : ToolCallBatch count)
    (results : ToolResultBatch count) (index : Fin count) :
    (indexedResults calls results)[index.val]'(by
        simp [indexedResults, index.isLt]) =
      ⟨index, (calls index).tool, results index⟩ := by
  simp [indexedResults]

/-- Model-emitted tool payloads are structurally decomposed calls only.

The pure-JSON fallback described by the report belongs to input preprocessing and therefore cannot
be constructed through this output type.
-/
structure AssistantBody (toolCount : ℕ) where
  /-- Internal reasoning-channel tokens, present only in thinking mode. -/
  think : List Token
  /-- User-visible response-channel tokens. -/
  response : List Token
  /-- Tool calls indexed by their position in the parallel batch. -/
  tools : ToolCallBatch toolCount

/-- Channels retained when an assistant message is inserted into history. -/
def historicalChannels {toolCount : ℕ} (mode : GenerationMode)
    (_body : AssistantBody toolCount) : List Channel :=
  match mode with
  | .thinking => [.think, .response, .tools]
  | .instruct => [.response, .tools]

/-- The thinking channel is retained in history exactly in thinking mode. -/
theorem think_mem_historicalChannels_iff {toolCount : ℕ} (mode : GenerationMode)
    (body : AssistantBody toolCount) :
    .think ∈ historicalChannels mode body ↔ mode = .thinking := by
  cases mode <;> simp [historicalChannels]

/-- Generation starts directly inside the mode-selected channel. -/
def generationPrefix : GenerationMode → Token
  | .thinking => .openTag "think" []
  | .instruct => .openTag "response" []

end ChatTemplate
end KimiK3
