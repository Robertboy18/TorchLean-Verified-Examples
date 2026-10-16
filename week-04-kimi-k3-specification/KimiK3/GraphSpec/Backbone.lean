/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Robert Joseph George
-/

module

public import KimiK3.GraphSpec.Backbone.KDADense
public import KimiK3.GraphSpec.Backbone.KDASparse
public import KimiK3.GraphSpec.Backbone.MLADense
public import KimiK3.GraphSpec.Backbone.MLASparse

/-!
# Composed Kimi K3 backbone graphs

The dense and sparse KDA and MLA decoder-layer graphs. All four share the packed AttnRes depth
interface, and each is proved equal to its reference transition. Sparse layers take their route as
a static argument, which the composed decoder relates to the computed route.
-/
