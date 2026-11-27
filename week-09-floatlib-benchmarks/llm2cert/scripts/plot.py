"""LLM2Cert's proof obligations with `interval` alone, next to LLM2Cert's own result.

    python3 scripts/plot.py [RESULTS [FIGURES]]

Reads the files of `results/` and writes `figures/NN-NAME.png`, `.svg` and `figures/CAPTIONS.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots"))
from benchplot import MACHINE, Suite, fs, plt  # noqa: E402

suite = Suite(__file__)
read, done = suite.read, suite.done


def fig_llm2cert() -> None:
    data = read("llm2cert.csv")
    facts = suite.facts("llm2cert-run.csv")
    drawn = [row for row in data if row["drawn"] == "yes"]
    grey = fs.GREY_DARK
    panels = [("obligations", "obligations closed (of 291)", "closed", 291, "{value:.0f}"),
              ("certificates", "certificates with every obligation closed (of 81)", "certificates", 81,
               "{value:.0f}"),
              ("time", "time Lean ran (hours)", "hours", None, "{text}")]
    fig, axes = plt.subplots(3, 1, figsize=(fs.WIDTH, 0.8 * 3 * len(drawn) + 4.6))
    for index, (ax, (title, quantity, column, limit, pattern)) in enumerate(zip(axes, panels)):
        values = [float(row[column]) for row in drawn]
        top = limit if limit else max(values) * 1.02
        for position, (row, value) in enumerate(zip(drawn, values)):
            y = len(drawn) - 1 - position
            colour = fs.BLUE if row["tool"] == "FloatLib" else grey
            ax.barh(y, value, height=0.42, color=colour, edgecolor=colour, linewidth=fs.LW_AXIS, zorder=2)
            ax.text(0.0, y + 0.29, row["label"], ha="left", va="bottom", fontsize=fs.SIZE_SMALL, color=fs.INK)
            text = pattern.format(value=value, text=row["time_label"])
            ax.text(value + top * 0.012, y, text, ha="left", va="center", fontsize=fs.SIZE_SMALL, color=fs.INK)
        ax.set_ylim(-0.45, len(drawn) - 0.15)
        ax.set_xlim(0, top)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(quantity)
        fs.value_grid(ax, "x")
        fs.panel_title(ax, index, title)
    fig.subplots_adjust(left=0.03, right=0.88, top=0.96, bottom=0.075, hspace=0.62)
    ours = {row["key"]: row for row in data}
    default, raised, theirs = ours["mode-a-defaults"], ours["mode-a-raised"], ours["llm2cert-all-lean"]
    done(fig, "01-llm2cert-obligations",
         "The 291 proof obligations of LLM2Cert's benchmark DynaCertBench, each given to FloatLib's interval "
         "tactic as the one-line proof `by interval` with no model, next to LLM2Cert's own All-Lean result, in "
         "which a language model writes the proofs. Panel (a) counts the obligations that Lean accepts with the "
         "three standard axioms only, panel (b) the 81 certificates all of whose obligations are closed, and "
         "panel (c) is the time Lean ran, in hours on a linear axis: for interval the sum over all 291 files, "
         "for LLM2Cert the time on the obligations it closed, to which its "
         f"{float(theirs['tokens']) / 1e6:.2f} million model tokens are added. interval was run on FloatLib "
         f"on {facts['date']} with Lean {facts['lean']} on {MACHINE}; LLM2Cert's numbers are those of its "
         "published records.",
         "Three panels of three horizontal bars each, two blue bars for interval with Lean's default limits and "
         "with raised limits and one grey bar for LLM2Cert. In the first two panels the three bars are of "
         "similar length, each ending in its count; in the third the two blue bars are a small fraction of the "
         "grey bar, and each bar ends in its time.",
         f"`interval` alone closes {default['closed']} of 291 obligations and completes {default['certificates']} "
         f"of 81 certificates with Lean's default `maxHeartbeats` and `maxRecDepth` ({default['time_label']} of "
         f"Lean time over all 291 files, median {default['median_s']} s per closed obligation), and "
         f"{raised['closed']} and {raised['certificates']} with the limits raised to 4,000,000 heartbeats and a "
         f"recursion depth of 100,000 ({raised['time_label']}, median {raised['median_s']} s); no model and no "
         f"tokens. LLM2Cert All-Lean: {theirs['closed']} of 291, {theirs['certificates']} of 81, "
         f"{theirs['time_label']} on the obligations it closed and {float(theirs['tokens']) / 1e6:.2f} million "
         f"tokens. {facts['overlap']} Six Lean processes at a time, 600 seconds per obligation; the slowest "
         f"closed obligation took {facts['max_s_defaults']} s and {facts['max_s_raised']} s on the shared machine, "
         f"and with 300 seconds the counts are {facts['closed_300_defaults']} and "
         f"{facts['certificates_300_defaults']} with the default limits, {facts['closed_300_raised']} and "
         f"{facts['certificates_300_raised']} with the raised ones. "
         "Every statement is LLM2Cert's, verbatim; only the proof is generated, and an "
         "obligation counts as closed only when Lean accepts the file and `#print axioms` shows exactly "
         "`propext`, `Classical.choice` and `Quot.sound`. Three of the 291 statements say that a function is "
         "radially unbounded, which is no set of constraints for `interval` to refute. From Lean: whether each "
         "obligation closed. From scripts: the files, the wall time of each Lean process, the counts and the "
         "sums; LLM2Cert's numbers (their `lean_records.json`). Source: `results/llm2cert.csv`.")


if __name__ == "__main__":
    fig_llm2cert()
    suite.write_captions("LLM2Cert's obligations with interval")
