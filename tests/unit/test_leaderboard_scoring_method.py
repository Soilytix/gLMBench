"""Leaderboard disclosure of mixed scoring methods.

When a variant-effect task column mixes AR sequence-loglikelihood scores with MLM
masked-marginal LLR scores, the board must disclose it: a per-task **footnote** naming the
fallback users, a ``scoring_method`` **CSV column**, and a footer note. The metric stays
abs-Spearman (comparable in the ProteinGym sense); only the method differs. Torch-free —
records are built by hand to pin the rendering contract.
"""

from __future__ import annotations

from glmbench.leaderboard.render import aggregate_dataframe, render_markdown

TASK = "rnagym-dms"


def _record(model_hash: str, display: str, spearman: float, method: str) -> dict:
    return {
        "model_hash": f"glmb:{model_hash}",
        "adapter_name": display,
        "adapter_version": "0.1.0",
        "benchmark": {"name": "glmbench", "version": "1.0"},
        "model_spec": {"display_name": display},
        "tasks": [
            {
                "task": TASK,
                "status": "ok",
                "primary_metric": "macro_spearman",
                "metrics": {"macro_spearman": spearman},
                "metadata": {"scoring_method": method},
            }
        ],
    }


def _ar(model_hash, display, spearman):
    return _record(model_hash, display, spearman, "sequence_loglikelihood")


def _mlm(model_hash, display, spearman):
    return _record(model_hash, display, spearman, "masked_marginal_llr")


def test_footnote_emitted_for_lone_mlm_record():
    """A single MLM (fallback) record renders the disclosure footnote."""
    md = render_markdown(
        [_mlm("aaa", "glm2", 0.31)],
        benchmark_name="glmbench",
        benchmark_version="1.0",
        task_order=[TASK],
    )
    assert "Mixed scoring methods" in md
    assert "masked-marginal LLR" in md
    assert "glm2" in md


def test_no_footnote_when_all_ar():
    """An all-AR board emits no per-task scoring-method footnote (back-compat)."""
    md = render_markdown(
        [_ar("aaa", "loam-hf", 0.40), _ar("bbb", "evo2-arc", 0.37)],
        benchmark_name="glmbench",
        benchmark_version="1.0",
        task_order=[TASK],
    )
    assert "Mixed scoring methods" not in md


def test_mixed_board_sorts_and_shows_both():
    """A mixed AR + MLM board sorts by metric desc and discloses the MLM fallback."""
    md = render_markdown(
        [_ar("aaa", "loam-hf", 0.40), _mlm("bbb", "glm2", 0.31), _ar("ccc", "evo2-arc", 0.37)],
        benchmark_name="glmbench",
        benchmark_version="1.0",
        task_order=[TASK],
    )
    # the per-task table lists all three; sort order: loam-hf (.40) > evo2-arc (.37) > glm2 (.31)
    task_section = md.split(f"## Task: {TASK}")[1]
    pos_loam = task_section.index("loam-hf")
    pos_evo2 = task_section.index("evo2-arc")
    pos_glm2 = task_section.index("glm2")
    assert pos_loam < pos_evo2 < pos_glm2
    # footnote names the MLM model + its method, not the AR models.
    footnote = md.split("Mixed scoring methods")[1]
    assert "glm2" in footnote and "scored via masked-marginal LLR" in footnote
    # AR models are not named as fallback users in the footnote clause list.
    assert "loam-hf scored via" not in footnote and "evo2-arc scored via" not in footnote


def test_csv_has_scoring_method_column():
    df = aggregate_dataframe([_ar("aaa", "loam-hf", 0.40), _mlm("bbb", "glm2", 0.31)])
    assert "scoring_method" in df.columns
    methods = dict(zip(df["adapter"], df["scoring_method"], strict=True))
    assert methods["loam-hf"] == "sequence_loglikelihood"
    assert methods["glm2"] == "masked_marginal_llr"


def test_footer_explains_llr_fallback():
    md = render_markdown(
        [_mlm("aaa", "glm2", 0.31)],
        benchmark_name="glmbench",
        benchmark_version="1.0",
        task_order=[TASK],
    )
    assert "masked-marginal LLR" in md.rsplit("---", 1)[-1]


def test_footer_omits_llr_sentence_on_an_all_ar_board():
    """The footer's LLR sentence is a claim about THIS board — not boilerplate.

    Emitted unconditionally, it would state a fallback that never happened on an all-AR
    board.
    """
    md = render_markdown(
        [_ar("aaa", "loam-hf", 0.40), _ar("bbb", "evo2-arc", 0.34)],
        benchmark_name="glmbench",
        benchmark_version="1.0",
        task_order=[TASK],
    )
    footer = md.rsplit("---", 1)[-1]
    assert "masked-marginal LLR" not in footer
    # The parts that are true of every board are still there.
    assert "RNAGym" in footer and "`N/A` = required capability missing" in footer
