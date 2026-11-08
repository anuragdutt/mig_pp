"""
Graph ILP (graph_placement.py) vs GPipe ILP (gpipe_placement.py) on every oracle_v2 sweep.
Both use the same measured constants, slice bandwidths, per-stage terms and memory bounds;
they differ only in how stage times become a step time. The CSVs are only used to score.

Run from the repo root:  python3 mig_pp/milp/compare_graph_gpipe.py
Edit CONFIG in main(). Writes three CSVs into CONFIG["out_dir"]:

  predictions.csv  one row per measured run: measured ms, each model's predicted ms, error %
  pairs.csv        one row per (dataset, batch, microbatch):
                     best measured split; a typical split's distance from it (median);
                     each model's favourite AMONG MEASURED splits and how much slower it ran;
                     which favourite was faster (graph / gpipe / tie within tie_pct);
                     each optimizer's free pick over ALL splits that fit, and whether it was run;
                     Kendall tau-b of each model's order of the measured splits vs measured order

Ties: splits whose predictions agree to REL_TOL cannot be told apart by that model (GPipe
often scores several splits identically at 2 microbatches). A tied favourite is scored by
the AVERAGE of its tied splits' "slower than best %" (a random pick among them); the split
listed is the lowest in sort order and *_tied_splits counts the tie. GPipe's free pick is
found by enumerating every split, with the same rule, so reruns give identical files.
  summary.csv      medians / worst / win counts per dataset, per setup, per microbatch count,
                   computed only from pairs.csv and predictions.csv, so they can be re-checked

"slower than best %" = (measured ms of that split - fastest measured ms) / fastest measured ms.
The network hop of the two-node runs is not in either model (it moves predictions < 0.2%).
"""

import ast
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import mig_pp.milp.graph_placement as graph  # noqa: E402
import mig_pp.milp.gpipe_placement as gpipe  # noqa: E402
import mig_pp.milp.hardware_constants as hardware_constants  # noqa: E402


def load_rows(path):
    """(rows with status ok, slice GB per rank from the peak_rank*_<GB>gb_mb headers)."""
    with open(path, newline="") as f:
        table = list(csv.reader(f))
    header, body = table[0], [r for r in table[1:] if r]
    col = {name: i for i, name in enumerate(header)}
    peaks = sorted((int(m.group(1)), int(m.group(2)))
                   for m in (re.match(r"^peak_rank(\d+)_(\d+)gb_mb$", h) for h in header) if m)
    rows = []
    for r in body:
        if r[col["status"]] != "ok":
            continue
        rows.append(dict(split=tuple(ast.literal_eval(r[col["split"]])), batch=int(r[col["batch_size"]]),
                         mb=int(r[col["microbatch_size"]]), measured_ms=float(r[col["total_latency_ms"]])))
    return rows, [gb for _, gb in peaks]


REL_TOL = 1e-9  # predictions this close (relative) are the same prediction


def same(x, y):
    return abs(x - y) <= REL_TOL * max(abs(x), abs(y))


def kendall_tau_b(a, b):
    """Kendall tau-b: ties in either list count in the denominator."""
    concordant = discordant = tied_a = tied_b = pairs = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            pairs += 1
            ta, tb = same(a[i], a[j]), same(b[i], b[j])
            tied_a += ta
            tied_b += tb
            if not ta and not tb:
                if (a[i] - a[j]) * (b[i] - b[j]) > 0:
                    concordant += 1
                else:
                    discordant += 1
    denominator = ((pairs - tied_a) * (pairs - tied_b)) ** 0.5
    return (concordant - discordant) / denominator if denominator else float("nan")


def favourite(predicted):
    """(representative split, every split tied for the lowest prediction). The model cannot
    tell tied splits apart, so callers score the tie by its average; the representative is
    the lowest split in sort order, so reruns list the same one."""
    low = min(predicted.values())
    tied = sorted(s for s, v in predicted.items() if same(v, low))
    return tied[0], tied


def pct(value, best):
    return (value - best) / best * 100


def split_str(split):
    return "[" + ", ".join(str(n) for n in split) + "]"


def run_dataset(dataset, config):
    """(prediction rows, pair rows) for one sweep CSV."""
    rows, slice_gb = load_rows(dataset["csv"])
    cfg = dict(model=dataset["model"], slice_gb=slice_gb, card=dataset["card"], seq_len=config["seq_len"],
               max_new_tokens=config["max_new_tokens"], samples=config["samples"],
               min_last_rank_layers=dataset["min_last_rank_layers"],
               constants=hardware_constants.CONSTANTS[config["constants"]])
    pairs = {}
    for r in rows:
        pairs.setdefault((r["batch"], r["mb"]), []).append(r)

    prediction_rows, pair_rows = [], []
    for (batch, mb), group in sorted(pairs.items()):
        n = batch // mb
        steps = graph.prepare(cfg, batch, mb, config["samples"])

        measured = {}  # split -> fastest measured ms (a split may be repeated)
        for r in group:
            measured[r["split"]] = min(measured.get(r["split"], float("inf")), r["measured_ms"])
        pred_graph = {s: graph.evaluate(list(s), steps, cfg) for s in measured}
        pred_gpipe = {s: gpipe.evaluate(list(s), steps, n, cfg) for s in measured}

        for r in group:
            s = r["split"]
            prediction_rows.append(dict(
                dataset=dataset["name"], setup=dataset["setup"], model=dataset["model"], split=split_str(s),
                batch=batch, microbatch=mb, num_microbatches=n, measured_ms=round(r["measured_ms"], 1),
                graph_pred_ms=round(pred_graph[s], 1), graph_err_pct=round(pct(pred_graph[s], r["measured_ms"]), 2),
                gpipe_pred_ms=round(pred_gpipe[s], 1), gpipe_err_pct=round(pct(pred_gpipe[s], r["measured_ms"]), 2)))

        order = sorted(measured, key=lambda s: (measured[s], s))
        best = measured[order[0]]
        fav_graph, tied_graph = favourite(pred_graph)
        fav_gpipe, tied_gpipe = favourite(pred_gpipe)
        # a tied favourite is scored by the average over its tied splits (= a random pick among them)
        fav_graph_pct = statistics.mean(pct(measured[s], best) for s in tied_graph)
        fav_gpipe_pct = statistics.mean(pct(measured[s], best) for s in tied_gpipe)
        gap = fav_gpipe_pct - fav_graph_pct
        winner = "graph" if gap > config["tie_pct"] else "gpipe" if gap < -config["tie_pct"] else "tie"

        free_graph = tuple(graph.optimize_placement(cfg, batch, mb)[0])
        # GPipe's free pick by enumeration, so its frequent exact ties are resolved the same
        # way every run (lowest split in sort order) and counted
        lower, upper = graph.layer_bounds(cfg, batch, mb)
        every = {tuple(s): gpipe.evaluate(s, steps, n, cfg)
                 for s in graph.all_splits(graph.memory_limits.REFERENCE_SHAPES[cfg["model"]]["L"], lower, upper)}
        free_gpipe, free_gpipe_tied = favourite(every)

        pair_rows.append(dict(
            dataset=dataset["name"], setup=dataset["setup"], model=dataset["model"], batch=batch, microbatch=mb,
            num_microbatches=n, measured_splits=len(measured),
            best_split=split_str(order[0]), best_ms=round(best, 1),
            typical_split_slower_pct=round(statistics.median(pct(v, best) for v in measured.values()), 2),
            worst_split_slower_pct=round(pct(measured[order[-1]], best), 2),
            graph_fav_split=split_str(fav_graph), graph_fav_tied_splits=len(tied_graph),
            graph_fav_slower_pct=round(fav_graph_pct, 2),
            gpipe_fav_split=split_str(fav_gpipe), gpipe_fav_tied_splits=len(tied_gpipe),
            gpipe_fav_slower_pct=round(fav_gpipe_pct, 2),
            faster_favourite=winner,
            graph_free_split=split_str(free_graph), graph_free_was_run=free_graph in measured,
            graph_free_slower_pct=round(pct(measured[free_graph], best), 2) if free_graph in measured else "",
            gpipe_free_split=split_str(free_gpipe), gpipe_free_tied_splits=len(free_gpipe_tied),
            gpipe_free_was_run=free_gpipe in measured,
            gpipe_free_slower_pct=round(pct(measured[free_gpipe], best), 2) if free_gpipe in measured else "",
            graph_kendall_tau_b=round(kendall_tau_b([pred_graph[s] for s in order], [measured[s] for s in order]), 3),
            gpipe_kendall_tau_b=round(kendall_tau_b([pred_gpipe[s] for s in order], [measured[s] for s in order]), 3)))
        print(f"  {dataset['name']} B{batch} mb{mb}: best {split_str(order[0])} | graph {split_str(fav_graph)} "
              f"| gpipe {split_str(fav_gpipe)} -> {winner}")
    return prediction_rows, pair_rows


def summarize(group_name, group_value, pairs, predictions):
    """One summary row over a set of pair rows and their prediction rows."""
    def med(rows, key):
        values = [r[key] for r in rows if r[key] != ""]
        return round(statistics.median(values), 2) if values else ""

    def worst(rows, key):
        values = [r[key] for r in rows if r[key] != ""]
        return round(max(values), 2) if values else ""

    return dict(
        group=group_name, value=group_value, settings=len(pairs), measured_runs=len(predictions),
        graph_err_median_pct=med(predictions, "graph_err_pct"), gpipe_err_median_pct=med(predictions, "gpipe_err_pct"),
        graph_fav_slower_median_pct=med(pairs, "graph_fav_slower_pct"), graph_fav_slower_worst_pct=worst(pairs, "graph_fav_slower_pct"),
        gpipe_fav_slower_median_pct=med(pairs, "gpipe_fav_slower_pct"), gpipe_fav_slower_worst_pct=worst(pairs, "gpipe_fav_slower_pct"),
        typical_split_slower_median_pct=med(pairs, "typical_split_slower_pct"),
        graph_faster=sum(p["faster_favourite"] == "graph" for p in pairs),
        gpipe_faster=sum(p["faster_favourite"] == "gpipe" for p in pairs),
        tie=sum(p["faster_favourite"] == "tie" for p in pairs),
        graph_free_picks_run=sum(p["graph_free_was_run"] for p in pairs),
        graph_free_slower_median_pct=med(pairs, "graph_free_slower_pct"), graph_free_slower_worst_pct=worst(pairs, "graph_free_slower_pct"),
        gpipe_free_picks_run=sum(p["gpipe_free_was_run"] for p in pairs),
        gpipe_free_slower_median_pct=med(pairs, "gpipe_free_slower_pct"), gpipe_free_slower_worst_pct=worst(pairs, "gpipe_free_slower_pct"),
        gpipe_fav_tied_settings=sum(p["gpipe_fav_tied_splits"] > 1 for p in pairs),
        graph_fav_tied_settings=sum(p["graph_fav_tied_splits"] > 1 for p in pairs),
        graph_kendall_tau_b_median=med(pairs, "graph_kendall_tau_b"), gpipe_kendall_tau_b_median=med(pairs, "gpipe_kendall_tau_b"))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def main():
    intra, inter = "oracle_v2/intra_gpu/80gb/40_20_10_10/", "oracle_v2/inter_gpu/40gb/20_10_5_5/"
    CONFIG = dict(
        datasets=[  # name, setup, csv, model key, card, min layers on the last rank (as swept)
            dict(name="intra_vicuna_13b", setup="intra", csv=intra + "vicuna/vicuna_13b.csv", model="vicuna_13b", card="A100-80GB", min_last_rank_layers=1),
            dict(name="intra_llama_13b", setup="intra", csv=intra + "llama/llama_13b.csv", model="llama_13b", card="A100-80GB", min_last_rank_layers=1),
            dict(name="intra_qwen_14b", setup="intra", csv=intra + "qwen/qwen_14b.csv", model="qwen_14b", card="A100-80GB", min_last_rank_layers=1),
            dict(name="intra_mistral_24b", setup="intra", csv=intra + "mistral/mistral_24b.csv", model="mistral_24b", card="A100-80GB", min_last_rank_layers=1),
            dict(name="inter_vicuna_7b", setup="inter", csv=inter + "vicuna/vicuna_7b.csv", model="vicuna_7b", card="A100-40GB", min_last_rank_layers=1),
            dict(name="inter_mistral_7b", setup="inter", csv=inter + "mistral/mistral7b.csv", model="mistral_7b", card="A100-40GB", min_last_rank_layers=1),
            dict(name="inter_qwen_7b", setup="inter", csv=inter + "qwen/qwen7b.csv", model="qwen_7b", card="A100-40GB", min_last_rank_layers=0),
            dict(name="inter_vicuna_13b", setup="inter", csv=inter + "vicuna/vicuna13b.csv", model="vicuna_13b", card="A100-40GB", min_last_rank_layers=1),
            dict(name="inter_qwen_14b", setup="inter", csv=inter + "qwen/qwen14b.csv", model="qwen_14b", card="A100-40GB", min_last_rank_layers=0),
        ],
        seq_len=64,
        max_new_tokens=512,
        constants="a100_80gb_pcie_whole_gpu",  # name of a set in hardware_constants.CONSTANTS
        samples=8,  # decode contexts per run (None = every step: exact but slow; differs < 0.01%)
        tie_pct=0.5,  # favourites within this many % of each other count as a tie
        out_dir="graph_vs_gpipe_results",
    )

    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    predictions, pairs = [], []
    for dataset in CONFIG["datasets"]:
        print(f"{dataset['name']}: {dataset['csv']}")
        p, q = run_dataset(dataset, CONFIG)
        predictions += p
        pairs += q

    summary = [summarize("all", "all", pairs, predictions)]
    for key in ("setup", "dataset"):
        for value in dict.fromkeys(p[key] for p in pairs):
            summary.append(summarize(key, value, [p for p in pairs if p[key] == value],
                                     [r for r in predictions if r[key] == value]))
    for setup in ("intra", "inter"):
        for n in sorted({p["num_microbatches"] for p in pairs if p["setup"] == setup}):
            summary.append(summarize(f"{setup}_num_microbatches", n,
                                     [p for p in pairs if p["setup"] == setup and p["num_microbatches"] == n],
                                     [r for r in predictions if r["setup"] == setup and r["num_microbatches"] == n]))

    write_csv(os.path.join(CONFIG["out_dir"], "predictions.csv"), predictions)
    write_csv(os.path.join(CONFIG["out_dir"], "pairs.csv"), pairs)
    write_csv(os.path.join(CONFIG["out_dir"], "summary.csv"), summary)


if __name__ == "__main__":
    main()
