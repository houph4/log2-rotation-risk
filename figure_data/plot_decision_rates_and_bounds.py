"""Plot the PR theory and interval figure from the frozen numerical summaries."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter


def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("decision_rates_and_bounds.pdf"))
    args = parser.parse_args()
    data = Path(__file__).resolve().parent
    witnesses = read_csv(data / "theory_witness_rates.csv")
    finite = read_csv(data / "finite_radius_summary.csv")

    plt.rcParams.update({
        "font.family": "Cambria",
        "font.size": 9.5,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8B8C8F",
        "legend.fontsize": 8,
        "text.color": "#26323B",
        "axes.labelcolor": "#26323B",
    })
    purple, slate = "#B23A48", "#516D87"
    fig, (gap_axis, error_axis, resolution_axis) = plt.subplots(
        1, 3, figsize=(8.8, 2.65), layout="constrained"
    )
    for order, label, color, marker in [(6, "Matrix9", slate, "s"), (10, "Log2", purple, "o")]:
        rows = sorted((row for row in witnesses if int(row["exponent"]) == order),
                      key=lambda row: float(row["radius_deg"]))
        gap_axis.loglog([float(row["radius_deg"]) for row in rows],
                        [float(row["gap"]) for row in rows],
                        marker=marker, color=color, lw=1.25, ms=3.5, label=label)
    gap_axis.set_ylabel("Common-action Bayes gap (rad²)")
    gap_axis.legend(frameon=False, loc="upper left")

    for regime, color, marker in [("local", purple, "o"), ("wide", slate, "s")]:
        rows = sorted((row for row in finite if row["regime"] == regime),
                      key=lambda row: float(row["radius_deg"]))
        radii = [float(row["radius_deg"]) for row in rows]
        error_axis.loglog(radii, [float(row["error"]) for row in rows],
                          marker=marker, color=color, lw=1.2, ms=3.5)
        error_axis.loglog(radii, [float(row["allowance"]) for row in rows],
                          linestyle="--", color=color, lw=1.25)
        resolution_axis.semilogx(radii, [100*float(row["resolution"]) for row in rows],
                                marker=marker, color=color, lw=1.25, ms=3.5,
                                label=regime.capitalize())

    error_axis.set_ylabel("Median error and allowance (rad²)")
    error_axis.legend(
        handles=[Line2D([0], [0], color="#48515A", linestyle="-", label="Absolute error"),
                 Line2D([0], [0], color="#48515A", linestyle="--", label="Allowance")],
        frameon=False, loc="upper left",
    )
    resolution_axis.set_ylabel("Resolved comparisons (%)")
    resolution_axis.set_ylim(94, 100.7)
    resolution_axis.set_yticks([94, 96, 98, 100])
    resolution_axis.legend(frameon=False, loc="lower left")
    for axis, tag, ticks in zip((gap_axis,error_axis,resolution_axis),("a","b","c"),
                               ([.25,1,5,29],[.5,5,29],[.5,5,29])):
        axis.set_xlabel("Support radius (degrees)")
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.grid(axis="y", alpha=.18, linewidth=.6)
        axis.tick_params(length=3, width=.6)
        axis.text(0, 1.03, "("+tag+")", transform=axis.transAxes,
                  fontsize=10, fontweight="bold", ha="left", va="bottom")
    fig.savefig(args.output, bbox_inches="tight", pad_inches=.055)
    plt.close(fig)


if __name__ == "__main__":
    main()
