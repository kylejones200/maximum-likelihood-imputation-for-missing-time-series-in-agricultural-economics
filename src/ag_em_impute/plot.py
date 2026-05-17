from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_yield_imputation(
    data: pd.DataFrame,
    output_path: Path,
    *,
    true_yields: pd.Series | None = None,
    dpi: int = 300,
    show: bool = False,
) -> None:
    missing = data["Yield"].isna()
    observed = ~missing

    fig, ax = plt.subplots()
    if observed.any():
        ax.plot(
            data.loc[observed, "Year"],
            data.loc[observed, "Yield"],
            "o-",
            color="tab:blue",
            label="Observed",
        )
    if missing.any():
        ax.plot(
            data.loc[missing, "Year"],
            data.loc[missing, "Estimated_Yield"],
            "s",
            color="tab:orange",
            markersize=7,
            linestyle="none",
            label="Imputed (missing years)",
        )
    ax.plot(
        data["Year"],
        data["Estimated_Yield"],
        "--",
        color="tab:red",
        alpha=0.8,
        label="EM estimate (full series)",
    )
    if true_yields is not None:
        aligned = true_yields.reindex(data["Year"])
        ax.plot(
            data["Year"],
            aligned.to_numpy(),
            ":",
            color="tab:gray",
            alpha=0.7,
            label="True (validation)",
        )

    ax.set_title(
        "Crop Yield Estimation Using Expectation-Maximization (tons per hectare)",
        fontsize=12,
    )
    ax.legend()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.tick_params(axis="both", direction="out")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
