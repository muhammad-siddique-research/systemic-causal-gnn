"""Generate a publication-quality interbank contagion network figure.

This script builds a stylized interbank network where banks are classified as
solvent, distressed, or defaulting. It uses NetworkX for graph construction and
Matplotlib for publication-style rendering, then saves the figure to the
assets directory.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


ASSET_PATH = Path(__file__).resolve().parents[1] / "assets"
OUTPUT_PATH = ASSET_PATH / "network_contagion.png"


def build_network():
    """Construct a stylized interbank contagion graph."""
    G = nx.Graph()

    positions = {
        "Bank A": (0.00, 1.00),
        "Bank B": (-0.82, 0.78),
        "Bank C": (0.82, 0.78),
        "Bank D": (-1.24, 0.20),
        "Bank E": (0.00, 0.20),
        "Bank F": (1.24, 0.20),
        "Bank G": (-0.82, -0.38),
        "Bank H": (0.00, -0.38),
        "Bank I": (0.82, -0.38),
        "Bank J": (-1.05, -1.00),
        "Bank K": (0.00, -1.00),
        "Bank L": (1.05, -1.00),
    }

    status_map = {
        "Bank A": "solvent",
        "Bank B": "solvent",
        "Bank C": "distressed",
        "Bank D": "solvent",
        "Bank E": "defaulting",
        "Bank F": "distressed",
        "Bank G": "solvent",
        "Bank H": "defaulting",
        "Bank I": "distressed",
        "Bank J": "solvent",
        "Bank K": "solvent",
        "Bank L": "distressed",
    }

    for bank, pos in positions.items():
        G.add_node(bank, pos=pos, status=status_map[bank])

    edge_weights = {
        ("Bank A", "Bank B"): 0.82,
        ("Bank A", "Bank C"): 0.70,
        ("Bank A", "Bank D"): 0.64,
        ("Bank B", "Bank E"): 0.90,
        ("Bank C", "Bank F"): 0.85,
        ("Bank D", "Bank G"): 0.73,
        ("Bank E", "Bank H"): 0.95,
        ("Bank E", "Bank I"): 0.88,
        ("Bank F", "Bank I"): 0.66,
        ("Bank G", "Bank J"): 0.62,
        ("Bank H", "Bank K"): 0.67,
        ("Bank I", "Bank L"): 0.71,
        ("Bank J", "Bank K"): 0.59,
        ("Bank K", "Bank L"): 0.63,
        ("Bank D", "Bank E"): 0.48,
        ("Bank F", "Bank K"): 0.46,
        ("Bank G", "Bank H"): 0.52,
    }

    for (u, v), weight in edge_weights.items():
        G.add_edge(u, v, weight=weight)

    return G, positions


def draw_figure(G, positions):
    """Render the contagion network with publication-quality styling."""
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    status_colors = {
        "solvent": "#2E7D32",
        "distressed": "#F9A825",
        "defaulting": "#C62828",
    }

    node_sizes = {
        "solvent": 550,
        "distressed": 700,
        "defaulting": 850,
    }

    node_labels = {node: node for node in G.nodes}
    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    edge_widths = [1.5 + w * 4.0 for w in edge_weights]
    edge_colors = ["#536878" for _ in edge_weights]

    nx.draw_networkx_edges(
        G,
        pos=positions,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        ax=ax,
    )

    for status in ["solvent", "distressed", "defaulting"]:
        nodes = [n for n, d in G.nodes(data=True) if d["status"] == status]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G,
            pos=positions,
            nodelist=nodes,
            node_color=status_colors[status],
            node_size=node_sizes[status],
            edgecolors="white",
            linewidths=1.5,
            alpha=0.96,
            ax=ax,
        )

    nx.draw_networkx_labels(
        G,
        pos=positions,
        labels=node_labels,
        font_size=9,
        font_family="sans-serif",
        font_weight="bold",
        ax=ax,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=12,
            markerfacecolor=status_colors[status],
            markeredgecolor="white",
            markeredgewidth=1.5,
            label=status.title(),
        )
        for status in ["solvent", "distressed", "defaulting"]
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d7de",
        fontsize=10,
        title="Bank status",
        title_fontsize=11,
    )
    legend.get_frame().set_alpha(0.95)

    ax.set_title(
        "Interbank Risk Contagion Network Topology",
        fontsize=18,
        fontweight="semibold",
        pad=20,
    )
    ax.set_xlabel("Financial exposure network", fontsize=11, color="#2d3748")
    ax.set_axis_off()

    return fig, ax


def main():
    """Build the network and save the resulting figure."""
    ASSET_PATH.mkdir(parents=True, exist_ok=True)
    G, positions = build_network()
    fig, _ = draw_figure(G, positions)

    output = OUTPUT_PATH
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    print(f"Network figure saved to: {output}")


if __name__ == "__main__":
    main()
 d118745 (Add contagion topology figure)
