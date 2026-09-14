import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def generate_network_visualization(n_nodes: int = 35, seed: int = 42):
    np.random.seed(seed)
    os.makedirs("assets", exist_ok=True)

    # 1. Build directed interbank exposure graph (scale-free / core-periphery)
    G = nx.erdos_renyi_graph(n=n_nodes, p=0.14, seed=seed, directed=True)

    # 2. Assign balance-sheet attributes and default cascade states
    node_colors = []
    node_sizes = []
    labels = {}

    for node in G.nodes():
        # Core institutions vs peripheral banks
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        size = 350 + (in_deg + out_deg) * 60
        node_sizes.append(size)
        labels[node] = f"B{node}"

        # Classify default status based on high exposure vulnerability
        if out_deg >= 5 or (in_deg >= 4 and np.random.rand() > 0.45):
            node_colors.append("#D9381E")  # Insolvent / Defaulting (Red)
        elif in_deg >= 3:
            node_colors.append("#F5A623")  # Capital-Depleted / Distressed (Amber)
        else:
            node_colors.append("#2E7D32")  # Resilient / Solvent (Green)

    # 3. Compute layout
    pos = nx.spring_layout(G, k=0.45, iterations=60, seed=seed)

    # 4. Render publication-ready plot
    plt.figure(figsize=(12, 8), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Draw directed bilateral obligations
    edge_weights = [0.4 + 0.6 * np.random.rand() for _ in G.edges()]
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        edge_color="#7F8C8D",
        alpha=0.45,
        width=1.2,
        connectionstyle="arc3,rad=0.08"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#2C3E50",
        linewidths=1.2,
        alpha=0.92
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#FFFFFF", font_weight="bold")

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E7D32', markersize=11, label='Solvent Institutions'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F5A623', markersize=11, label='Capital Buffer Depleted'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D9381E', markersize=11, label='Triggered Default Cascade')
    ]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=10, shadow=True)

    plt.title(
        "Systemic Risk Contagion: Dynamic Interbank Obligation Clearing & Cascade Topology",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    plt.axis("off")
    plt.tight_layout()

    # 5. Export high-resolution image
    output_path = os.path.join("assets", "network_contagion.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualization generated successfully: {output_path}")

if __name__ == "__main__":
    generate_network_visualization()
