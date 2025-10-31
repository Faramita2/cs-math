import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset/PageRank_Dataset.csv')

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row['node_1'], row['node_2'])

# Compute PageRank values with damping factor alpha=0.85
pagerank_scores = nx.pagerank(G, alpha=0.85)

# Sort PageRank scores in descending order
sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# Print top 20 nodes with highest PageRank values
top_20_nodes = sorted_pagerank[:20]
print("Top 20 nodes with highest PageRank values:\n")
for node, score in top_20_nodes:
    print(f"Node {node}: {score}")

# Create directory to save results
results_base_path = 'results/pagerank'
os.makedirs(results_base_path, exist_ok=True)

# Save full PageRank results to a CSV file
pagerank_df = pd.DataFrame(sorted_pagerank, columns=['NodeID', 'PageRank'])
pagerank_df.to_csv(f'{results_base_path}/pagerank_results.csv', index=False)

# Optional: Compute PageRank with different damping values
betas = [0.7, 0.75, 0.8, 0.85, 0.9]
for beta in betas:
    pagerank_scores = nx.pagerank(G, alpha=beta)
    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    with open(f'{results_base_path}/pagerank_results_beta_{beta}.csv', 'w') as f:
        f.write(f"Top 20 nodes with highest PageRank values for beta={beta}:\n")
        f.write('NodeID,PageRank\n')
        for node, score in sorted_pagerank[:20]:
            f.write(f'{node},{score}\n')

# Extract node IDs and PageRank scores for visualization
nodes = list(pagerank_scores.keys())
values = list(pagerank_scores.values())

# Plot PageRank value distribution
plt.bar(nodes, values)
plt.xlabel('Node')
plt.ylabel('PageRank Value')
plt.title('PageRank Values Distribution')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{results_base_path}/pagerank_distribution.png', dpi=1024)
plt.close()