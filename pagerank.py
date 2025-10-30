import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# 读取数据集
df = pd.read_csv('dataset/PageRank_Dataset.csv')

# 创建有向图
G = nx.DiGraph()

# 添加边到图中
for _, row in df.iterrows():
    G.add_edge(row['node_1'], row['node_2'])

# 使用NetworkX的PageRank算法计算每个节点的PageRank值
pagerank_scores = nx.pagerank(G, alpha=0.85)

# 将PageRank值按从高到低排序
sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

# 输出前20个PageRank值最高的节点
top_20_nodes = sorted_pagerank[:20]
print("Top 20 nodes with highestPageRankvalues:\n")
for node, score in top_20_nodes:
    print(f"Node {node}: {score}")

os.makedirs('results', exist_ok=True)

# 将结果保存到CSV文件
pagerank_df = pd.DataFrame(sorted_pagerank, columns=['NodeID', 'PageRank'])
pagerank_df.to_csv('results/pagerank_results.csv', index=False)

# Optional: 不同阻尼系数下的PageRank计算
betas = [0.7, 0.75, 0.8, 0.85, 0.9]
for beta in betas:
    pagerank_scores = nx.pagerank(G, alpha=beta)
    sorted_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    with open(f'results/pagerank_results_beta_{beta}.csv', 'w') as f:
        f.write(f"Top 20 nodes with highestPageRankvalues for beta={beta}:\n")
        f.write('NodeID,PageRank\n')
        for node, score in sorted_pagerank[:20]:
            f.write(f'{node},{score}\n')

# 提取节点和对应的PageRank值
nodes = list(pagerank_scores.keys())
values = list(pagerank_scores.values())

# Optional: 绘制PageRank值分布
plt.bar(nodes, values)
plt.xlabel('Node')
plt.ylabel('PageRank Value')
plt.title('PageRank Values of Nodes')
plt.xticks(rotation=90)
plt.savefig('results/pagerank_distribution.png', dpi=1024)