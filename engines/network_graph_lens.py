"""
Network / Graph Theory Lens
===========================
Analyzes relationships between variables as a network structure,
revealing hidden connections, clusters, and influence pathways.

Key Applications:
- Financial contagion and systemic risk analysis
- Climate teleconnection networks
- Market sector interdependencies

Methods:
- Minimum Spanning Tree (MST): Core structure extraction
- Centrality Measures: Node importance quantification
- Community Detection: Cluster identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from collections import defaultdict
import heapq


@dataclass
class NetworkNode:
    """Represents a node in the network."""
    id: str
    centrality_scores: Dict[str, float]
    neighbors: List[str]
    cluster: Optional[int] = None


@dataclass
class NetworkEdge:
    """Represents an edge in the network."""
    source: str
    target: str
    weight: float
    correlation: float
    distance: float


class GraphTheoryLens:
    """
    Network/Graph Theory Lens for analyzing relationships between time series.
    
    Transforms correlation/dependency matrices into network structures,
    revealing:
    - Core dependencies (MST backbone)
    - Most influential variables (centrality)
    - Natural groupings (communities)
    - Systemic risk pathways
    """
    
    def __init__(self, distance_method: str = 'correlation'):
        """
        Initialize the Graph Theory Lens.
        
        Parameters
        ----------
        distance_method : str
            Method to convert correlations to distances:
            'correlation': d = sqrt(2(1-r))
            'absolute': d = sqrt(2(1-|r|))
            'information': d = sqrt(1 - MI/max(MI))
        """
        self.distance_method = distance_method
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self.adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.mst_edges: List[NetworkEdge] = []
    
    def _correlation_to_distance(self, corr: float) -> float:
        """Convert correlation to distance metric."""
        if self.distance_method == 'correlation':
            return np.sqrt(2 * (1 - corr))
        elif self.distance_method == 'absolute':
            return np.sqrt(2 * (1 - abs(corr)))
        else:
            return np.sqrt(2 * (1 - corr))
    
    def build_network_from_correlation(
        self,
        correlation_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> None:
        """
        Build network from correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Square correlation matrix
        labels : list, optional
            Node labels (variable names)
        threshold : float, optional
            Minimum |correlation| to include edge
        """
        n = correlation_matrix.shape[0]
        
        if labels is None:
            labels = [f"V{i}" for i in range(n)]
        
        # Initialize nodes
        for label in labels:
            self.nodes[label] = NetworkNode(
                id=label,
                centrality_scores={},
                neighbors=[]
            )
        
        # Build edges
        self.edges = []
        self.adjacency = defaultdict(dict)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                
                if threshold is not None and abs(corr) < threshold:
                    continue
                
                distance = self._correlation_to_distance(corr)
                
                edge = NetworkEdge(
                    source=labels[i],
                    target=labels[j],
                    weight=1 / (distance + 1e-10),  # Higher weight = stronger connection
                    correlation=corr,
                    distance=distance
                )
                
                self.edges.append(edge)
                self.adjacency[labels[i]][labels[j]] = distance
                self.adjacency[labels[j]][labels[i]] = distance
                
                self.nodes[labels[i]].neighbors.append(labels[j])
                self.nodes[labels[j]].neighbors.append(labels[i])
    
    def build_network_from_timeseries(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        threshold: Optional[float] = None
    ) -> None:
        """
        Build network directly from time series data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Time series data (rows=time, columns=variables)
        method : str
            Correlation method: 'pearson', 'spearman', 'kendall'
        threshold : float, optional
            Minimum |correlation| to include edge
        """
        correlation_matrix = data.corr(method=method).values
        labels = data.columns.tolist()
        self.build_network_from_correlation(correlation_matrix, labels, threshold)
    
    def compute_minimum_spanning_tree(self) -> List[NetworkEdge]:
        """
        Compute Minimum Spanning Tree using Kruskal's algorithm.
        
        The MST reveals the backbone of the network - the minimum set
        of edges that connect all nodes with minimum total distance.
        
        Returns
        -------
        list
            MST edges
        """
        if not self.edges:
            raise ValueError("Network has no edges. Build network first.")
        
        # Sort edges by distance
        sorted_edges = sorted(self.edges, key=lambda e: e.distance)
        
        # Union-Find data structure for cycle detection
        parent = {node: node for node in self.nodes}
        rank = {node: 0 for node in self.nodes}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Kruskal's algorithm
        self.mst_edges = []
        for edge in sorted_edges:
            if union(edge.source, edge.target):
                self.mst_edges.append(edge)
                if len(self.mst_edges) == len(self.nodes) - 1:
                    break
        
        return self.mst_edges
    
    def compute_degree_centrality(self) -> Dict[str, float]:
        """
        Compute degree centrality for all nodes.
        
        Degree centrality = number of connections / (n-1)
        Higher values indicate more connected nodes.
        """
        n = len(self.nodes)
        centrality = {}
        
        for node_id, node in self.nodes.items():
            centrality[node_id] = len(node.neighbors) / (n - 1) if n > 1 else 0
            node.centrality_scores['degree'] = centrality[node_id]
        
        return centrality
    
    def compute_betweenness_centrality(self) -> Dict[str, float]:
        """
        Compute betweenness centrality for all nodes.
        
        Betweenness = fraction of shortest paths passing through node.
        High betweenness nodes are critical for information flow.
        """
        nodes = list(self.nodes.keys())
        n = len(nodes)
        betweenness = {node: 0.0 for node in nodes}
        
        # For each source node
        for source in nodes:
            # BFS/Dijkstra for shortest paths
            distances = {node: float('inf') for node in nodes}
            distances[source] = 0
            num_paths = {node: 0 for node in nodes}
            num_paths[source] = 1
            predecessors = {node: [] for node in nodes}
            
            # Priority queue: (distance, node)
            pq = [(0, source)]
            visited = set()
            
            while pq:
                dist, current = heapq.heappop(pq)
                
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor, edge_dist in self.adjacency[current].items():
                    new_dist = dist + edge_dist
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        num_paths[neighbor] = num_paths[current]
                        predecessors[neighbor] = [current]
                        heapq.heappush(pq, (new_dist, neighbor))
                    elif abs(new_dist - distances[neighbor]) < 1e-10:
                        num_paths[neighbor] += num_paths[current]
                        predecessors[neighbor].append(current)
            
            # Accumulate betweenness
            dependency = {node: 0.0 for node in nodes}
            
            # Process nodes in reverse order of distance
            sorted_nodes = sorted(nodes, key=lambda x: -distances[x])
            
            for node in sorted_nodes:
                if node == source:
                    continue
                for pred in predecessors[node]:
                    if num_paths[node] > 0:
                        dependency[pred] += (num_paths[pred] / num_paths[node]) * (1 + dependency[node])
                if node != source:
                    betweenness[node] += dependency[node]
        
        # Normalize
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            betweenness = {k: v * norm for k, v in betweenness.items()}
        
        for node_id in self.nodes:
            self.nodes[node_id].centrality_scores['betweenness'] = betweenness[node_id]
        
        return betweenness
    
    def compute_closeness_centrality(self) -> Dict[str, float]:
        """
        Compute closeness centrality for all nodes.
        
        Closeness = (n-1) / sum(shortest path distances)
        High closeness nodes can reach all others quickly.
        """
        nodes = list(self.nodes.keys())
        n = len(nodes)
        closeness = {}
        
        for source in nodes:
            # Dijkstra's algorithm
            distances = {node: float('inf') for node in nodes}
            distances[source] = 0
            pq = [(0, source)]
            visited = set()
            
            while pq:
                dist, current = heapq.heappop(pq)
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor, edge_dist in self.adjacency[current].items():
                    new_dist = dist + edge_dist
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
            
            # Compute closeness
            reachable = [d for d in distances.values() if d < float('inf') and d > 0]
            if reachable:
                closeness[source] = (len(reachable)) / sum(reachable)
            else:
                closeness[source] = 0
            
            self.nodes[source].centrality_scores['closeness'] = closeness[source]
        
        return closeness
    
    def compute_eigenvector_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute eigenvector centrality using power iteration.
        
        Eigenvector centrality: connected to important nodes makes you important.
        """
        nodes = list(self.nodes.keys())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build adjacency matrix (using weights)
        adj_matrix = np.zeros((n, n))
        for edge in self.edges:
            i, j = node_idx[edge.source], node_idx[edge.target]
            adj_matrix[i, j] = edge.weight
            adj_matrix[j, i] = edge.weight
        
        # Power iteration
        centrality = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            new_centrality = adj_matrix @ centrality
            norm = np.linalg.norm(new_centrality)
            if norm > 0:
                new_centrality /= norm
            
            if np.linalg.norm(new_centrality - centrality) < tol:
                break
            centrality = new_centrality
        
        # Normalize to [0, 1]
        if np.max(centrality) > 0:
            centrality = centrality / np.max(centrality)
        
        result = {nodes[i]: centrality[i] for i in range(n)}
        
        for node_id in self.nodes:
            self.nodes[node_id].centrality_scores['eigenvector'] = result[node_id]
        
        return result
    
    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute PageRank centrality.
        
        Similar to eigenvector centrality but with damping factor.
        """
        nodes = list(self.nodes.keys())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build transition matrix
        adj_matrix = np.zeros((n, n))
        for edge in self.edges:
            i, j = node_idx[edge.source], node_idx[edge.target]
            adj_matrix[i, j] = edge.weight
            adj_matrix[j, i] = edge.weight
        
        # Normalize rows
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition = adj_matrix / row_sums
        
        # Power iteration with damping
        pagerank = np.ones(n) / n
        
        for _ in range(max_iter):
            new_pagerank = (1 - damping) / n + damping * (transition.T @ pagerank)
            
            if np.linalg.norm(new_pagerank - pagerank) < tol:
                break
            pagerank = new_pagerank
        
        result = {nodes[i]: pagerank[i] for i in range(n)}
        
        for node_id in self.nodes:
            self.nodes[node_id].centrality_scores['pagerank'] = result[node_id]
        
        return result
    
    def detect_communities_louvain(self) -> Dict[str, int]:
        """
        Detect communities using a simplified Louvain-style algorithm.
        
        Communities are groups of densely connected nodes,
        useful for identifying natural clusters.
        """
        nodes = list(self.nodes.keys())
        n = len(nodes)
        
        # Initialize each node in its own community
        community = {node: i for i, node in enumerate(nodes)}
        
        # Build adjacency with weights
        total_weight = sum(edge.weight for edge in self.edges) * 2
        
        def modularity_gain(node, target_comm, node_degree):
            """Calculate modularity gain of moving node to target community."""
            sum_in = 0
            sum_tot = 0
            
            for neighbor, dist in self.adjacency[node].items():
                weight = 1 / (dist + 1e-10)
                if community[neighbor] == target_comm:
                    sum_in += weight
                if community[neighbor] == target_comm:
                    sum_tot += sum(1 / (d + 1e-10) for d in self.adjacency[neighbor].values())
            
            if total_weight == 0:
                return 0
            
            return (sum_in / total_weight - 
                    (sum_tot * node_degree) / (total_weight ** 2))
        
        # Iterate until no improvement
        improved = True
        iterations = 0
        max_iterations = 10
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for node in nodes:
                node_degree = sum(1 / (d + 1e-10) for d in self.adjacency[node].values())
                current_comm = community[node]
                
                # Find best community
                best_comm = current_comm
                best_gain = 0
                
                neighbor_comms = set(community[n] for n in self.adjacency[node])
                neighbor_comms.add(current_comm)
                
                for comm in neighbor_comms:
                    if comm != current_comm:
                        gain = modularity_gain(node, comm, node_degree)
                        if gain > best_gain:
                            best_gain = gain
                            best_comm = comm
                
                if best_comm != current_comm:
                    community[node] = best_comm
                    improved = True
        
        # Renumber communities
        unique_comms = sorted(set(community.values()))
        comm_map = {c: i for i, c in enumerate(unique_comms)}
        community = {k: comm_map[v] for k, v in community.items()}
        
        for node_id in self.nodes:
            self.nodes[node_id].cluster = community[node_id]
        
        return community
    
    def compute_network_metrics(self) -> Dict:
        """
        Compute global network metrics.
        
        Returns
        -------
        dict
            density, avg_clustering, avg_path_length, diameter
        """
        n = len(self.nodes)
        num_edges = len(self.edges)
        max_edges = n * (n - 1) / 2
        
        # Density
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Average clustering coefficient
        clustering_coeffs = []
        for node_id, node in self.nodes.items():
            neighbors = node.neighbors
            k = len(neighbors)
            if k < 2:
                clustering_coeffs.append(0)
                continue
            
            # Count edges between neighbors
            neighbor_edges = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n2 in self.adjacency[n1]:
                        neighbor_edges += 1
            
            max_neighbor_edges = k * (k - 1) / 2
            clustering_coeffs.append(neighbor_edges / max_neighbor_edges if max_neighbor_edges > 0 else 0)
        
        avg_clustering = np.mean(clustering_coeffs)
        
        # Average path length (from MST if computed)
        if self.mst_edges:
            mst_total_distance = sum(e.distance for e in self.mst_edges)
            avg_path_length = mst_total_distance / len(self.mst_edges) if self.mst_edges else 0
        else:
            avg_path_length = None
        
        return {
            'num_nodes': n,
            'num_edges': num_edges,
            'density': density,
            'avg_clustering_coefficient': avg_clustering,
            'avg_mst_edge_length': avg_path_length
        }
    
    def identify_critical_nodes(self, top_k: int = 5) -> Dict[str, List[str]]:
        """
        Identify critical nodes by various centrality measures.
        
        Parameters
        ----------
        top_k : int
            Number of top nodes to return per metric
            
        Returns
        -------
        dict
            Top nodes for each centrality measure
        """
        # Ensure all centralities are computed
        if not self.nodes[list(self.nodes.keys())[0]].centrality_scores:
            self.compute_degree_centrality()
            self.compute_betweenness_centrality()
            self.compute_closeness_centrality()
            self.compute_eigenvector_centrality()
        
        critical = {}
        metrics = ['degree', 'betweenness', 'closeness', 'eigenvector']
        
        for metric in metrics:
            sorted_nodes = sorted(
                self.nodes.items(),
                key=lambda x: x[1].centrality_scores.get(metric, 0),
                reverse=True
            )
            critical[metric] = [node_id for node_id, _ in sorted_nodes[:top_k]]
        
        return critical
    
    def analyze(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[List[str]] = None,
        correlation_threshold: Optional[float] = None
    ) -> Dict:
        """
        Comprehensive network analysis.
        
        Parameters
        ----------
        data : DataFrame or ndarray
            Either correlation matrix or time series data
        labels : list, optional
            Variable names
        correlation_threshold : float, optional
            Minimum |correlation| for edge inclusion
            
        Returns
        -------
        dict
            Complete network analysis results
        """
        # Build network
        if isinstance(data, pd.DataFrame) and data.shape[0] != data.shape[1]:
            # Time series data
            self.build_network_from_timeseries(data, threshold=correlation_threshold)
        else:
            # Correlation matrix
            if isinstance(data, pd.DataFrame):
                labels = data.columns.tolist()
                data = data.values
            self.build_network_from_correlation(data, labels, correlation_threshold)
        
        # Compute MST
        mst = self.compute_minimum_spanning_tree()
        
        # Compute all centralities
        degree = self.compute_degree_centrality()
        betweenness = self.compute_betweenness_centrality()
        closeness = self.compute_closeness_centrality()
        eigenvector = self.compute_eigenvector_centrality()
        pagerank = self.compute_pagerank()
        
        # Detect communities
        communities = self.detect_communities_louvain()
        
        # Network metrics
        metrics = self.compute_network_metrics()
        
        # Critical nodes
        critical = self.identify_critical_nodes()
        
        return {
            'mst': {
                'edges': [(e.source, e.target, e.correlation) for e in mst],
                'total_distance': sum(e.distance for e in mst)
            },
            'centrality': {
                'degree': degree,
                'betweenness': betweenness,
                'closeness': closeness,
                'eigenvector': eigenvector,
                'pagerank': pagerank
            },
            'communities': communities,
            'num_communities': len(set(communities.values())),
            'metrics': metrics,
            'critical_nodes': critical
        }


# Convenience function
def network_analysis(
    data: Union[pd.DataFrame, np.ndarray],
    labels: Optional[List[str]] = None,
    threshold: Optional[float] = None
) -> Dict:
    """
    Quick network analysis of correlation data.
    
    Parameters
    ----------
    data : DataFrame or ndarray
        Correlation matrix or time series data
    labels : list, optional
        Variable names
    threshold : float, optional
        Minimum correlation for edge inclusion
        
    Returns
    -------
    dict
        Network analysis results
    """
    lens = GraphTheoryLens()
    return lens.analyze(data, labels, threshold)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create synthetic correlated data
    n_samples = 500
    n_vars = 8
    
    # Create correlation structure
    base = np.random.randn(n_samples, 3)
    data = np.zeros((n_samples, n_vars))
    data[:, 0] = base[:, 0] + 0.1 * np.random.randn(n_samples)
    data[:, 1] = 0.8 * base[:, 0] + 0.2 * np.random.randn(n_samples)
    data[:, 2] = 0.6 * base[:, 0] + 0.4 * np.random.randn(n_samples)
    data[:, 3] = base[:, 1] + 0.1 * np.random.randn(n_samples)
    data[:, 4] = 0.7 * base[:, 1] + 0.3 * np.random.randn(n_samples)
    data[:, 5] = base[:, 2] + 0.1 * np.random.randn(n_samples)
    data[:, 6] = 0.9 * base[:, 2] + 0.1 * np.random.randn(n_samples)
    data[:, 7] = 0.5 * base[:, 0] + 0.5 * base[:, 1] + 0.1 * np.random.randn(n_samples)
    
    df = pd.DataFrame(data, columns=['SPY', 'QQQ', 'IWM', 'TLT', 'AGG', 'GLD', 'SLV', 'XLF'])
    
    # Analyze
    results = network_analysis(df, threshold=0.3)
    
    print("Network Analysis Complete")
    print(f"\nNetwork Metrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    
    print(f"\nMinimum Spanning Tree ({len(results['mst']['edges'])} edges):")
    for src, tgt, corr in results['mst']['edges'][:5]:
        print(f"  {src} -- {tgt}: r={corr:.3f}")
    
    print(f"\nCommunities detected: {results['num_communities']}")
    for node, comm in results['communities'].items():
        print(f"  {node}: Community {comm}")
    
    print(f"\nCritical Nodes:")
    for metric, nodes in results['critical_nodes'].items():
        print(f"  {metric}: {', '.join(nodes[:3])}")
