import networkx as nx

def analyze_graph(interactions):
    """
    Creates a directed NetworkX graph from interaction pairs [attacker, victim].
    Computes network metrics to detect repeated attackers, victims, and cyberbullying clusters.
    
    Args:
        interactions (list of lists): E.g., [["@toxic_dan", "@kind_katie"], ...]
        
    Returns:
        dict: A dictionary containing nodes, edges, top abusers, and group clusters.
    """
    # Initialize a Directed Graph
    G = nx.DiGraph()
    
    # Track repeated attacks on the same victim using edge weights
    for attacker, victim in interactions:
        # Standardize handle casing
        u = attacker.strip()
        v = victim.strip()
        
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
            
    # If graph is empty, return structure
    if len(G.nodes) == 0:
        return {
            "nodes": [],
            "edges": [],
            "top_abusers": [],
            "clusters": []
        }
        
    # Calculate degree centrality
    # In-degree centrality = victimhood ratio
    # Out-degree centrality = bullying activity ratio
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    
    # Calculate clustering coefficient (requires converting to undirected for standard formula)
    undirected_G = G.to_undirected()
    clustering_coef = nx.clustering(undirected_G)
    
    # Identify bullying clusters (weakly connected components: groups of people talking to each other)
    weak_components = list(nx.weakly_connected_components(G))
    clusters = [list(c) for c in weak_components]
    
    # Compile node details
    nodes_data = []
    for node in G.nodes():
        attacks_sent = G.out_degree(node)
        attacks_received = G.in_degree(node)
        
        # Classify user role in the network
        if attacks_sent > 0 and attacks_received == 0:
            role = "Abuser"
        elif attacks_received > 0 and attacks_sent == 0:
            role = "Victim"
        elif attacks_sent > 0 and attacks_received > 0:
            role = "Aggressive Participator"
        else:
            role = "Neutral"
            
        nodes_data.append({
            "id": node,
            "role": role,
            "in_centrality": round(in_centrality.get(node, 0.0), 3),
            "out_centrality": round(out_centrality.get(node, 0.0), 3),
            "clustering": round(clustering_coef.get(node, 0.0), 3),
            "attacks_sent": attacks_sent,
            "attacks_received": attacks_received
        })
        
    # Compile edge details
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1)
        })
        
    # Find top abusers: Sort nodes by out_degree (attacks_sent) descending
    top_abusers = sorted(
        [n for n in nodes_data if n["attacks_sent"] > 0],
        key=lambda x: x["attacks_sent"],
        reverse=True
    )
    
    return {
        "nodes": nodes_data,
        "edges": edges_data,
        "top_abusers": top_abusers[:5],  # Return top 5
        "clusters": clusters
    }
