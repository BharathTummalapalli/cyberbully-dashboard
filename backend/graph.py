import networkx as nx
from datetime import datetime, timedelta

def analyze_graph(logs):
    """
    Creates a directed NetworkX graph from SQLite database logs of cyberbullying incidents.
    Filters to only include incidents from the last 7 days.
    Computes network metrics to detect repeated attackers, victims, and cyberbullying clusters.
    
    Args:
        logs (list of dict): Database log dictionaries containing timestamp, user_id, target_user, prediction.
        
    Returns:
        dict: A dictionary containing nodes, edges, top abusers, and group clusters.
    """
    G = nx.DiGraph()
    
    # Filter logs for rolling 7-day window
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    # We track total incidents in the last 7 days per user to calculate roles
    attacks_sent_7d = {}
    attacks_received_7d = {}
    
    # Gather nodes that had any activity in the last 7 days
    active_users = set()
    
    # Step 1: Parse logs and build edges within the 7-day window
    for log in logs:
        # Only process cyberbullying prediction logs
        if log.get("prediction") != "Cyberbullying":
            continue
            
        attacker = log.get("user_id") or log.get("username")
        target = log.get("target_user")
        
        if not attacker or not target:
            continue
            
        attacker = attacker.strip()
        target = target.strip()
        
        # Validate attacker and target: reject anonymous and self-directed loops
        if attacker == "@anonymous" or target == "@anonymous" or attacker == target:
            continue
            
        # Parse timestamp (supporting standard ISO strings)
        try:
            ts_str = log.get("timestamp", "")
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            # Fallback to now if parse fails
            ts = now
            
        # Check if it falls within the rolling 7-day window
        if ts >= seven_days_ago:
            active_users.add(attacker)
            active_users.add(target)
            
            # Increment attack counts
            attacks_sent_7d[attacker] = attacks_sent_7d.get(attacker, 0) + 1
            attacks_received_7d[target] = attacks_received_7d.get(target, 0) + 1
            
            # Add/increment edge weight
            if G.has_edge(attacker, target):
                G[attacker][target]['weight'] += 1
            else:
                G.add_edge(attacker, target, weight=1)
                
    # Ensure all active users are added as nodes
    for user in active_users:
        if not G.has_node(user):
            G.add_node(user)
            
    # If graph is empty, return empty structure
    if len(G.nodes) == 0:
        return {
            "nodes": [],
            "edges": [],
            "top_abusers": [],
            "clusters": []
        }
        
    # Calculate degree centrality indices
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)
    
    # Calculate clustering coefficient (requires converting to undirected)
    undirected_G = G.to_undirected()
    clustering_coef = nx.clustering(undirected_G)
    
    # Identify bullying cliques/clusters (weakly connected components)
    weak_components = list(nx.weakly_connected_components(G))
    clusters = [list(c) for c in weak_components]
    
    # Compile node details
    nodes_data = []
    for node in G.nodes():
        sent = attacks_sent_7d.get(node, 0)
        received = attacks_received_7d.get(node, 0)
        
        # Role classification based on 7-day threshold (Abuser if >= 3 attacks sent)
        if sent >= 3 and received >= 3:
            role = "Aggressive Participator"
        elif sent >= 3:
            role = "Abuser"
        elif received >= 3:
            role = "Victim"
        else:
            role = "Neutral"
            
        nodes_data.append({
            "id": node,
            "role": role,
            "in_centrality": round(in_centrality.get(node, 0.0), 3),
            "out_centrality": round(out_centrality.get(node, 0.0), 3),
            "clustering": round(clustering_coef.get(node, 0.0), 3),
            "attacks_sent": sent,
            "attacks_received": received
        })
        
    # Compile edge details
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "weight": data.get("weight", 1)
        })
        
    # Find top abusers: Sort nodes by attacks_sent descending
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
