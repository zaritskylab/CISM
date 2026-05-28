# motif_hits_delaunay.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay


# ============================== Motif spec =============================== #
@dataclass
class MotifSpec:
    """
    Motif definition with node variables (A, B, C, ...) and typed constraints.
    Types can be integers OR strings (e.g., "CD4T"). Edges use A -> B syntax.

    Example:
        A.type = CD4T
        B.type = Tumor
        A -> B
    """
    node_types: Dict[str, Union[int, str]]
    edges: List[Tuple[str, str]]

def parse_motif_text(text: str) -> MotifSpec:
    node_types: Dict[str, Union[int, str]] = {}
    edges: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            a, b = [s.strip() for s in line.split("->", 1)]
            if a and b:
                edges.append((a, b))
            continue
        if ".type" in line and "=" in line:
            lhs, rhs = [s.strip() for s in line.split("=", 1)]
            var = lhs.split(".")[0].strip()
            try:
                node_types[var] = int(rhs)
            except ValueError:
                node_types[var] = rhs
    if not node_types:
        raise ValueError("Motif text parse error: no node types found.")
    return MotifSpec(node_types=node_types, edges=edges)


# ========================== Graph construction =========================== #
def graph_from_fov_delaunay(
    df_fov: pd.DataFrame,
    *,
    id_col: str = "cellID",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    type_col: str = "class",
) -> nx.Graph:
    """
    Build an undirected graph for a single FOV using Delaunay triangulation
    over (x,y) centroids. Each simplex adds 3 undirected edges (triangle edges).
    Node attributes:
        - 'type_name' : original class string
        - 'type'      : numeric factorized code
        - 'pos'       : (y, x)
        - 'cell_id'   : original id
    """
    if df_fov.empty:
        return nx.Graph()

    missing = [c for c in (id_col, x_col, y_col, type_col) if c not in df_fov.columns]
    if missing:
        raise KeyError(f"Missing required columns in FOV dataframe: {missing}")

    ids = df_fov[id_col].to_numpy()
    xs = df_fov[x_col].astype(float).to_numpy()
    ys = df_fov[y_col].astype(float).to_numpy()
    tnames = df_fov[type_col].astype(str).to_numpy()

    # stable numeric encoding by class string (kept per-FOV)
    tcodes, _ = pd.factorize(tnames)

    G = nx.Graph()
    for cid, x, y, tname, tcode in zip(ids, xs, ys, tnames, tcodes):
        G.add_node(
            int(cid),
            type=int(tcode),
            type_name=str(tname),
            pos=(float(y), float(x)),
            cell_id=int(cid),
        )

    # Delaunay triangulation
    pts = np.column_stack([xs, ys])
    # Need >= 3 non-collinear points to triangulate
    if len(pts) >= 3:
        # Handle rare Qhull errors for degenerate sets by falling back to no edges
        try:
            tri = Delaunay(pts)
            for tri_inds in tri.simplices:  # shape (n_triangles, 3)
                a, b, c = tri_inds
                u, v, w = int(ids[a]), int(ids[b]), int(ids[c])
                G.add_edge(u, v)
                G.add_edge(v, w)
                G.add_edge(w, u)
        except Exception:
            # If triangulation fails (collinear or degenerate), no edges are added
            pass

    return G


# ============================ Motif matching ============================= #
def _candidate_nodes_by_type(G: nx.Graph, desired: Union[int, str]) -> List[int]:
    """Integers match node['type'], strings match node['type_name']."""
    out: List[int] = []
    if isinstance(desired, int):
        for n, d in G.nodes(data=True):
            if int(d.get("type", -999)) == desired:
                out.append(n)
    else:
        want = str(desired)
        for n, d in G.nodes(data=True):
            if str(d.get("type_name", "")) == want:
                out.append(n)
    return out

def find_motif_hits_in_graph(
    G: nx.Graph, spec: MotifSpec, *, induced: bool = False
) -> List[Tuple[int, ...]]:
    """
    Backtracking matcher:
      1) Build candidate pool for each motif variable by type constraint.
      2) Assign variables; for each motif edge (A->B), require an edge between chosen nodes (undirected check).
      3) If induced=True, forbid extra edges among the chosen nodes that are not present in the motif.
    Returns unique hits deduplicated by node-set.
    """
    if G.number_of_nodes() == 0:
        return []

    order = list(spec.node_types.keys())
    pools = {v: _candidate_nodes_by_type(G, t) for v, t in spec.node_types.items()}
    if any(len(p) == 0 for p in pools.values()):
        return []

    motif_edge_dir = set(spec.edges)
    motif_edge_undirected = {tuple(sorted(e)) for e in motif_edge_dir}

    results: List[Tuple[int, ...]] = []
    used: set[int] = set()
    chosen: Dict[str, int] = {}

    def ok_edges_partial() -> bool:
        # Edge presence for already-bound motif endpoints
        for (a, b) in motif_edge_dir:
            if a in chosen and b in chosen:
                if not G.has_edge(chosen[a], chosen[b]):
                    return False
        # Induced: forbid extra edges among chosen nodes
        if induced and len(chosen) >= 2:
            items = list(chosen.items())  # (var, node)
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    va, na = items[i]
                    vb, nb = items[j]
                    if G.has_edge(na, nb):
                        if tuple(sorted((va, vb))) not in motif_edge_undirected:
                            return False
        return True

    def backtrack(i: int) -> None:
        if i == len(order):
            tup = tuple(chosen[v] for v in order)
            results.append(tup)
            return
        var = order[i]
        for node in pools[var]:
            if node in used:
                continue
            chosen[var] = node
            used.add(node)
            if ok_edges_partial():
                backtrack(i + 1)
            used.remove(node)
            del chosen[var]

    backtrack(0)

    # Deduplicate by node-set to avoid automorphism duplicates
    dedup: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for t in results:
        key = tuple(sorted(t))
        dedup.setdefault(key, t)
    return list(dedup.values())


# ============================= Public API ================================= #
def motif_instances_per_fov_from_csv(
    csv_path: str,
    motif_text: str,
    *,
    fov_col: str = "fov",
    id_col: str = "cellID",
    x_col: str = "centroid_x",
    y_col: str = "centroid_y",
    type_col: str = "class",
    #class_col: str = "Group:",
    classes: List[str] = ['NN', 'NP'],
    patient_col: Optional[str] = None,  # e.g., "patient number" if present
    induced: bool = False,
) -> Tuple[Dict[str, List[Tuple[int, ...]]], Optional[pd.DataFrame]]:
    """
    Build Delaunay graph per FOV from the CSV, find motif hits, and return:
      - dict: {FOV: [(cell_id,...), ...]}
      - wide dataframe: columns 'patient','FOV','cell1','cell2',...
    Motif node types can be integers (match node['type']) or strings (match node['type_name']).
    """
    spec = parse_motif_text(motif_text)
    df = pd.read_csv(csv_path)
    #df = df[df[class_col].isin(classes)]
    if fov_col not in df.columns or id_col not in df.columns:
        raise KeyError(f"Expected columns '{fov_col}' and '{id_col}' in CSV.")
    for c in (x_col, y_col, type_col):
        if c not in df.columns:
            raise KeyError(f"Expected column '{c}' in CSV.")

    out: Dict[str, List[Tuple[int, ...]]] = {}
    wide_rows: List[Dict[str, object]] = []

    # groupby FOV, build Delaunay graph, match motif
    for fov, gdf in df.groupby(fov_col, sort=False):
        G = graph_from_fov_delaunay(gdf, id_col=id_col, x_col=x_col, y_col=y_col, type_col=type_col)
        hits = find_motif_hits_in_graph(G, spec, induced=induced)
        out[str(fov)] = hits

        # Optional: build wide records for a DataFrame
        for tup in hits:
            row = {
                "patient": (gdf[patient_col].iloc[0] if (patient_col and patient_col in gdf.columns) else np.nan),
                "FOV": fov,
            }
            for i, cid in enumerate(tup, start=1):
                row[f"cell{i}"] = cid
            wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows) if wide_rows else None
    return out, wide_df


# ============================= CLI example ================================ #
if __name__ == "__main__":
    # Example motif: CD4T -- Tumor pair (by string class names)
    motif = """
    
    A.type = mac
    B.type = b cell
    A -> B
    """

    hits_by_fov, wide = motif_instances_per_fov_from_csv(
        csv_path="mapped_cell_types.csv",
        motif_text=motif,
        fov_col="fov",
        id_col="cell_id",
        x_col="centroid-0",
        y_col="centroid-1",
        type_col="pred_mapped",
        #class_col= "Group",
        classes=['NN', 'NP'],
        patient_col="patient number",   # set to your patient column if present
        induced=False      # True to require induced subgraph
        
    )


    # Save the optional wide dataframe for later analysis
    if wide is not None:
        wide.to_csv("motif_hits_wide.csv", index=False)
        print("Saved motif_hits_wide.csv")
