#!/usr/bin/env python3
"""
step1_generate_and_metrics.py

Standalone script to:
 - Generate synthetic grid datasets (CSV + GeoJSON)
 - Compute baseline accessibility & capacity metrics for a dataset
 - Save a small JSON baseline report and print a summary to console

Usage:
  python step1_generate_and_metrics.py
"""

import os
import json
import math
import random
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_grid(grid_size=10,
                  population_range=(50, 500),
                  seed=0,
                  num_schools=3,
                  num_parks=3,
                  num_hospitals=1,
                  num_groceries=2,
                  num_transit=3,
                  num_commercial=5,
                  locked_fraction=0.0,
                  non_buildable_fraction=0.0):
    """
    Generates a synthetic grid DataFrame.
    Columns: x, y, cell_id, land_use, population, locked, non_buildable
    """
    random.seed(seed)
    np.random.seed(seed)
    cells = []
    cell_id = 0
    for x, y in product(range(grid_size), range(grid_size)):
        population = int(np.random.randint(population_range[0], population_range[1] + 1))
        cells.append({
            "x": int(x),
            "y": int(y),
            "cell_id": cell_id,
            "land_use": "Empty",
            "population": population,
            "locked": False,
            "non_buildable": False
        })
        cell_id += 1
    df = pd.DataFrame(cells)
    
    # mark some locked and non_buildable cells if requested
    total_cells = grid_size * grid_size
    if locked_fraction > 0:
        k = max(1, int(round(total_cells * locked_fraction)))
        locked_idxs = random.sample(list(df.index), k)
        df.loc[locked_idxs, "locked"] = True
        df.loc[locked_idxs, "land_use"] = "Locked"
    if non_buildable_fraction > 0:
        k = max(1, int(round(total_cells * non_buildable_fraction)))
        cand = [i for i in df.index if not df.loc[i, "locked"]]
        if cand:
            nb_idxs = random.sample(cand, min(k, len(cand)))
            df.loc[nb_idxs, "non_buildable"] = True
            df.loc[nb_idxs, "land_use"] = "NonBuildable"
    
    def place_amenity(count, amenity_name):
        empty_idxs = [i for i in df.index if df.loc[i, "land_use"] == "Empty" and not df.loc[i, "non_buildable"] and not df.loc[i, "locked"]]
        if len(empty_idxs) == 0:
            return []
        chosen = random.sample(empty_idxs, min(count, len(empty_idxs)))
        df.loc[chosen, "land_use"] = amenity_name
        return chosen
    
    place_amenity(num_schools, "School")
    place_amenity(num_parks, "Park")
    place_amenity(num_hospitals, "Hospital")
    place_amenity(num_groceries, "Grocery")
    place_amenity(num_transit, "Transit")
    place_amenity(num_commercial, "Commercial")
    
    return df

def df_to_geojson(df, grid_cell_size=1.0, crs_epsg=4326):
    """
    Convert grid DataFrame to a GeoJSON FeatureCollection of square polygons.
    """
    features = []
    for _, row in df.iterrows():
        x = row["x"]
        y = row["y"]
        coords = [
            [x, y],
            [x + grid_cell_size, y],
            [x + grid_cell_size, y + grid_cell_size],
            [x, y + grid_cell_size],
            [x, y]
        ]
        props = {
            "cell_id": int(row["cell_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "land_use": row["land_use"],
            "population": int(row["population"]),
            "locked": bool(row["locked"]),
            "non_buildable": bool(row["non_buildable"])
        }
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": props
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{crs_epsg}"}}
    }

# ---- Baseline metric functions ----
def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def compute_baseline_metrics(df,
                             amenity_types=("School", "Park", "Hospital", "Grocery", "Transit"),
                             distance_metric="manhattan",
                             service_radii=None,
                             capacities=None):
    """
    Compute baseline metrics for a layout in df.
    Returns: dict with amenity_metrics, population_total, per_cell_distances (DataFrame), landuse_ratios, equity
    """
    if distance_metric == "manhattan":
        dist_fn = manhattan_distance
    else:
        dist_fn = euclidean_distance
    
    default_radii = {"School": 3, "Park": 2, "Hospital": 5, "Grocery": 2, "Transit": 2}
    if service_radii is None:
        service_radii = default_radii.copy()
    else:
        for k, v in default_radii.items():
            service_radii.setdefault(k, v)
    if capacities is None:
        capacities = {k: float("inf") for k in amenity_types}
    
    amenities = {a: [] for a in amenity_types}
    for _, row in df.iterrows():
        if row["land_use"] in amenity_types:
            amenities[row["land_use"]].append((row["x"], row["y"], int(row["cell_id"])))
    
    demand_cells = df[(df["population"] > 0) & (df["land_use"] != "NonBuildable")].copy().reset_index(drop=True)
    population_total = int(demand_cells["population"].sum())
    
    per_cell = []
    for _, row in demand_cells.iterrows():
        cell_pos = (int(row["x"]), int(row["y"]))
        cell_population = int(row["population"])
        entry = {
            "cell_id": int(row["cell_id"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
            "population": cell_population
        }
        for a in amenity_types:
            amen_pos = [(ax, ay) for ax, ay, cid in amenities[a]]
            if len(amen_pos) == 0:
                entry[f"d_{a}"] = None
            else:
                dmin = min(dist_fn(cell_pos, (ax, ay)) for ax, ay in amen_pos)
                entry[f"d_{a}"] = dmin
        per_cell.append(entry)
    per_cell_df = pd.DataFrame(per_cell)
    
    metrics = {}
    for a in amenity_types:
        col = f"d_{a}"
        if per_cell_df[col].isnull().all():
            metrics[a] = {
                "population_weighted_avg_distance": None,
                "median_distance": None,
                "std_distance": None,
                "p90_distance": None,
                "coverage_within_radius": 0.0,
                "capacity_violations": None,
                "num_amenities": len(amenities[a])
            }
            continue
        distances = per_cell_df[col].dropna().astype(float)
        pops = per_cell_df.loc[distances.index, "population"].astype(int)
        if pops.sum() > 0 and len(distances) > 0:
            weighted_avg = float((distances * pops).sum() / pops.sum())
        else:
            weighted_avg = None
        median_distance = float(distances.median()) if len(distances) > 0 else None
        std_distance = float(distances.std()) if len(distances) > 0 else None
        p90_distance = float(distances.quantile(0.9)) if len(distances) > 0 else None
        
        radius = service_radii.get(a, default_radii.get(a, 2))
        covered_mask = per_cell_df[col].notnull() & (per_cell_df[col] <= radius)
        covered_population = int(per_cell_df.loc[covered_mask, "population"].sum())
        coverage = covered_population / population_total if population_total > 0 else 0.0
        
        cap = capacities.get(a, float("inf"))
        violations = 0
        if len(amenities[a]) == 0:
            violations = None
        else:
            amen_demand = {cid: 0 for _, _, cid in amenities[a]}
            for _, crow in per_cell_df.iterrows():
                d = crow[col]
                if pd.isnull(d):
                    continue
                min_cid = None
                min_dist = float("inf")
                for ax, ay, cid in amenities[a]:
                    dd = dist_fn((crow["x"], crow["y"]), (ax, ay))
                    if dd < min_dist:
                        min_dist = dd
                        min_cid = cid
                if min_cid is not None:
                    amen_demand[min_cid] += int(crow["population"])
            violations = sum(max(0, demand - cap) for demand in amen_demand.values())
        
        metrics[a] = {
            "population_weighted_avg_distance": weighted_avg,
            "median_distance": median_distance,
            "std_distance": std_distance,
            "p90_distance": p90_distance,
            "coverage_within_radius": coverage,
            "capacity_violations": violations,
            "num_amenities": len(amenities[a])
        }
    counts = df["land_use"].value_counts(normalize=True).to_dict()
    landuse_ratios = counts
    dcols = [c for c in per_cell_df.columns if c.startswith("d_")]
    per_cell_df["mean_distance_across_amenities"] = per_cell_df[dcols].mean(axis=1, skipna=True)
    overall_std = float(per_cell_df["mean_distance_across_amenities"].std()) if not per_cell_df["mean_distance_across_amenities"].isnull().all() else None
    overall_p90 = float(per_cell_df["mean_distance_across_amenities"].quantile(0.9)) if not per_cell_df["mean_distance_across_amenities"].isnull().all() else None
    
    out = {
        "amenity_metrics": metrics,
        "population_total": population_total,
        "per_cell_distances": per_cell_df,
        "landuse_ratios": landuse_ratios,
        "equity": {"std_mean_distance": overall_std, "p90_mean_distance": overall_p90}
    }
    return out

def save_geojson_from_df(df, outpath):
    gj = df_to_geojson(df)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(gj, f)

def main():
    datasets = [
        {"name": "dataset_10x10", "grid_size": 10, "seed": 1, "num_schools": 3, "num_parks": 3, "num_hospitals": 1, "locked_fraction": 0.0},
        {"name": "dataset_20x20_locked", "grid_size": 20, "seed": 2, "num_schools": 4, "num_parks": 4, "num_hospitals": 2, "locked_fraction": 0.03, "non_buildable_fraction": 0.02},
        {"name": "dataset_10x10_lowpop", "grid_size": 10, "seed": 3, "population_range": (10, 80), "num_schools": 2, "num_parks": 2, "num_hospitals": 1, "locked_fraction": 0.0},
    ]
    created = []
    for spec in datasets:
        name = spec.pop("name")
        df = generate_grid(**spec)
        csv_path = OUTPUT_DIR / f"{name}.csv"
        geojson_path = OUTPUT_DIR / f"{name}.geojson"
        df.to_csv(csv_path, index=False)
        save_geojson_from_df(df, geojson_path)
        created.append({"name": name, "csv": str(csv_path), "geojson": str(geojson_path)})
    print("Generated datasets:")
    for c in created:
        print(f"- {c['name']}: CSV -> {c['csv']}, GeoJSON -> {c['geojson']}")
    # Run baseline metrics on the first dataset
    df_example = pd.read_csv(created[0]["csv"])
    metrics_out = compute_baseline_metrics(df_example, distance_metric="manhattan")
    print("\nBaseline metrics (summary) for dataset_10x10:")
    print(f"- Population total: {metrics_out['population_total']}")
    print("- Land-use ratios (fraction):")
    for k, v in metrics_out["landuse_ratios"].items():
        print(f"   {k}: {v:.3f}")
    print("- Amenity metrics (summary):")
    for a, m in metrics_out["amenity_metrics"].items():
        num = m["num_amenities"]
        cov = m["coverage_within_radius"]
        avgd = m["population_weighted_avg_distance"]
        print(f"  {a}: num_amenities={num}, coverage={cov:.3f}, avg_dist={avgd}")
    # Save baseline report
    report_path = OUTPUT_DIR / f"{created[0]['name']}_baseline_report.json"
    r = {
        "dataset": created[0]["name"],
        "population_total": metrics_out["population_total"],
        "landuse_ratios": metrics_out["landuse_ratios"],
        "amenity_metrics": metrics_out["amenity_metrics"],
        "equity": metrics_out["equity"]
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    print(f"\nBaseline report saved to: {report_path}")
    print("\nFiles created in folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
