import numpy as np
import pandas as pd
from pathlib import Path


def load_grand_caricaie(gateway_csv_path, output_dir=None):
    """
    Load and preprocess Grand Caricaie marsh food web data from Gateway database.

    Parameters
    ----------
    gateway_csv_path : str or Path
        Path to the Gateway database CSV file (graph.csv)
    output_dir : str or Path, optional
        Directory to save preprocessed data. If None, returns data without saving.

    Returns
    -------
    adjacency : ndarray of shape (n_species, n_species)
        Symmetric adjacency matrix
    species_data : DataFrame
        DataFrame with species metadata including:
        - species: Species name
        - metabolic_type: endotherm vertebrate / ectotherm vertebrate / invertebrate
        - body_mass_g: Body mass in grams
        - in_degree: Number of predators (directed)
        - out_degree: Number of prey items (directed)
        - degree: Total degree in symmetric network
    """
    print("Loading Gateway database...")
    df = pd.read_csv(gateway_csv_path, low_memory=False)

    fw_name = "Grand Caricaie  marsh dominated by Cladietum marisci, mown  Clmown1"
    fw_data = df[df["foodweb.name"] == fw_name].copy()

    print(f"Extracted food web: {fw_name}")
    print(f"  Raw links: {len(fw_data)}")

    fw_data["con.mass.mean.g."] = pd.to_numeric(
        fw_data["con.mass.mean.g."], errors="coerce"
    )
    fw_data["res.mass.mean.g."] = pd.to_numeric(
        fw_data["res.mass.mean.g."], errors="coerce"
    )
    fw_data["con.mass.mean.g."] = fw_data["con.mass.mean.g."].replace(-999, np.nan)
    fw_data["res.mass.mean.g."] = fw_data["res.mass.mean.g."].replace(-999, np.nan)

    all_species = sorted(
        set(fw_data["con.taxonomy"].unique()) | set(fw_data["res.taxonomy"].unique())
    )
    n_species = len(all_species)

    print(f"  Unique species: {n_species}")

    species_to_idx = {sp: i for i, sp in enumerate(all_species)}

    A_dir = np.zeros((n_species, n_species))
    for _, row in fw_data.iterrows():
        con_idx = species_to_idx[row["con.taxonomy"]]
        res_idx = species_to_idx[row["res.taxonomy"]]
        A_dir[con_idx, res_idx] = 1

    A_sym = ((A_dir + A_dir.T) > 0).astype(float)

    print(f"  Directed edges: {int(A_dir.sum())}")
    print(f"  Symmetric edges: {int(A_sym.sum())//2}")
    print(f"  Density: {A_sym.sum()/(n_species**2):.3f}")

    species_info = []
    for sp in all_species:
        idx = species_to_idx[sp]

        is_consumer = sp in fw_data["con.taxonomy"].values
        is_resource = sp in fw_data["res.taxonomy"].values

        metabolic_type = np.nan
        body_mass = np.nan
        movement_type = np.nan

        if is_consumer:
            con_rows = fw_data[fw_data["con.taxonomy"] == sp].iloc[0]
            metabolic_type = con_rows["con.metabolic.type"]
            body_mass = con_rows["con.mass.mean.g."]
            movement_type = con_rows["con.movement.type"]

        if is_resource and pd.isna(body_mass):
            res_rows = fw_data[fw_data["res.taxonomy"] == sp].iloc[0]
            if pd.notna(res_rows["res.mass.mean.g."]):
                body_mass = res_rows["res.mass.mean.g."]
            if pd.isna(metabolic_type) and pd.notna(res_rows["res.metabolic.type"]):
                metabolic_type = res_rows["res.metabolic.type"]

        in_degree = int(A_dir[:, idx].sum())
        out_degree = int(A_dir[idx, :].sum())
        degree = int(A_sym[idx].sum())

        species_info.append(
            {
                "species": sp,
                "metabolic_type": metabolic_type,
                "body_mass_g": body_mass if pd.notna(body_mass) else np.nan,
                "movement_type": movement_type,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "degree": degree,
            }
        )

    species_data = pd.DataFrame(species_info)

    print("\nMetadata coverage:")
    print(
        f'  Body mass: {species_data["body_mass_g"].notna().sum()}/{n_species} species'
    )
    print(
        f'  Metabolic type: {species_data["metabolic_type"].notna().sum()}/{n_species} species'
    )

    print("\nMetabolic type distribution:")
    print(species_data["metabolic_type"].value_counts())

    valid_mass = species_data[species_data["body_mass_g"] > 0]["body_mass_g"]
    if len(valid_mass) > 0:
        print(f"\nBody mass range: {valid_mass.min():.2e} to {valid_mass.max():.2e} g")
        print(f"Orders of magnitude: {np.log10(valid_mass.max()/valid_mass.min()):.1f}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        species_data.to_csv(output_dir / "species.csv", index=False)
        print(f'\nSaved: {output_dir / "species.csv"}')

        np.save(output_dir / "adjacency_symmetric.npy", A_sym)
        print(f'Saved: {output_dir / "adjacency_symmetric.npy"}')

        np.save(output_dir / "adjacency_directed.npy", A_dir)
        print(f'Saved: {output_dir / "adjacency_directed.npy"}')

        edges_df = pd.DataFrame(
            {
                "consumer": [all_species[i] for i, j in zip(*np.where(A_dir))],
                "resource": [all_species[j] for i, j in zip(*np.where(A_dir))],
            }
        )
        edges_df.to_csv(output_dir / "edges.csv", index=False)
        print(f'Saved: {output_dir / "edges.csv"}')

        metadata = {
            "n_species": n_species,
            "n_edges_directed": int(A_dir.sum()),
            "n_edges_symmetric": int(A_sym.sum()) // 2,
            "density": float(A_sym.sum() / (n_species**2)),
            "food_web_name": fw_name,
            "location": "Switzerland, Lake Neuchatel",
            "ecosystem_type": "terrestrial aboveground (marsh)",
            "source": "iDIV Gateway Database",
            "reference": "Cattin Blandenier (2004)",
        }

        import json

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f'Saved: {output_dir / "metadata.json"}')

        print(f"\nAll files saved to: {output_dir}")

    return A_sym, species_data


if __name__ == "__main__":
    gateway_csv = Path("data/gateway/graph.csv")
    output_dir = Path("data/gateway/grand_caricaie")

    adjacency, species = load_grand_caricaie(gateway_csv, output_dir)

    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE")
    print("=" * 70)
    print(f"\nAdjacency matrix shape: {adjacency.shape}")
    print(f"Species data shape: {species.shape}")
    print(f"\nTo load the data in your analysis:")
    print("  import numpy as np")
    print("  import pandas as pd")
    print("  ")
    print('  adjacency = np.load("data/adjacency_symmetric.npy")')
    print('  species = pd.read_csv("data/species.csv")')
