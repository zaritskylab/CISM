import pandas as pd


def switch_all(motif_to_cells_identity_hash: dict,
               target_motif_ids: list,
               motif_space_features: pd.DataFrame,
               include_all_motifs: bool,
               include_all_cell_identity: bool) -> (dict, dict):
    motifs_features_dict = dict()
    ci_features_dict = dict()

    for idx, row in motif_space_features.iterrows():
        patient_unique_id = row['test_patient_id']
        motif_ids = row['features']
        # transform to cell identity composition
        ci_features = []
        motifs_features = []
        for motif_id, cell_identity_hash in motif_to_cells_identity_hash.items():
            if target_motif_ids is not None and motif_id in target_motif_ids:
                motifs_features.append(motif_id)
                continue

            if include_all_cell_identity:
                ci_features.append(cell_identity_hash)

            if target_motif_ids is None and include_all_motifs:
                motifs_features.append(motif_id)

        ci_features = list(set(ci_features))
        motifs_features_dict[patient_unique_id] = motifs_features
        ci_features_dict[patient_unique_id] = ci_features

    return motifs_features_dict, ci_features_dict
