import contextlib
import io
import math

import numpy as np

from cism.cism import (CISM, DiscriminativeFeatureKey, GetMotifsResult, HardDiscriminativeFC,
                       TissueStateDiscriminativeMotifs)
from cism.helpers import create_weight_matrix_from_motifs
from figures.helpers.datasets import REPO_ROOT, DatasetConfig, tissue_state_kwargs

# only run when a FOV has no cached FANMOD result
FANMOD_PATH = REPO_ROOT / 'cism' / 'FANMOD_binaries'
# the motif sizes the stringency sweep figures compare, and their coarse stringency grid (0.05, 0.10, ..., 0.60)
MOTIF_SIZES = (3, 4, 5)
STRINGENCY_GRID = tuple(round(0.05 * step, 2) for step in range(1, 13))
# around the dataset's own stringency the sweeps sample every FINE_STEP instead, up to FINE_REACH away from it
FINE_STEP = 0.02
FINE_REACH = 0.05


def sweep_stringencies(selected: float, stringencies=STRINGENCY_GRID) -> list:
    """The stringencies a sweep figure evaluates, ascending: the coarse grid, resampled every FINE_STEP within
    FINE_REACH of the dataset's own (selected) stringency so the curve around it is resolved. The selected stringency
    takes the place of any fine step within half a step of it, so no point crowds it: 0.4411 on the default grid
    gives ..., 0.35, 0.40, 0.42, 0.4411, 0.46, 0.48, 0.50, 0.55, 0.60."""
    selected = round(selected, 4)
    first = max(math.ceil(round((selected - FINE_REACH) / FINE_STEP, 6)), 0)
    last = math.floor(round((selected + FINE_REACH) / FINE_STEP, 6))
    fine = {round(step * FINE_STEP, 4) for step in range(first, last + 1)}
    fine = {stringency for stringency in fine if round(abs(stringency - selected), 6) > FINE_STEP / 2}
    coarse = {round(stringency, 4) for stringency in stringencies if round(abs(stringency - selected), 6) > FINE_REACH}
    return sorted(coarse | fine | {selected})


def load_discriminator(config: DatasetConfig, motif_size: int, n_jobs: int) -> TissueStateDiscriminativeMotifs:
    """The dataset's motifs, parsed from its cached FANMOD results, with its patient classes."""
    cism = CISM(fanmod_exe='LocalFANMOD', fanmod_path=str(FANMOD_PATH),
                network_dataset_root_path=str(config.graphs_dir.parent),
                fanmod_output_root_path=str(config.motif_run.fanmod_output_root),
                fanmod_cache_root_path=str(config.motif_run.fanmod_cache_root),
                motif_size=motif_size, iterations=1000)
    cism.add_dataset(config.graphs_dir.name, 'Disease', config.patient_uid_prefix, n_jobs=n_jobs)
    return TissueStateDiscriminativeMotifs(cism=cism, common_cells_type=config.cells_type, **tissue_state_kwargs(config))


def hard_discriminative_fc(labels: list, stringency: float, max_class_features: int) -> HardDiscriminativeFC:
    """CISM's hard-discriminative motif selection: motifs found in one class's patients only, carried by more than the
    stringency fraction of them, the max_class_features most frequent per class."""
    return HardDiscriminativeFC(labels=labels, extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
                                use_cells_type_composition=False, use_motifs=True,
                                shared_percentage=float(stringency), max_class_features=max_class_features)


def discriminative_motifs(discriminator: TissueStateDiscriminativeMotifs, labels: list, stringency: float,
                          max_class_features: int, n_jobs: int) -> GetMotifsResult:
    """CISM's hard-discriminative motifs of the two classes: the ones get_features selects in any leave-one-out fold."""
    feature_conf = hard_discriminative_fc(labels, stringency, max_class_features)
    features = discriminator.get_features(feature_conf=feature_conf, exclude_patients=[], n_jobs=n_jobs)
    return GetMotifsResult(motif_space_features=features, discriminator=discriminator, classes=labels)


def motif_induced_matrices(discriminator: TissueStateDiscriminativeMotifs, cells_type: dict, labels: list,
                           stringency: float, max_class_features: int, n_jobs: int) -> tuple:
    """Class -> cell type x cell type probabilities of an edge in the pooled instances of the class's discriminative
    motifs, and class -> its number of discriminative motifs."""
    motifs = discriminative_motifs(discriminator, labels, stringency, max_class_features, n_jobs)
    motifs_by_class = motifs.get_motifs_by_class()
    motifs_weight = motifs.get_motifs_mean_weight()

    matrices = {}
    for label in labels:
        with contextlib.redirect_stdout(io.StringIO()):  # it prints every motif it adds
            matrix = create_weight_matrix_from_motifs(discriminator=discriminator, cells_type=cells_type,
                                                      cell_identity_to_motif_hash=motifs.cell_identity_to_motif_hash,
                                                      motifs_ids=motifs_by_class[label], motifs_weight=motifs_weight)
        # each motif edge lands in one of (i, j) and (j, i); fold them onto pairwise_interactions_matrix.py's
        # symmetric scale, where the diagonal is a pair's probability and an off-diagonal cell half of it
        matrices[label] = (matrix + matrix.T) / 2

    # the cell types of the run's motifs, alphabetically, as in pairwise_interactions_matrix.py
    colors_vecs = discriminator.cism.motifs_dataset.drop_duplicates('colors_vec_hash').colors_vec
    present = np.stack(colors_vecs.to_numpy()).sum(axis=0) > 0
    cell_types = sorted(name for name, is_present in zip(cells_type.values(), present) if is_present)
    matrices = {label: matrix.loc[cell_types, cell_types] for label, matrix in matrices.items()}
    return matrices, {label: len(motifs_by_class[label]) for label in labels}
