from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from cism.cism import TissueStateDiscriminativeMotifs

REPO_ROOT = Path(__file__).resolve().parents[2]
# the tutorials' network datasets and FANMOD results (tutorial/cism_tutorial.py, tutorial/run_melanoma_pipeline.py)
PET_DATA_ROOT = REPO_ROOT / 'pet_data'
TUTORIAL_RUNTIME_ROOT = REPO_ROOT / 'tutorial' / 'runtime'

# same class names as the tutorials' map_survival_rate_to_class
SHORT_TERM = 'Short-term survival'
LONG_TERM = 'Long-term survival'

# graph color id -> cell type name, the common_cells_type of each dataset's CISM run (checked against the cell tables)
TNBC_CELLS_TYPE = {
    0: 'B cells', 1: 'CD3 T cells', 2: 'CD4 t cells', 3: 'CD8 T cells', 4: 'DC', 5: 'DC/Mono', 6: 'Endothelial',
    7: 'Immune other', 8: 'Macrophages', 9: 'Mesenchyme', 10: 'Mono/Neu', 11: 'NK cells', 12: 'Neutrophils',
    13: 'Tregs', 14: 'Tumor', 15: 'Unidentified',
}
# meta_pred labels, except tumor cells, which keep their pred subtype
GLIOMA_CELLS_TYPE = {
    0: 'APC', 1: 'Bcells', 2: 'DC_Mac_CD209', 3: 'Endothelial_cells', 4: 'Immune_unassigned', 5: 'Macrophage_CD206',
    6: 'Macrophage_CD68', 7: 'Macrophage_CD68_CD163', 8: 'Mast_cells', 9: 'Microglia', 10: 'Microglia_CD163',
    11: 'Myeloid_CD11b_HLADRminus', 12: 'Myeloid_CD11b_HLADRplus', 13: 'Myeloid_CD14', 14: 'Myeloid_CD141',
    15: 'Myeloid_CD14_CD163', 16: 'Neurons', 17: 'Neutrophils', 18: 'Tcell_CD4', 19: 'Tcell_CD8', 20: 'Tcell_FoxP3',
    21: 'Tumor', 22: 'Tumor_ApoE', 23: 'Tumor_B7H3', 24: 'Tumor_EGFR', 25: 'Tumor_GFAP', 26: 'Tumor_GM2_GD2',
    27: 'Tumor_GPC2', 28: 'Tumor_H3K27M', 29: 'Tumor_IDH1', 30: 'Tumor_Olig2', 31: 'Undetermined',
}
# meta_pred of analysis/melanoma/melanoma_cells.csv: the paper's 16 types (pred_mapped of
# Datasets/Melanoma/mapped_cell_types.csv), with the same color codes; 13 (tumor) never occurs in these graphs
MELANOMA_CELLS_TYPE = {
    0: 'B cell', 1: 'CD4 APC', 2: 'CD4 T cell', 3: 'CD4 Treg T cell', 4: 'CD8 T cell', 5: 'DCs',
    6: 'Germinal Center cell', 7: 'Hevs', 8: 'Macrophage', 9: 'Memory CD4 T cell', 10: 'NK cell', 11: 'Neutrophil',
    12: 'Stroma', 13: 'Tumor', 14: 'Unidentified', 15: 'Vessels',
}
# meta_pred of analysis/seaad/adnc_neighborhoods/seaad_cells_column_cortex_neighborhood.csv: SEA-AD's cellular
# neighborhoods for neurons, glial subclasses and Microglia-PVM kept apart, Endothelial + VLMC as Vascular
SEAAD_NEIGHBORHOOD_CELLS_TYPE = {
    0: 'Astrocyte', 1: 'CGE inhibitory', 2: 'Deep-projecting excitatory', 3: 'IT excitatory', 4: 'MGE inhibitory',
    5: 'Microglia-PVM', 6: 'OPC', 7: 'Oligodendrocyte', 8: 'Vascular',
}


@dataclass(frozen=True)
class SurvivalLabels:
    # headerless "<patient uid>,<survival days>" rows, the patient_class.csv format from the README
    csv_path: Path
    # patients with survival days < cutoff_days are short-term, the rest long-term
    cutoff_days: int
    axis_label: str = 'Survival days'


@dataclass(frozen=True)
class MotifRun:
    """Where the dataset's FANMOD results are cached, and the CISM settings of its discriminative motifs."""
    fanmod_output_root: Path
    fanmod_cache_root: Path
    motif_size: int
    stringency: float
    max_class_features: int = 30
    # FANMOD IDs of the discriminative motifs the paper's figures mark, in their numbering (I, II, III...)
    highlighted_motifs: tuple = ()


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    # folder of CISM network files (Patient_<id>_FOV<fov>.txt); None while the dataset has no graphs in data/
    graphs_dir: Optional[Path] = None
    cells_type: Optional[dict] = None
    # Patient_uId = patient_uid_prefix + patient id, the way the patient class csv keys patients
    patient_uid_prefix: str = ''
    # None for datasets whose classes are not short vs long survival
    survival: Optional[SurvivalLabels] = None
    # classes of non-survival datasets: headerless "<patient uid>,<raw class>" csv, and raw class -> class name
    patient_class_csv: Optional[Path] = None
    class_names: Optional[dict] = None
    # the two classes figures compare unless told otherwise, for datasets with more than two
    default_classes: Optional[tuple] = None
    motif_run: Optional[MotifRun] = None
    # leave-one-patient-out ROC AUC per seed of the GNN baseline (baselines/GNN, run separately) on the default
    # classes' patients; figures draw them as given
    gnn_aucs: tuple = ()


DATASETS = {
    'tnbc': DatasetConfig(
        key='tnbc',
        display_name='TNBC',
        graphs_dir=PET_DATA_ROOT / 'TNBC',
        cells_type=TNBC_CELLS_TYPE,
        patient_uid_prefix='TNBC',
        survival=SurvivalLabels(csv_path=REPO_ROOT / 'analysis' / 'tnbc' / 'patient_class_survival_days.csv',
                                cutoff_days=1000),
        # analysis/tnbc/mibitof_tnbc_tutorial.ipynb's motifs. Figure 8's motifs I-III are the CD4 T - DC/Mono - Tumor,
        # B - CD8 T - Tumor and CD4 T - CD4 T - CD4 T triangles
        motif_run=MotifRun(fanmod_output_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_output',
                           fanmod_cache_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_cache',
                           motif_size=3, stringency=0.45, highlighted_motifs=(6077900, 1867212, 6052940))),
    # survival_duration_days_first_diag from the glioma metadata_complete.csv, for the 49 patients with
    # spatial transcriptomics (NS == 1). 548 days (1.5 years) reproduces the paper's short/long labels exactly.
    'glioma': DatasetConfig(
        key='glioma',
        display_name='Human Glioma',
        graphs_dir=PET_DATA_ROOT / 'Human_Glioma_Short_vs_Long_Paper_Aligned',
        cells_type=GLIOMA_CELLS_TYPE,
        patient_uid_prefix='Glioma',
        survival=SurvivalLabels(csv_path=REPO_ROOT / 'analysis' / 'glioma' / 'patient_class_survival_days.csv',
                                cutoff_days=548,
                                axis_label='Survival days (from first diagnosis)'),
        # the paper run's tuned settings (CISM_paper_revision/melanoma_tutorials/cism_tutorial.py)
        motif_run=MotifRun(fanmod_output_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_output',
                           fanmod_cache_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_cache',
                           motif_size=4, stringency=0.2526),
        gnn_aucs=(0.7407, 0.7862, 0.7407, 0.7492, 0.6919)),
    'melanoma': DatasetConfig(
        key='melanoma',
        display_name='Melanoma',
        graphs_dir=PET_DATA_ROOT / 'Melanoma',
        cells_type=MELANOMA_CELLS_TYPE,
        patient_uid_prefix='Melanoma',
        patient_class_csv=PET_DATA_ROOT / 'Melanoma' / 'patient_class.csv',
        class_names={'NN': 'NN', 'NP': 'NP', 'PN': 'PN'},
        default_classes=('NN', 'NP'),
        # the old Melanoma_100_distance_NN_NP.ipynb's motif-induced matrices: 5-node motifs at stringency 0.46
        motif_run=MotifRun(fanmod_output_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_output',
                           fanmod_cache_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_cache',
                           motif_size=5, stringency=0.46)),
    # SEA-AD MTG MERFISH, cortical-column FOVs, run by tutorial/run_seaad_pipeline.py: ADNC Intermediate/High vs
    # Not AD/Low (the NIA-AA threshold), 9 neighborhood colors, at that run's tuned stringency
    'seaad_adnc': DatasetConfig(
        key='seaad_adnc',
        display_name='SEA-AD MTG',
        graphs_dir=PET_DATA_ROOT / 'SEAAD_ADNC_Neighborhoods',
        cells_type=SEAAD_NEIGHBORHOOD_CELLS_TYPE,
        patient_uid_prefix='SEAAD_ADNC_Neighborhoods',
        patient_class_csv=PET_DATA_ROOT / 'SEAAD_ADNC_Neighborhoods' / 'patient_class.csv',
        # no spaces: _load_tissue_state strips them from mapped class names
        class_names={'adnc_not_low': 'ADNC-Not/Low', 'adnc_int_high': 'ADNC-Int/High'},
        default_classes=('ADNC-Not/Low', 'ADNC-Int/High'),
        motif_run=MotifRun(fanmod_output_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_output',
                           fanmod_cache_root=TUTORIAL_RUNTIME_ROOT / 'fanmod_cache',
                           motif_size=4, stringency=0.4411),
        gnn_aucs=(0.489, 0.539, 0.445, 0.670, 0.528)),
    # graphs not built yet: the cells are in Datasets/CRC/CRC_cells.csv
    'crc': DatasetConfig(key='crc', display_name='CRC'),
}
# datasets the motif figures can run on: graphs and a CISM run's settings
MOTIF_DATASETS = [key for key, config in DATASETS.items() if config.motif_run is not None]


def survival_class_func(cutoff_days: int):
    def map_survival_rate_to_class(survival_rate: int):
        if survival_rate < cutoff_days:
            return SHORT_TERM
        return LONG_TERM

    return map_survival_rate_to_class


def tissue_state_kwargs(config: DatasetConfig) -> dict:
    """The patient class arguments of TissueStateDiscriminativeMotifs / PairwiseAnalysis for this dataset."""
    if config.survival is not None:
        return dict(tissue_state_csv_path=str(config.survival.csv_path),
                    tissue_state_to_string=None,
                    tissue_state_func=survival_class_func(config.survival.cutoff_days))
    return dict(tissue_state_csv_path=str(config.patient_class_csv),
                tissue_state_to_string=config.class_names,
                tissue_state_func=None)


def all_classes(config: DatasetConfig) -> list:
    if config.survival is not None:
        return [SHORT_TERM, LONG_TERM]
    return list(config.class_names.values())


def class_labels(config: DatasetConfig, classes=None) -> list:
    """The two classes a figure compares: the given ones, else the dataset's default pair."""
    available = all_classes(config)
    classes = list(classes or config.default_classes or available)
    if len(set(classes)) != 2 or len(classes) != 2 or not set(classes) <= set(available):
        raise ValueError(f'{config.display_name}: pick two of its classes {available}, got {classes}')
    return classes


def classes_tag(config: DatasetConfig, classes: list) -> str:
    """'_<A>_vs_<B>' for datasets with more than two classes, so outputs of different pairs don't overwrite each other."""
    return '' if len(all_classes(config)) == 2 else f'_{classes[0]}_vs_{classes[1]}'


def load_patient_classes(config: DatasetConfig) -> pd.DataFrame:
    """Per patient (Patient_uId index): raw class (patient_class_id) and class name (patient_class), as CISM loads them."""
    return TissueStateDiscriminativeMotifs._load_tissue_state(**tissue_state_kwargs(config))


def load_survival_classes(config: DatasetConfig) -> pd.DataFrame:
    """Per patient: survival days (patient_class_id) and short/long class (patient_class), split as CISM does."""
    if config.survival is None:
        raise ValueError(f'{config.display_name} has no survival-days labels')

    return load_patient_classes(config)
