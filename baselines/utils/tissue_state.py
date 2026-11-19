import pandas as pd

from .common import Columns


def load_tissue_state(
    tissue_state_csv_path: str,
    tissue_state_to_string: dict[int, str],
    tissue_state_func=None,
) -> pd.DataFrame:
    """Per patient (Patient_uId index), from a headerless "<patient uid>,<raw class>" csv: its raw class
    (Columns.PATIENT_CLASS_ID) and class name (Columns.PATIENT_CLASS). tissue_state_func names the class when given,
    else tissue_state_to_string, with spaces removed."""
    patient_class_df = pd.read_csv(tissue_state_csv_path, index_col=0, names=[Columns.PATIENT_CLASS_ID])
    if tissue_state_func:
        patient_class_df[Columns.PATIENT_CLASS] = patient_class_df[Columns.PATIENT_CLASS_ID].transform(
            tissue_state_func, axis=0
        )
    else:
        patient_class_df[Columns.PATIENT_CLASS] = patient_class_df[Columns.PATIENT_CLASS_ID].transform(
            lambda row: tissue_state_to_string[row].replace(" ", ""), axis=0
        )
    return patient_class_df
