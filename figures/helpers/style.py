from pathlib import Path

import matplotlib.pyplot as plt

from figures.helpers.datasets import LONG_TERM, SHORT_TERM

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'

CLASS_COLORS = {
    # TNBC & Human Glioma
    SHORT_TERM: '#C62828',
    LONG_TERM: '#1F6FC5',
    # Melanoma
    'NN': '#2A78D6',
    'NP': '#EB6834',
    'PN': '#1BAF7A',
    # SEA-AD: the worse class red, as in the CRC and glioma figures
    'ADNC-Not/Low': '#2A78D6',
    'ADNC-Int/High': '#C62828',
}
# sequential colormaps in each class's hue, for per-class heatmaps
CLASS_CMAPS = {
    SHORT_TERM: 'Reds',
    LONG_TERM: 'Blues',
    'NN': 'Blues',
    'NP': 'Oranges',
    'PN': 'Greens',
    'ADNC-Not/Low': 'Blues',
    'ADNC-Int/High': 'Reds',
}
MOTIF_SIZE_COLORS = {3: '#3F9E3F', 4: '#7B2FBE', 5: '#D9822B'}  # as in the published melanoma panels
CUTOFF_COLOR = '#8FBF6F'
ANNOTATION_COLOR = '#52514E'
GRID_COLOR = '#E6E5E1'


def apply_style():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        # keep text editable when the pdf is opened in Illustrator/Inkscape
        'pdf.fonttype': 42,
    })


def results_dir(dataset_key: str) -> Path:
    """figures/results/<dataset_key>, created if missing."""
    out_dir = RESULTS_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, dataset_key: str, name: str, formats=('png', 'pdf')) -> list[Path]:
    """Save to figures/results/<dataset_key>/<name>.<format> for each format."""
    out_dir = results_dir(dataset_key)

    paths = []
    for fmt in formats:
        path = out_dir / f'{name}.{fmt}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        paths.append(path)
    return paths
