import toml
from pathlib import Path


class Config:
    def __init__(self, cfg_name: str = 'config.toml') -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        cfg = toml.load(self.root_dir / 'config' / cfg_name)
        p = cfg['paths']

        # Core dirs
        self.data_dir   = self.root_dir / p['data_dir']
        self.things_dir = self.root_dir / p['things_dir']
        self.macq_dir   = self.root_dir / p['macq_dir']
        self.timeavg_dir     = self.root_dir / p['macq_time_averaged_dir']
        self.timeres_dir     = self.root_dir / p['macq_time_resolved_dir']
        self.mri_dir    = self.root_dir / p['mri_dir']
        self.dnn_dir    = self.root_dir / p['dnn_dir']
        self.fig_dir    = self.root_dir / p['fig_dir']
        self.results_dir = self.root_dir / p['results_dir']
        self.cache_file  = self.root_dir / p['cache_file']
        # Optional explicit categories TSV path
        if 'categories_tsv' in p:
            self.categories_tsv = self.root_dir / p['categories_tsv']

        # Expose other config sections
        self.analysis = cfg.get('analysis', {})
        self.hyperparameters = cfg.get('hyperparameters', {})
        self.plotting = cfg.get('plotting', {})

    # Helpers
    def get_macq_paths(self, monkey: str) -> dict:
        return {
            'data': self.timeavg_dir / f'monkey{monkey}.npy',
            'stiminfo': self.timeavg_dir / f'monkey{monkey}_stiminfo.csv',
        }

    def get_macq_tr_paths(self, monkey: str) -> dict:
        return {
            'data': self.timeres_dir / f'monkey{monkey}_tr.npy',
            'stiminfo': self.timeres_dir / f'monkey{monkey}_tr_stiminfo.csv',
        }

    def get_mri_paths(self, sub: str) -> dict:
        return {
            'betas': self.mri_dir / 'betas_csv' / f'sub-{sub}_ResponseData.h5',
            'stiminfo': self.mri_dir / f'sub-{sub}_stiminfo.csv',
            'brainmask': self.mri_dir / 'brainmasks' / f'sub-{sub}_space-T1w_brainmask.nii.gz',
            'voxmeta': self.mri_dir / 'betas_csv' / f'sub-{sub}_VoxelMetadata.csv',
        }


config = Config()
