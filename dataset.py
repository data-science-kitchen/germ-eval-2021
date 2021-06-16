from flair.datasets import CSVClassificationCorpus
import pandas as pd
from pathlib import Path
import os
from typing import Union


class GermEval2021(CSVClassificationCorpus):
    SPLIT_RANGES = [(0, 811), (811, 1622), (1622, 2433), (2433, 3244)]

    def __init__(self,
                 base_path: Union[str, Path],
                 fold: int = 0,
                 seed: int = 1234,
                 **corpusargs):
        data_frame = pd.read_csv(base_path, header=0)
        data_frame = data_frame.sample(frac=1, random_state=seed).reset_index(drop=True)
        data_frame = data_frame.drop(columns=['comment_id'])

        fold_dir = Path(os.path.dirname(base_path), 'fold_{}'.format(fold))
        fold_dir.mkdir(parents=True, exist_ok=True)

        dev_fold_indices = list(range(self.SPLIT_RANGES[fold][0], self.SPLIT_RANGES[fold][1]))

        training_data = data_frame.drop(data_frame.index[dev_fold_indices]).reset_index(drop=True)
        training_data.to_csv(fold_dir / 'train.csv', index=False)

        dev_data = data_frame.iloc[data_frame.index[dev_fold_indices]].reset_index(drop=True)
        dev_data.to_csv(fold_dir / 'dev.csv', index=False)
        
        super(GermEval2021, self).__init__(
            data_folder=fold_dir,
            column_name_map={0: 'text', 1: 'label_toxic', 2: 'label_engaging', 3: 'label_fact_claiming'},
            train_file='train.csv',
            dev_file='dev.csv',
            **corpusargs
        )
