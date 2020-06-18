import toml
import argparse
import collections
import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.models.jasper import Jasper
from src.hyperparams import Hyperparams
from src.data import SpeechDataset, PaddingCollator, ctc_decode, BatchProcessor
from src.data.transforms import (
    ChainFeatureProcessor, AmplitudeToDBProcessor,
    StackingMelProcessor
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-csv', type=str)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--model-config', type=str, default='src/models/jasper_nomask.toml')
    parser.add_argument('--path-to-weights', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    with open(args.model_config, 'r') as fout:
        config = toml.load(fout)

    state_dict = torch.load(args.path_to_weights, map_location='cpu')

    model = Jasper(jasper_model_definition=config, num_classes=11, feat_in=512).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    dataset = SpeechDataset(args.path_to_csv)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=PaddingCollator())
    feature_processor = ChainFeatureProcessor([
        AmplitudeToDBProcessor(),
        StackingMelProcessor(3, 3),
    ])
    batch_processor = BatchProcessor(Hyperparams(), feature_processor=feature_processor) \
        .to(device)

    result = collections.defaultdict(list)
    for batch in dataloader:
        if device.type == 'cuda':
            batch = batch.cuda(non_blocking=True)

        processed_batch = batch_processor(batch)

        with torch.no_grad():
            log_probs = model(processed_batch.x)

        argmax_decodings = log_probs.detach().cpu().argmax(dim=-1)

        for decoding in argmax_decodings.tolist():
            decoding = ctc_decode(decoding)

            result['path'].append(batch.paths[0])
            result['number'].append(decoding)

    df_result = pd.DataFrame(result)
    df_result.to_csv('result.csv', index=False)
