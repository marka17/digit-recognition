import pathlib
import argparse
import toml
import json
from tqdm import tqdm
from pprint import pprint

from src.models.jasper import Jasper
from src.losses import CTCLoss
from src.utils.metrics import calculate_cer

from src.hyperparams import Hyperparams
from src.data import (
    SpeechDataset, PaddingCollator,
    BatchProcessor
)
from src.data.transforms import (
    ChainFeatureProcessor,
    AmplitudeToDBProcessor,
    MelAugmenter,
    StackingMelProcessor,
    NoiseInjector,
    AudioLevelAdjuster,
    ChainAudioProcessor,
    NoiseDataset
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import torch
from torch import optim
from torch.utils.data import DataLoader


def main(args):
    writer = SummaryWriter(args.experiment_log_path)
    hparams = Hyperparams()
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    # Prepare train data
    train_audio_processor = ChainAudioProcessor([
        NoiseInjector(NoiseDataset('data/noise/noise.txt')),
        AudioLevelAdjuster(),
    ])
    train_dataset = SpeechDataset('data/numbers/train.csv', train_audio_processor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  collate_fn=PaddingCollator(), shuffle=True,
                                  pin_memory=True, num_workers=4)

    train_feature_processor = ChainFeatureProcessor([
        AmplitudeToDBProcessor(),
        StackingMelProcessor(3, 3),
        MelAugmenter(prob=0.4)
    ])
    train_batch_processor = BatchProcessor(hparams, feature_processor=train_feature_processor) \
        .to(device)

    # Prepare val data
    valid_dataset = SpeechDataset('data/numbers/val.csv')
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset) // 4,
                                  collate_fn=PaddingCollator())

    valid_feature_processor = ChainFeatureProcessor([
        AmplitudeToDBProcessor(),
        StackingMelProcessor(3, 3),
    ])
    valid_batch_processor = BatchProcessor(hparams, feature_processor=valid_feature_processor) \
        .to(device)

    with open(args.model_config, 'r') as fout:
        config = toml.load(fout)

    model = Jasper(jasper_model_definition=config, num_classes=11, feat_in=512).to(device)
    criterion = CTCLoss(blank=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Length of train: {len(train_dataloader)}")
    print(f"Length of val: {len(valid_dataloader)}")
    print(f"Num of params: {model.num_weights()}")

    for epoch in tqdm(range(args.num_epoch)):

        model.train()
        for i, batch in tqdm(enumerate(train_dataloader)):
            if device.type == 'cuda':
                batch = batch.cuda(non_blocking=True)

            processed_batch = train_batch_processor(batch)

            log_probs = model(processed_batch.x)
            input_lengths = processed_batch.x_len // 2 + 1
            targets = processed_batch.y + 1

            loss = criterion(log_probs, targets,
                             input_lengths, processed_batch.y_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            argmax_decoding = log_probs.detach().cpu().argmax(dim=-1)
            cer, pairs = calculate_cer(targets, argmax_decoding)

            # Log metrics
            global_step = i + epoch * len(train_dataloader)
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/cer', cer, global_step)
            writer.add_text('train/target', "\n\n".join(pairs[:5]), global_step)

        model.eval()
        for i, batch in enumerate(valid_dataloader):
            if device.type == 'cuda':
                batch = batch.cuda(non_blocking=True)

            processed_batch = valid_batch_processor(batch)

            with torch.no_grad():
                log_probs = model(processed_batch.x)
                input_lengths = processed_batch.x_len // 2 + 1
                targets = processed_batch.y + 1

                loss = criterion(log_probs, targets,
                                 input_lengths, processed_batch.y_len)

            argmax_decoding = log_probs.detach().cpu().argmax(dim=-1)
            cer, pairs = calculate_cer(targets, argmax_decoding)

            global_step = i + epoch * len(train_dataloader)
            writer.add_scalar('valid/loss', loss.item(), global_step)
            writer.add_scalar('valid/cer', cer, global_step)
            writer.add_text('valid/target', "\n\n".join(pairs), global_step)

        checkpoint_path = pathlib.Path(experiment_model_path) / f"{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--experiment-name', type=str, default='debug')
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--model-config', type=str, default='src/models/jasper_nomask.toml')

    # Training
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    experiment_root = pathlib.Path('./experiments') / args.experiment_name
    args.experiment_root = str(experiment_root)
    if not experiment_root.exists():
        print(experiment_root)
        experiment_root.mkdir()

    with open(experiment_root / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    experiment_log_path = experiment_root / 'logs'
    args.experiment_log_path = str(experiment_log_path)
    if not experiment_log_path.exists():
        experiment_log_path.mkdir()

    experiment_model_path = experiment_root / 'models'
    args.experiment_model_path = str(experiment_model_path)
    if not experiment_model_path.exists():
        experiment_model_path.mkdir()

    pprint(vars(args))
    main(args)
