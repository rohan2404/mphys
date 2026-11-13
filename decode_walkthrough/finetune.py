"""Warm-start DECODE's 0.10 training engine from a model state dict.

Run in a fresh process; this deliberately changes no DECODE source files.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', required=True)
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    weights = Path(args.weights).resolve(strict=True)
    import torch
    import decode.neuralfitter.train.train as trainer
    original = trainer.setup_trainer

    def setup_with_weights(*positional, **keywords):
        result = original(*positional, **keywords)
        model = result[2]
        state = torch.load(str(weights), map_location='cpu')
        model.load_state_dict(state, strict=True)
        print('WARM START: loaded model weights from', weights, flush=True)
        print('Optimizer and learning-rate scheduler start fresh.', flush=True)
        return result

    trainer.setup_trainer = setup_with_weights
    trainer.live_engine_setup(str(Path(args.param).resolve()))


if __name__ == '__main__':
    main()
