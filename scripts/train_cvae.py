
import argparse


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CVAE on spiral data.')
    parser.add_argument('--num_modes', type=int, default=2, help='Number of modes to use as performance attributes.')
    parser.add_argument('--data_fraction', type=float, default=1, help='Fraction of data to use.')
    parser.add_argument('--only_freqs', type=bool, default=False, help='Whether only frequencies or also modal masses should be considered.')
    parser.add_argument('--model_name', type=str, default='spiral', help='Name of the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--val_split', type=float, default=.1, help='Fraction of data to use for validation.')
    args = parser.parse_args()
    train_cvae(args.num_modes, args.data_fraction, args.only_freqs, args.model_name, args.batch_size, args.val_split)