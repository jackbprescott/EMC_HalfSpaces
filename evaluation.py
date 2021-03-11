import json
import warnings
import argparse
import time
from io_util import *
import evaluation_utils as eu

def main(args):
    outfile = f'results/eval/{args.dataset}/{int(time.time())}'
    if args.outfile_suffix:
        outfile += f'_{args.outfile_suffix}'
    outfile += '.json'

    warnings.filterwarnings('ignore')

    results = {args.dataset: {}}

    print(f'Dataset: {args.dataset}')
    for alpha in args.alpha:
        print(f'Alpha: {alpha}')
        results[args.dataset][alpha] = {}
        for eps in args.epsilon:
            results[args.dataset][alpha][eps] = []
            print(f'Epsilon: {eps}')
            for seed in args.seeds:
                print(f'Seed: {seed}')
                tr, ts = eu.load_data(args.dataset, seed=seed)
                risk, adv_risk = eu.evaluate(tr, ts, alpha, eps)
                results[args.dataset][alpha][eps].append((risk, adv_risk))

                write_json(results, outfile)
                print(f'\n{json.dumps(results, indent=4)}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the model, with the ability to try different combinations of model parameters.")

    # These args are separate from the cartesian product taken over the other args
    parser.add_argument(
        '--dataset',
        default='cifar',
        help='Name of dataset to use - can be cifar, mnist, or norm'
    )
    parser.add_argument(
        '--alpha',
        type=str,
        default='0.05',
        help='Comma separated minimum traditional risk'
    )
    parser.add_argument(
        '--epsilon',
        type=str,
        default='0.007843',
        help=f"Comma separated assumed l-inf perturbational capabilities of the adversary"
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='0,1,2,3,4',
        help=f'Comma separated random seeds to use, so that we can assess the variance of the estimator'
    )
    parser.add_argument(
        '--outfile_suffix',
        default=None,
        help=f'The format of the output file will be results/eval/{{dataset}}/{{timestamp}}_{{outfile_suffix}}.json'
    )

    #    global args
    args = parser.parse_args()

    args.alpha = [float(a) for a in args.alpha.replace(' ', '').split(',')]
    args.epsilon = [float(eps) for eps in args.epsilon.replace(' ', '').split(',')]
    args.seeds = [int(seed) for seed in args.seeds.replace(' ', '').split(',')]

    main(args)
