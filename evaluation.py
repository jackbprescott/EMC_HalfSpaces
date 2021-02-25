import json
import warnings
import evaluation_utils as eu

def main():
    outfile = 'results/eval/cifar_results.json'
    warnings.filterwarnings('ignore')

    datasets = ['cifar', 'mnist', 'norm']
    seeds = [0, 10, 20, 30, 40]
    params = {
                'cifar': {
                        'alpha': 0.05,
                        'eps': [0.007843, 0.015686, 0.031372, 0.062745]
                    },
                'mnist': {
                        'alpha': 0.01,
                        'eps': [0.1, 0.2, 0.3, 0.4]
                    },
                'norm': {
                        'alpha': 0.05,
                        'eps': [1]
                    }
             }

    results = {'cifar': {}, 'mnist': {}, 'norm': {}}

    for dataset in datasets:
        print(f'Dataset: {dataset}\n')
        alpha = params[dataset]['alpha']
        for eps in params[dataset]['eps']:
            results[dataset][eps] = []
            print(f'Epsilon: {eps}')
            for seed in seeds:
                print(f'Seed: {seed}')
                tr, ts = eu.load_data(dataset, seed=seed)
                risk, adv_risk = eu.evaluate(tr, ts, alpha, eps)
                results[dataset][eps].append((risk, adv_risk))

                with open(outfile, 'w') as handle:
                    json.dump(results, handle, indent=4)

                print('')
                print(json.dumps(results, indent=4))
                print('')

if __name__ == '__main__':
    main()
