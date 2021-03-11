# EMC_HalfSpaces
Empirically estimates the concentration of measure of a dataset using halfspaces. Based on this ICLR 2021 paper: https://openreview.net/forum?id=BUlyHkzjgmA.

### Libraries
NumPy v1.19.4

Torchvision 0.5.0

SciKit-Learn 0.22.1

### Usage

To recreate the MNIST experiments from the paper: ```python -m evaluation --dataset mnist --alpha 0.01 --epsilon 0.1,0.2,0.3,0.4 --seeds 0,10,20,30,40```

To recreate the CIFAR-10 experiments from the paper: ```python -m evaluation --dataset cifar --alpha 0.05 --epsilon 0.007843,0.015686,0.031372,0.062745 --seeds 0,10,20,30,40```

To recreate the Gaussian distribution experiments from the paper: ```python -m evaluation --dataset cifar --alpha 0.05,0.5 --epsilon 1.0 --seeds 0,10,20,30,40```

The script will output its progress along the way. It will also write all results to a JSON file at the path ```results/eval/{dataset}/{timestamp in seconds}_{optional file suffix}.json```. This JSON file will have the structure

```
{
  DATASET_NAME: {
    ALPHA_VALUE_1: {
      EPSILON_VALUE_1: [
        [
          TEST_TRADITIONAL_RISK_FOR_SEED_1,
          TEST_ADVERSARIAL_RISK_FOR_SEED_1,
        ],
        [
          TEST_TRADITIONAL_RISK_FOR_SEED_2,
          TEST_ADVERSARIAL_RISK_FOR_SEED_2,
        ],
        ...
      ],
      EPSILON_VALUE_2: [
        ...
      ]
    },
    ALPHA_VALUE_2: {
      ...
    },
    ...
  }
}
```
