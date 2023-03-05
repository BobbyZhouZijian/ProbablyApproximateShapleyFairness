# Probably Approximate Shapley Fairness with Applications in Machine Learning

This repository is the official implementation of [Probably Approximate Shapley Fairness with Applications in Machine Learning](https://arxiv.org/abs/2212.00630). Accepted to AAAI-23 Oral Presentation.

>📋 In this work, we propose a Probably Approximate Shapley Fairness framework which allows Shapley value estimations to achieve theoretical gaurantees on the various Shapley fairness properties - an essential motivation for utilising Shapley values in various Machine Learning scenarios.

## Citing
If you have found our work to be useful in your research, please consider citing it with the following bibtex:
```
@InProceedings{Zhou2023,
    author="Zhou, Zijian
    and Xu, Xinyi
    and Sim, Rachel Hwee Ling
    and Foo, Chuan Sheng
    and Low, Kian Hsiang",
    title="Probably Approximate Shapley Fairness with Applications in Machine Learning",
    bookTitle="37th AAAI Conference on Artificial Intelligence",
    year="2023",
}
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
>📋  We recommend managing your environment using Anaconda, both for the versions of the packages used here and for easy management. 


## Estimation

To reproduce the experiment results presented in the paper, first run SV estimator with the corresponding python script and appropriate arguments, e.g.:
```
python feature_exp.py -d adult -m 1000 -t 5 -n 7
```
for estimating SVs under `feature attribution` scenario on `adult` dataset with `1000` samples, `5` trials, and `7` total number of features. You may check the `argparse` implementations in each Python file for more details.

>📋  You may also supply your own datasets for estimation. Just modify the data loading part in the corresponding Python files.

## Evaluation

The `/notesbooks` folder includes jupyter notebooks which run evaluations on different experiments mentioned in the paper. You may search for the experiment you are interested in by looking at the name of the notebook. For example, if you want to run an experiment on nullity, you may first run `python nullity_exp.py ...` with appropriate arguments. Then, use `notebooks/nullity.ipynb` to evaluate.


## Contributing

>📋 Suggestions and questions are welcome through issues. All contributions welcome! All content in this repository is licensed under the MIT license.
