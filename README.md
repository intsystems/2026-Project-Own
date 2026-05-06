# Reducing the computational complexity of the Muon method through weight matrix decompositions

<!-- Change `kisnikser/m1p-template` to `intsystems/your-repository`-->
[![License](https://badgen.net/github/license/kisnikser/m1p-template?color=green)](https://github.com/kisnikser/m1p-template/blob/main/LICENSE)
[![GitHub Contributors](https://img.shields.io/github/contributors/kisnikser/m1p-template)](https://github.com/kisnikser/m1p-template/graphs/contributors)
[![GitHub Issues](https://img.shields.io/github/issues-closed/kisnikser/m1p-template.svg?color=0088ff)](https://github.com/kisnikser/m1p-template/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr-closed/kisnikser/m1p-template.svg?color=7f29d6)](https://github.com/kisnikser/m1p-template/pulls)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Ruslan Kabirov </td>
    </tr>
    <tr>
        <td align="left"> <b> Consultant </b> </td>
        <td> Aleksandr Shestakov </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Aleksandr Beznosikov, DSc </td>
    </tr>
</table>

## Assets

- [LinkReview](LINKREVIEW.md)
- [Code](code)
- [Paper](paper/main.pdf)
- [Slides](slides/main.pdf)

## Abstract

Muon has recently gained attention as an optimizer for neural network training because of its use of orthogonalized matrix-valued updates. In its standard form, Muon relies on Newton--Schulz iterations to approximate the polar factor of a matrix update, which introduces additional computational overhead. We study whether this cost can be reduced by replacing selected dense layers with Monarch-parameterized layers. This structure allows Newton--Schulz iterations to be applied to smaller block matrices and creates opportunities for parallel block-wise computation. We evaluate the proposed approach in pretraining experiments with a compact GPT-2 style model and compare it against Muon and AdamW baselines. We also evaluate a Dion2-inspired random column-partitioned variant as an ablation and find that it does not noticeably improve iteration-wise convergence over the simpler block-wise Monarch Muon update. Our goal is to reduce the cost of Muon's orthogonalization step while keeping validation loss and perplexity close to the baseline methods.
\end{abstract}

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
