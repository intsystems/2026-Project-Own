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

Over the past year, the Muon method has gained substantial popularity in deep learning. In its standard form, Muon computes (an approximation to) the polar decomposition of a weight matrix at each iteration and applies an update using the resulting orthogonal factor. A natural direction for further development is to reduce the per-step cost by avoiding decompositions of the full weight matrix.
In this project, we focus on improving the efficiency of the Muon optimizer. Our goal is to reduce the number of arithmetic operations per optimization step without materially degrading training quality. We plan to achieve this through a set of matrix-structured heuristics: instead of decomposing the entire weight matrix, we will operate on compressed representations (for example, submatrices, low-rank sketches, or other structured approximations) that retain the information most relevant to the update. We will systematically evaluate which matrix decompositions and approximation schemes provide the best trade-off between computational savings and optimization performance.

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
