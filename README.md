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

Over the past year, the Muon method has gained substantial popularity in deep learning. In its standard form, Muon computes an approximation to the polar decomposition of a weight matrix at each iteration and applies an update using the resulting orthogonal factor, which introduces additional computational overhead. We propose [...] as a more efficient alternative and show, through pretraining experiments on several models, that it reduces computational cost while preserving the key optimization benefits of Muon.

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
