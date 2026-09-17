# AtlasFold benchmark results and protocols

## Release contents

The benchmark FASTA files are included under [`assets/benchmarks/`](../assets/benchmarks/). The benchmark release is distributed from [Google Drive](https://drive.google.com/drive/folders/1KjhQe4yvLMSBEJdxXZ6oC9Wi-pw5a439) and contains CAMEO22, CASP14, and CASP15 structure predictions and FoldBench evaluation results. All metrics were computed with OpenStructure 2.9.1.

## Monomer protocol

AtlasFold is evaluated on 183 CAMEO22, 70 CASP14, and 56 CASP15 targets.
Generate the released AtlasFold predictions with:

```bash
atlasfold monomer -i assets/benchmarks/cameo22.fasta -o predictions/cameo22 --num-samples 1 --seed 1 2 3 4 5
atlasfold monomer -i assets/benchmarks/casp14.fasta -o predictions/casp14 --num-samples 1 --seed 1 2 3 4 5
atlasfold monomer -i assets/benchmarks/casp15.fasta -o predictions/casp15 --num-samples 1 --seed 1 2 3 4 5
```

### CAMEO22

| Type | Model | TM-score | GDT-TS | lDDT | lDDT-Cα | RMSD (Å) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MSA-based | RoseTTAFold | 0.780 / 0.860 | 0.715 / 0.775 | 0.575 / 0.605 | 0.798 / 0.827 | 5.721 / 2.864 |
| MSA-based | RoseTTAFold2 | 0.864 / 0.947 | 0.844 / 0.900 | 0.751 / 0.794 | 0.891 / 0.923 | 3.513 / 1.728 |
| MSA-based | AlphaFold2 | 0.879 / 0.955 | 0.863 / 0.914 | 0.826 / 0.869 | 0.904 / 0.932 | 3.178 / 1.574 |
| PLM-based | ESMFold | 0.853 / 0.933 | 0.826 / 0.875 | 0.791 / 0.832 | 0.871 / 0.906 | 3.995 / 2.018 |
| PLM-based | SimpleFold-3B | 0.835 / 0.911 | 0.801 / 0.854 | 0.776 / 0.807 | 0.855 / 0.889 | 4.234 / 2.183 |
| PLM-based | AtlasFold | 0.865 / 0.945 | 0.847 / 0.907 | 0.838 / 0.880 | 0.898 / 0.936 | 3.538 / 1.909 |

### CASP14

| Type | Model | TM-score | GDT-TS | lDDT | lDDT-Cα | RMSD (Å) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MSA-based | RoseTTAFold | 0.654 / 0.678 | 0.562 / 0.572 | 0.464 / 0.456 | 0.705 / 0.723 | 9.676 / 6.420 |
| MSA-based | RoseTTAFold2 | 0.802 / 0.881 | 0.744 / 0.815 | 0.670 / 0.706 | 0.832 / 0.877 | 6.614 / 3.167 |
| MSA-based | AlphaFold2 | 0.844 / 0.893 | 0.784 / 0.842 | 0.778 / 0.803 | 0.865 / 0.893 | 4.481 / 2.881 |
| PLM-based | ESMFold | 0.702 / 0.790 | 0.621 / 0.713 | 0.635 / 0.700 | 0.722 / 0.797 | 8.652 / 4.069 |
| PLM-based | SimpleFold-3B | 0.716 / 0.788 | 0.638 / 0.705 | 0.666 / 0.697 | 0.748 / 0.829 | 7.969 / 4.048 |
| PLM-based | AtlasFold | 0.732 / 0.804 | 0.650 / 0.724 | 0.683 / 0.748 | 0.751 / 0.839 | 7.299 / 3.988 |

### CASP15

| Type | Model | TM-score | GDT-TS | lDDT | lDDT-Cα | RMSD (Å) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MSA-based | RoseTTAFold | 0.639 / 0.685 | 0.550 / 0.554 | 0.636 / 0.700 | 0.721 / 0.790 | 13.961 / 6.740 |
| MSA-based | RoseTTAFold2 | 0.724 / 0.843 | 0.668 / 0.725 | 0.644 / 0.746 | 0.803 / 0.877 | 14.302 / 4.500 |
| MSA-based | AlphaFold2 | 0.751 / 0.854 | 0.702 / 0.747 | 0.759 / 0.829 | 0.842 / 0.897 | 9.759 / 4.870 |
| PLM-based | ESMFold | 0.671 / 0.759 | 0.608 / 0.667 | 0.671 / 0.774 | 0.755 / 0.835 | 14.259 / 8.323 |
| PLM-based | ESMFold2 | 0.695 / 0.734 | 0.645 / 0.712 | 0.726 / 0.827 | 0.790 / 0.892 | 14.499 / 5.755 |
| PLM-based | SimpleFold-3B | 0.653 / 0.694 | 0.575 / 0.573 | 0.637 / 0.700 | 0.728 / 0.804 | 15.802 / 8.380 |
| PLM-based | AtlasFold | 0.701 / 0.789 | 0.645 / 0.711 | 0.725 / 0.812 | 0.790 / 0.885 | 10.307 / 7.952 |


## FoldBench protocol

AtlasFold-M is evaluated on the antibody–antigen and protein–protein subsets of [FoldBench](https://github.com/BEAM-Labs/FoldBench).

The settings for each five-seed evaluation subset are:

- AlphaFold-Multimer v2.3: 10 recycles, 5 models, 5 seeds
- AlphaFold3: 10 recycles, 5 seeds, 5 diffusion samples, 200 diffusion steps, no templates
- Boltz-1: 10 recycles, 5 seeds, 5 diffusion samples, 200 diffusion steps
- Protenix-v1: 10 recycles, 5 seeds, 5 diffusion samples, 200 diffusion steps
- ESMFold2-MSA: 10 loops, 5 seeds, 5 diffusion samples, 68 truncated diffusion steps
- ESMFold2: 10 loops, 5 seeds, 5 diffusion samples, 68 truncated diffusion steps
- AtlasFold-M: 10 recycles, 5 seeds, 5 diffusion samples, 200 diffusion steps, no templates

For confidence-based selection, we use the official ranking score for each model.

To reduce seed-to-seed variation, we generate predictions for ten seeds (`1`–`10`) and evaluate every 5-of-10 seed combination (`C(10,5) = 252`). Each combination contains 25 candidates, and the reported performance is the mean over all combinations.

Generate the released AtlasFold-M predictions with:

```bash
atlasfold multimer -i assets/benchmarks/foldbench_abag.fasta -o predictions/foldbench_abag --num-samples 5 --seed 1 2 3 4 5 6 7 8 9 10
atlasfold multimer -i assets/benchmarks/foldbench_pp.fasta -o predictions/foldbench_pp --num-samples 5 --seed 1 2 3 4 5 6 7 8 9 10
```

### Antibody–antigen
#### Rank

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 8.516 | 27.935 | 40.924 | 0.276 | 0.275 | 8.221 | 27.611 | 0.797 |
| AlphaFold3 | 20.951 | 39.636 | 49.285 | 0.376 | 0.383 | 7.267 | 23.962 | 0.877 |
| Boltz-1 | 5.987 | 22.566 | 34.967 | 0.235 | 0.240 | 8.734 | 28.759 | 0.841 |
| Protenix-v1 | 21.288 | 37.394 | 44.945 | 0.354 | 0.362 | 7.477 | 23.275 | 0.866 |
| ESMFold2-MSA | 21.036 | 42.693 | 52.270 | 0.391 | 0.401 | 7.165 | 24.615 | 0.869 |
| ESMFold2 | 20.981 | 42.290 | 48.602 | 0.378 | 0.394 | 7.690 | 25.724 | 0.849 |
| AtlasFold-M | 18.173 | 35.359 | 46.955 | 0.346 | 0.361 | 7.616 | 25.631 | 0.861 |

#### Oracle

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 11.213 | 34.542 | 52.921 | 0.355 | 0.395 | 5.365 | 17.155 | 0.826 |
| AlphaFold3 | 29.614 | 49.285 | 66.168 | 0.486 | 0.519 | 4.009 | 12.203 | 0.891 |
| Boltz-1 | 10.341 | 24.958 | 39.505 | 0.283 | 0.310 | 6.870 | 21.137 | 0.853 |
| Protenix-v1 | 28.899 | 46.099 | 65.818 | 0.467 | 0.505 | 4.247 | 12.771 | 0.882 |
| ESMFold2-MSA | 33.230 | 49.059 | 63.427 | 0.480 | 0.513 | 4.100 | 11.747 | 0.881 |
| ESMFold2 | 31.467 | 48.491 | 61.489 | 0.474 | 0.516 | 4.196 | 11.717 | 0.862 |
| AtlasFold-M | 22.626 | 47.628 | 64.733 | 0.460 | 0.512 | 4.142 | 13.203 | 0.874 |

#### Average

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 3.349 | 12.628 | 24.477 | 0.175 | 0.169 | 9.689 | 31.560 | 0.772 |
| AlphaFold3 | 17.988 | 35.523 | 43.547 | 0.339 | 0.344 | 7.686 | 25.063 | 0.875 |
| Boltz-1 | 5.663 | 19.953 | 32.221 | 0.220 | 0.227 | 9.021 | 29.969 | 0.840 |
| Protenix-v1 | 16.802 | 32.035 | 40.779 | 0.311 | 0.320 | 8.083 | 26.182 | 0.863 |
| ESMFold2-MSA | 18.733 | 36.837 | 44.709 | 0.342 | 0.350 | 7.949 | 26.945 | 0.862 |
| ESMFold2 | 16.930 | 34.802 | 42.395 | 0.323 | 0.336 | 8.847 | 29.476 | 0.840 |
| AtlasFold-M | 13.267 | 27.640 | 38.453 | 0.283 | 0.284 | 8.585 | 28.197 | 0.855 |


### Protein–protein
#### Rank

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 37.416 | 59.244 | 65.206 | 0.515 | 0.542 | 8.144 | 17.005 | 0.718 |
| AlphaFold3 | 43.424 | 68.334 | 73.006 | 0.592 | 0.641 | 5.656 | 13.852 | 0.844 |
| Boltz-1 | 37.316 | 62.229 | 71.120 | 0.551 | 0.598 | 6.015 | 14.499 | 0.800 |
| Protenix-v1 | 44.043 | 66.966 | 74.445 | 0.588 | 0.637 | 5.585 | 13.790 | 0.832 |
| ESMFold2-MSA | 42.983 | 66.638 | 73.230 | 0.585 | 0.635 | 5.817 | 14.183 | 0.806 |
| ESMFold2 | 35.703 | 61.786 | 69.216 | 0.537 | 0.586 | 6.418 | 15.016 | 0.791 |
| AtlasFold-M | 26.829 | 54.291 | 63.358 | 0.474 | 0.517 | 7.570 | 18.198 | 0.768 |

#### Oracle

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 43.294 | 63.312 | 68.615 | 0.565 | 0.611 | 6.252 | 12.649 | 0.757 |
| AlphaFold3 | 55.488 | 72.499 | 78.028 | 0.659 | 0.712 | 3.882 | 8.766 | 0.859 |
| Boltz-1 | 46.834 | 66.505 | 76.632 | 0.605 | 0.664 | 4.612 | 10.728 | 0.816 |
| Protenix-v1 | 54.832 | 74.499 | 81.572 | 0.666 | 0.723 | 3.674 | 8.284 | 0.852 |
| ESMFold2-MSA | 55.434 | 71.363 | 77.237 | 0.652 | 0.713 | 3.942 | 8.506 | 0.823 |
| ESMFold2 | 47.917 | 66.906 | 74.071 | 0.613 | 0.669 | 4.186 | 8.506 | 0.809 |
| AtlasFold-M | 40.636 | 64.314 | 71.560 | 0.576 | 0.635 | 4.669 | 10.627 | 0.799 |

#### Average

| Model | High (%) | Medium+ (%) | Accept.+ (%) | DockQ | Fnat | iRMSD (Å) | LRMSD (Å) | lDDT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AF-Multimer v2.3 | 32.065 | 54.281 | 59.698 | 0.471 | 0.498 | 8.649 | 18.553 | 0.693 |
| AlphaFold3 | 43.640 | 66.561 | 71.504 | 0.582 | 0.626 | 5.833 | 14.119 | 0.841 |
| Boltz-1 | 36.770 | 60.942 | 70.230 | 0.538 | 0.587 | 6.268 | 15.188 | 0.791 |
| Protenix-v1 | 40.986 | 60.719 | 69.144 | 0.545 | 0.584 | 6.359 | 15.625 | 0.815 |
| ESMFold2-MSA | 38.712 | 62.662 | 68.971 | 0.548 | 0.598 | 6.487 | 15.880 | 0.800 |
| ESMFold2 | 31.978 | 58.547 | 65.712 | 0.507 | 0.556 | 7.087 | 17.059 | 0.784 |
| AtlasFold-M | 26.971 | 52.496 | 60.676 | 0.456 | 0.496 | 7.965 | 19.632 | 0.766 |
