## Revision of Tabs. 13, 14, 15, and 16 in the paper.

Tables 13–16 in the paper were intended to present the ANOVA results comparing the activations of the top-1k transfer neurons (both Type-1 and Type-2) under the **language-specificity label** setting. However, we mistakenly reported the results obtained under the language-family specificity setting in these tables, which caused Tables 13–16 (intended for language specificity) and Tables 17–20 (intended for language-family specificity) to contain identical values. Therefore, we provide here the correct results computed under the language-specificity label setting.


Table 13: % of Type-1 Transfer Neurons with correlation ratio > 0.1 and p < 0.05
|| LLaMA3-8B | Mistral-7B | Aya expanse-8B |
|----------|----------:|-----------:|---------------:|
| Japanese |     0.409 |      0.381 |          0.383 |
| Dutch    |     0.054 |      0.075 |          0.023 |
| Korean   |     0.131 |      0.181 |          0.118 |
| Italian  |     0.69 |      0.072 |          0.058 |

Table 14: % of Type-2 Transfer Neurons with correlation ratio > 0.1 and p < 0.05
|| LLaMA3-8B | Mistral-7B | Aya expanse-8B |
|----------|----------:|-----------:|---------------:|
| Japanese |     0.597 |      0.604 |          0.568 |
| Dutch    |     0.586 |      0.522 |          0.496 |
| Korean   |     0.455 |      0.343 |          0.440 |
| Italian  |     0.433 |      0.442 |          0.536 |

Table 15: % of Type-1 Transfer Neurons with correlation ratio > 0.25 and p < 0.05
|| LLaMA3-8B | Mistral-7B | Aya expanse-8B |
|----------|----------:|-----------:|---------------:|
| Japanese |     0.230 |      0.249 |          0.246 |
| Dutch    |     0.014 |      0.013 |          0.007 |
| Korean   |     0.043 |      0.061 |          0.034 |
| Italian  |     0.013 |      0.011 |          0.016 |

Table 16: % of Type-2 Transfer Neurons with correlation ratio > 0.25 and p < 0.05
|| LLaMA3-8B | Mistral-7B | Aya expanse-8B |
|----------|----------:|-----------:|---------------:|
| Japanese |     0.370 |      0.445 |          0.376 |
| Dutch    |     0.330 |      0.326 |          0.345 |
| Korean   |     0.241 |      0.145 |          0.252 |
| Italian  |     0.204 |      0.268 |          0.381 |