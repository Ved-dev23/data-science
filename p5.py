import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
group1 = [23, 25, 29, 34, 30]
group2 = [19, 20, 22, 24, 25]
group3 = [15, 18, 20, 21, 17]
group4 = [28, 24, 26, 30, 29]
data = pd.DataFrame({'value': group1 + group2 + group3 + group4,
'group': ['Group1'] * len(group1) + ['Group2'] * len(group2) +
['Group3'] * len(group3) + ['Group4'] * len(group4)})
f_statistics, p_value = stats.f_oneway(group1, group2, group3, group4)
print("One-way ANOVA:")
print("F-statistic:", f_statistics)
print("P-value:", p_value)
tukey_results = pairwise_tukeyhsd(data['value'], data['group'])
print("\nTukey-Kramer post-hoc test:")
print(tukey_results)
