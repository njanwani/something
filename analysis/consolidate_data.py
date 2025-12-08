import pandas as pd
from scipy.stats import wilcoxon, binomtest
sheets = pd.read_excel("analysis/Responses.xlsx", sheet_name=None)
list(sheets.keys())