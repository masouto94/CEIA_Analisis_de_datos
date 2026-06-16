import pandas as pd
import numpy as np
import io
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

class IQR():
    def __init__(self, serie:pd.Series):
        self.serie = serie
        self.range = self.serie.quantile(0.75) - self.serie.quantile(0.25)
        self.upper = self.serie.quantile(0.75) + 1.5 * self.range
        self.lower = self.serie.quantile(0.25) - 1.5 * self.range

    def filter_outliers(self) -> pd.Series:
        return self.serie[self.serie.between(self.lower,self.upper)].copy()
    
    def get_outliers(self) -> pd.Series:
        return self.serie[~self.serie.between(self.lower,self.upper)].copy()
    
    def get_outliers_proportion(self):
        return round((self.get_outliers().size / self.serie.size) * 100,2)
    
    def get_ranges(self):
        values = self.__dict__.copy()
        del values["serie"]
        return values

def read_multi_csv(filepath: str) -> dict[str, pd.DataFrame]:
    dataframes = {}
    current_rows = []

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped in ("", ","):
                if current_rows:
                    df = pd.read_csv(io.StringIO("\n".join(current_rows)))
                    df = df.rename(
                        columns={
                            df.columns[
                                1
                            ]: f"{df.columns[0].replace('_id','')}_{df.columns[1]}"
                        }
                    )
                    dataframes[df.columns[0]] = df
                    current_rows = []
            else:
                current_rows.append(stripped)

    if current_rows:
        df = pd.read_csv(io.StringIO("\n".join(current_rows)))
        df = df.rename(
            columns={
                df.columns[1]: f"{df.columns[0].replace('_id','')}_{df.columns[1]}"
            }
        )
        dataframes[df.columns[0]] = df

    return dataframes


def recover_feature(
    data: pd.DataFrame, missing_feature: str, index_feature: str, empty_value="?"
) -> pd.DataFrame:
    records_without_field = set(
        data[data[missing_feature] == empty_value][index_feature]
    )
    records_with_field = set(data[data[missing_feature] != empty_value][index_feature])
    records_to_impute = records_without_field.intersection(records_with_field)
    recovered_mapping = (
        data[
            (data[index_feature].isin(records_to_impute))
            & (data[missing_feature] != empty_value)
        ]
        .drop_duplicates(index_feature)
        .set_index(index_feature)[missing_feature]
    )

    mask = (data[index_feature].isin(records_to_impute)) & (
        data[missing_feature] == empty_value
    )
    data.loc[mask, missing_feature] = data.loc[mask, index_feature].map(
        recovered_mapping
    )

    return data


def group_by_condition(df, grouping_cols, target_col, condition=pd.isnull):
    grouped_data = (
        df.groupby(grouping_cols, observed=True)[target_col]
        .apply(lambda x: condition(x).mean() * 100)
        .reset_index(name=f"percentage_of_{target_col}")
    )
    return grouped_data


def eta_squared(continuous, categorical):
    mask = continuous.notna() & categorical.notna()
    y, grp = continuous[mask], categorical[mask]
    groups = [g.values for _, g in y.groupby(grp, observed=True)]
    F, p = f_oneway(*groups)
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total, p, F


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * min(k - 1, r - 1)))


def icd9_to_category(code):
    try:
        code = str(code).strip()
        if code.startswith("V"):
            return 18
        if code.startswith("E"):
            return 19

        num = float(code)
        if 250 <= num <= 250.99:
            return 20  # Diabetes (opcional separarlo)
        if 1 <= num <= 139:
            return 1
        elif 140 <= num <= 239:
            return 2
        elif 240 <= num <= 279:
            return 3
        elif 280 <= num <= 289:
            return 4
        elif 290 <= num <= 319:
            return 5
        elif 320 <= num <= 389:
            return 6
        elif 390 <= num <= 459:
            return 7
        elif 460 <= num <= 519:
            return 8
        elif 520 <= num <= 579:
            return 9
        elif 580 <= num <= 629:
            return 10
        elif 630 <= num <= 679:
            return 11
        elif 680 <= num <= 709:
            return 12
        elif 710 <= num <= 739:
            return 13
        elif 740 <= num <= 759:
            return 14
        elif 760 <= num <= 779:
            return 15
        elif 780 <= num <= 799:
            return 16
        elif 800 <= num <= 999:
            return 17
        else:
            return 0
    except:
        return 0

def get_cols(df):
    selected = []
    for col in ["diag_1",
        "diag_3",
        "diag_2",
        "num_medications",
        "num_lab_procedures",
        "number_diagnoses",
        "num_procedures",
        "age"]:
        for c in df.columns:
            if col in c:
                selected.append(c)
    return selected