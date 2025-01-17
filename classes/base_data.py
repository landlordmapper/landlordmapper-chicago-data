from datetime import datetime

import numpy as np
import pandas as pd
import os


class BaseData:

    """
    Base class for all data_root processing. Functions belong here if they can be used across the board in any workflow or
    with any dataset.
    """

    def __init__(self):
        pass

    @classmethod
    def get_timestamp(cls):
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    @classmethod
    def format_elapsed_time(cls, elapsed):
        if elapsed < 60:
            number = round(elapsed)
            units = "seconds"
        else:
            number = round(elapsed/60, 1)
            units = "minutes"
        formatted_time = {
            "number": number,
            "units": units
        }
        return formatted_time

    @classmethod
    def remove_hash_char(cls, input_string):
        return input_string.replace("#", " ")

    #------------------------------------
    #----GENERAL DATAFRAME OPERATIONS----
    #------------------------------------

    @classmethod
    def get_df(cls, path, dtype):
        try:
            return pd.read_csv(path, dtype=dtype)
        except ValueError as e:
            print(f"ValueError while loading {path}: {e}")
            df = pd.read_csv(path)
            for col, col_type in (dtype or {}).items():
                if col_type == int:
                    try:
                        df[col] = df[col].astype(int)
                    except ValueError:
                        print(f"Column '{col}' contains NaN or non-integer values, unable to convert to int.")
            return df

    @classmethod
    def print_cols(cls, df, sort=False):
        if sort:
            for col in sorted(df.columns):
                print(col)
        else:
            for col in df.columns:
                print(col)

    @classmethod
    def get_subset(cls, df, col, val):
        return df[df[col] == val]

    @classmethod
    def remove_subset(cls, df, col, collection):
        df_out = df[~df[col].isin(collection)]
        return df_out

    @classmethod
    def get_num_unique(cls, df, colname):
        return len(df[colname].dropna().unique())

    @classmethod
    def save_df(cls, df, path, data_root=None):
        df.to_csv(path, index=False)
        if data_root:
            print("Success. Dataset saved at the following location:", path.replace(data_root, "DATA_ROOT"))


    @classmethod
    def get_duplicates(cls, df, col_name):
        """
        Returns a DataFrame containing only rows where the specified column has duplicate values.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The column name to check for duplicates.

        Returns:
        pd.DataFrame: A DataFrame containing only the rows with duplicate values in the specified column.
        """
        # Find the duplicate values in the specified column
        duplicate_values = df[df[col_name].duplicated(keep=False)]
        return duplicate_values

    @classmethod
    def combine_columns(cls, c1, c2):
        if pd.isnull(c1) == True:
            return c2
        else:
            return c1

    # removes duplicated, redundant columns (based on _x and _y auto-added suffixes)
    @classmethod
    def combine_columns_parallel(cls, df):
        for column in df.columns:
            if column[-2:] == '_x':
                df[column[:-2]] = df.apply(lambda x: cls.combine_columns(x[column], x[column[:-2] + '_y']), axis =1)
                df.drop(columns = [column, column[:-2] + '_y'], inplace = True)
        return df

    @classmethod
    def clean_column_name(cls, col_name):
        return col_name.replace(' ', '_').replace(":", "").replace("/", "_").upper()

    @classmethod
    def fix_column_names(cls, df):

        # df = pd.read_csv(filepath)
        cleaned_columns = [cls.clean_column_name(col) for col in df.columns]

        # Handle duplicates by appending "_2", "_3", etc.
        seen = {}
        for i, col in enumerate(cleaned_columns):
            if col in seen:
                seen[col] += 1
                cleaned_columns[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 1

        # Rename the columns in the DataFrame
        df.columns = cleaned_columns
        print(cleaned_columns)

        return df

    @classmethod
    def set_colnames_upper(cls, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [col.upper() for col in df.columns]
        return df

    @classmethod
    def make_cols_upper(cls, df):
        new_cols = [col.upper() for col in df.columns]
        new_cols_remove_special = [col.replace(":", "") for col in new_cols]
        df.columns = new_cols_remove_special
        return df

    @classmethod
    def combine_partials(cls, directory, dtypes):
        """
        Returns concatenation of partial dataframes saved as a result of scraping, geocodio validation, api calls, etc.

        Allows for saving paid API call or scrape URL calls in small chunks so as not to lose data_root in the case of
        network interruption or errors thrown during execution.
        """
        files = os.listdir(directory)
        if len(files) == 0: return None
        dfs = []
        for file in files:
            if file.endswith(".csv"):
                path = f"{directory}/{file}"
                df = pd.read_csv(path, dtype=dtypes)
                dfs.append(df)
        df_combined = pd.concat(dfs, ignore_index=True)
        return df_combined


    #------------
    #----MISC----
    #------------

    @classmethod
    def is_int(cls, string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    @classmethod
    def set_zip_main(cls, raw_zip):
        """
        Fixes zip codes that the raw data_root includes as 9-digit by separating out and returning the first 5
        """
        zip_str = str(raw_zip)
        if len(zip_str) > 5:
            return zip_str[0:5]
        return raw_zip

    @classmethod
    def set_zip_4(cls, raw_zip):
        """
        Pulls out and returns the last 4 digits of a 9-digit zip code
        """
        zip_str = str(raw_zip)
        if len(zip_str) > 5:
            return zip_str[5:]
        return np.nan

    @classmethod
    def fix_zip(cls, zip_string):
        try:
            zip_split = zip_string.split(".")
            return zip_split[0]
        except:
            return zip_string

    @classmethod
    def is_encoded_empty(cls, x):
        if isinstance(x, str):
            # Check if string contains mostly non-printable characters
            return any(ord(c) < 32 for c in x)
        return False

    @classmethod
    def save_partial(cls, records, interval, partial_fn):
        if len(records) == interval:
            partial_fn(records)
            return []
        else:
            return records

