import json
import string
from datetime import datetime
from typing import List
from xml.etree.ElementInclude import include

import numpy as np
import word2number as w2n
import pandas as pd
import re
import time
from collections import Counter

import Levenshtein as lev
import networkx as nx
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.dialects.mssql.information_schema import columns
from tqdm.notebook import tqdm

from .base_data import BaseData
from constants.landlord_fields import LandlordFields
from constants.constants import LANDLORD_DTYPES
from constants.constants import DATA_ROOT


class BaseLandlordData(BaseData):
    """
    All subclasses to this base class represent a category of data operations related to landlord data processing
    workflows. The subclasses exist to group functions that specifically serve the different operation categories.

    Functions that are defined in this base class, as opposed to in a category subclass, meet one of the following
    criteria:

      1. They are used across multiple workflow categories
      2. They do not neatly fit into any workflow category
    """

    RENTAL_CODES = [
        "211", "219", "225", "313", "314", "315", "318", "391", "396", "397", "399", "913", "914", "915", "918", "959",
        "991", "996"
    ]

    UNIQUE_KEYS = [
        'CIR ', 'APARTMENTS ', 'SERVICES ', 'INVESTMENTS ', 'HOLDINGS ',
        'LN ', 'COMPANY ', 'AUTHORITY ', 'INC ', 'FORECLOSURE ',
        'ESTABLISHED ', 'CONDO TRUST ', 'COOPERATIVE ', 'PARTNERS ', 'CR ',
        'PARTNERSHIP ', 'GROUP ', 'ASSOCIATION ', 'TRUSTEES ', 'TRUST ',
        'PROPERTIES ', 'MANAGEMENT ', 'SQUARE ', 'MANAGERS ', 'EXCHANGE ',
        'REAL ESTATE ', 'DEVELOPMENT ', 'REDEVELOPMENT ', 'MORTGAGE ',
        'RESIDENTIAL ', 'REALTY TRUST ', 'CORPORATION ', 'LIMITED ', 'LLC ',
        'ORGANIZATION ', 'REALTY ', 'PRT ', 'VENTURE ', 'RENTAL ', 'UNION ',
        'CONDO '
    ]

    aa = LandlordFields.AddressAnalysisFields

    def __init__(self):
        super().__init__()

    #-------------------------------
    #----BUILDING CLASS FUNCTIONS----
    #-------------------------------

    # used to assign/associate class code for buildings, general prerequisite for rental property subsetting
    # don't fit into any subclass clearly, therefore belong here
    @classmethod
    def fix_less_6(cls, df_less_6):

        # define new column where combined classes will be stored
        df_less_6["CLASS_COMBINED"] = ""

        # fetch ONLY pins that appear more than once in the original dataset, store in sub df
        df_counts = df_less_6["PIN"].value_counts().reset_index()
        df_mult_pins = df_counts[df_counts["count"] > 1]
        pins = df_mult_pins["PIN"].tolist()
        df_pins = df_less_6[df_less_6["PIN"].isin(pins)]

        # for each pin, fetch unique class codes and store in new column
        for pin in pins:
            df_pin = df_pins[df_pins["PIN"] == pin]
            classes = df_pin["CLASS"].unique().tolist()
            if len(classes) > 1:
                classes_joined = ", ".join(classes)
                df_less_6.loc[df_less_6["PIN"] == pin, "CLASS_COMBINED"] = classes_joined

        return df_less_6

    @classmethod
    def fix_greater_7(cls, df_greater_7):

        # remove dashes from class codes
        df_greater_7["KEYPIN"] = df_greater_7["KEYPIN"].apply(lambda x: x.replace('-', '') if isinstance(x, str) else "")

        # define new column where combined classes will be stored
        df_greater_7["CLASS_COMBINED"] = ""

        # fetch ONLY pins that appear more than once in the original dataset, store in sub df
        df_counts = df_greater_7["KEYPIN"].value_counts().reset_index()
        df_mult_pins = df_counts[df_counts["count"] > 1]
        pins = df_mult_pins["KEYPIN"].tolist()
        df_pins = df_greater_7[df_greater_7["KEYPIN"].isin(pins)]

        # for each pin, fetch unique class codes and store in new column
        for pin in pins:

            df_pin = df_pins[df_pins["KEYPIN"] == pin]
            classes_list = df_pin["CLASS(ES)"].dropna().unique().tolist()
            classes = []
            for clas in classes_list:
                class_split = clas.split(',')
                for cl in class_split:
                    cl_strip = cl.strip()
                    if cl_strip not in classes:
                        classes.append(cl_strip)

            if len(classes) > 1:
                classes_joined = ", ".join(classes)
                df_greater_7.loc[df_greater_7["KEYPIN"] == pin, "CLASS_COMBINED"] = classes_joined

        df_greater_7["CLASS_COMBINED"] = df_greater_7["CLASS_COMBINED"].apply(lambda x: x.replace('-', '') if isinstance(x, str) else "")

        return df_greater_7

    @classmethod
    def fix_classes(cls, x):
        x_split = x.split("-")
        x_val = x_split[0].strip()
        if x_val == "EX" or not cls.is_int(x_val):
            return ""
        else:
            return x_val

    @classmethod
    def combine_classes(cls, x, y, z):
        final_classes = []
        if pd.notnull(x) and x != "":
            final_classes.append(str(x).strip())
        if pd.notnull(y) and y != "":
            final_classes.append(str(y).strip())
        if pd.notnull(z) and z != "":
            final_classes.append(str(z).strip())
        if len(final_classes) > 0:
            return ",".join(final_classes)
        else:
            return ""

    @classmethod
    def fix_class_codes(cls, x):
        codes = x.split(",")
        final_codes = []
        for code in codes:
            code_s = str(code).strip()
            if len(code_s) > 3:
                final_codes.append(code[:3])
                final_codes.append(code[3:])
            else:
                final_codes.append(str(code).strip())
        return ",".join(list(set(final_codes)))

    @classmethod
    def is_rental(cls, x, rental_codes):
        codes = [code.strip() for code in x.split(",")]
        return any(code in rental_codes for code in codes)


    #--------------------------------
    #----GENERAL CLEANING METHODS----
    #--------------------------------
    # general string cleaning methods that could be used for both taxpayer AND corp/llc cleaning
    @classmethod
    def delete_symbols_spaces(cls, text):
        try:
            text = text.replace('&', 'and')
            text = text.replace(",", " ")
            text = text.replace(".", " ")
            # text = text.replace('-', ' ')
            text = text.translate(str.maketrans(string.punctuation.replace('/','').replace('-',''), ' '*len(string.punctuation.replace('/','').replace('-',''))))
            text = text.replace('/', ' / ')
            text = [x.strip() for x in text.split(' ')]
            text = [x for x in text if x]
            text = ' '.join(text).upper()
            return text
        except:
            return text

    @classmethod
    def dedup_words(cls, text):
        text_list = text.split()
        if len(set(text_list)) < len(text_list):
            single_words = []
            double_words = []
            for val in text_list:
                if val not in single_words:
                    single_words.append(val)
                else:
                    double_words.append(val)
            de_dups = []

            for i, val in enumerate(text_list):
                if val in double_words:
                    de_dups.append(i)

            text_list.pop(min(de_dups))
        return ' '.join(text_list)

    @classmethod
    def switch_the(cls, text):
        if text[-4:] == ' THE':
            return 'THE ' + text[:-4]
        else:
            return text

    @classmethod
    def convert_ordinals(cls, text):
        try:
            if (type(cls.words_to_num(text.split('TH')[0])) == int) and (text[-2:] == 'TH'):
                text = str(cls.words_to_num(text.split('TH')[0])) + 'TH'
            return text
        except:
            return text

    @classmethod
    def take_first(cls, text):
        try:
            text = re.findall(r'\d+-\d+', text)[0].split('-')[0] + re.sub(r'\d+-\d+', '', text)
        except:
            return text
        return text

    @classmethod
    def drop_letters(cls, text):
        try:
            text = re.sub(r'\d+[a-zA-Z]', re.findall(r'\d+[a-zA-Z]', text)[0][:-1], text)
        except:
            return text
        return text

    @classmethod
    def drop_floors(cls, text):
        try:
            text = re.sub(r' \d+D','',text)
        except:
            return text
        return text

    @classmethod
    def combine_numbers(cls, text_list):

        whole_list = []
        start = False
        end = False
        text_list.append('JUNK')
        numbers = []

        for p in text_list:
            try:
                int(p)
                numbers.append(p)
                start = True
            except:

                if start is True:
                    end = True
                else:
                    end = False
                    whole_list.append(p)
            if start and end:

                if len(numbers) == 1:
                    whole_list.append(numbers[0])
                elif len(numbers) == 2:
                    if str(numbers[0])[-1:] == '0' and str(numbers[1])[-1:] != '0':
                        complete = str(numbers[0] + numbers[1])
                    else:
                        complete = str(numbers[0]) + str(numbers[1])
                    whole_list.append(complete)
                else:
                    if str(numbers[0])[-1:] == '0':
                        complete = str(numbers[0][:-1] + numbers[1])
                    else:
                        complete = str(numbers[0]) + str(numbers[1])
                    for i in numbers[2:]:
                        if complete[-1:] == '0':
                            complete = str(int(complete) + int(i))
                        else:
                            complete = complete + str(i)
                    whole_list.append(complete)
                numbers = []
                start = False
                end = False
                whole_list.append(p)

        return whole_list[:-1]

    @classmethod
    def words_to_num(cls, text):
        if text == 'POINT':
            return text
        else:
            try:
                return w2n.word_to_num(text)
            except:
                return text

    @classmethod
    def convert_mixed(cls, text):
        try:
            convert = re.sub(r'\d', '', text)
            if type(cls.words_to_num(convert)) == int:
                text = re.sub(r'{}'.format(convert), ' ' + str(cls.words_to_num(convert)) + ' ', text)
                text = text.strip()
            return text
        except:
            return text


    #--------------------------------
    #----ADDRESS CLEANING METHODS----
    #--------------------------------

    # these could be used by taxpayer OR corp/llc cleaners
    @classmethod
    def convert_nesw(cls, text):
        try:
            directions = {'NORTH':' N ', 'SOUTH': ' S ', 'EAST':' E ', 'WEST':' W '}
            if any([x for x in ['NORTH', 'SOUTH', 'EAST', 'WEST'] if x in text]):
                for direction in [x for x in ['NORTH', 'SOUTH', 'EAST', 'WEST'] if x in text]:
                    text = text.replace(direction, directions[direction])
                    text = ' '.join([x.strip() for x in text.split()])
            return text
        except:
            return text

    @classmethod
    def convert_st(cls, text):
        try:
            text = text.split('#')[0]
            text = text.split('APT')[0]
            text = text.split('UNIT')[0]
            text = text.split('FLOOR')[0]
            text = text.split('SUITE')[0]
            return text
        except:
            return text

    @classmethod
    def change_NESW(cls, text):
        try:
            directions = {'N ': 'NORTH ', 'E ':'EAST ', 'W ':'WEST ', 'S ':'SOUTH '}
            if text[:2] in ['E ', 'W ', 'S ','N ']:
                text = directions[text[:2]] + text[2:]
            return text
        except:
            return text

    @classmethod
    def convert_zip(cls, text):
        try:
            text = str(int(text)).zfill(5)
        except:
            return ''
        return text


    #-----------------------------------
    #----NETWORK / STRING MATCH BASE----
    #-----------------------------------

    # used by string matching AND network analysis subclasses
    @classmethod
    def check_address(
            cls,
            address: str | None,
            df_analysis: pd.DataFrame,
            include_orgs: bool,
            include_unresearched: bool
    ) -> bool:
        """
        Returns "True" if the address SHOULD be included in the network analysis, and False if it should be ignored.
        """
        if not address:
            return False

        # the address has NOT already been analyzed
        if address not in list(df_analysis["ADDRESS"].dropna().unique()):
            return include_unresearched

        # address has been analyzed - check if it should be ignored
        else:
            df_addr_analysis = df_analysis[df_analysis["ADDRESS"] == address]
            if df_addr_analysis[cls.aa.IS_LANDLORD_ORG].eq("t").any():
                return include_orgs
            elif df_addr_analysis[cls.aa.IS_LAWFIRM].eq("t").any():
                return False
            elif df_addr_analysis[cls.aa.IS_MISSING_SUITE].eq("t").any():
                return False
            elif df_addr_analysis[cls.aa.IS_FINANCIAL_SERVICES].eq("t").any():
                return False
            elif df_addr_analysis[cls.aa.IS_VIRTUAL_OFFICE_AGENT].eq("t").any():
                return False
            elif df_addr_analysis[cls.aa.FIX_ADDRESS].eq("t").any():
                return False
            elif df_addr_analysis[cls.aa.IS_IGNORE_MISC].eq("t").any():
                return False
            else:
                return True

    @classmethod
    def set_cleaned_address(cls, row):
        if pd.isna(row["GCD_FORMATTED_MATCH"]) or row["GCD_FORMATTED_MATCH"].strip() == "":
            return row["RAW_ADDRESS"]
        else:
            return row["GCD_FORMATTED_MATCH"]

    @classmethod
    def get_addrs_analysis_subset(cls, df_component, address_col, address_analysis_master_path):
        """
        Accepts dataframe with a gcd-validated address column, returns dataframe subset containing all addresses NOT
        found in the address_analysis_master file
        """
        # fetch list of addresses already analyzed
        df_analysis_master = cls.get_df(address_analysis_master_path, LANDLORD_DTYPES)
        addresses_analyzed = list(df_analysis_master["ADDRESS"].dropna().unique())

        # pull out ONLY addresses from df_component not found in the list
        df_component_subset = df_component[~df_component[address_col].isin(addresses_analyzed)]
        df_analysis_out = pd.DataFrame(list(df_component_subset[address_col]), columns=["ADDRESS"])

        for col in cls.aa.COLS_LIST[1:]:
            df_analysis_out[col] = None

        if len(df_analysis_out) == 0:
            print("All addresses from component have already been analyzed.")
        else:
            cls.save_df(
                df=df_analysis_out,
                path=f"{DATA_ROOT}/address_analysis/to_analyze_{datetime.now().strftime("%Y_%m_%d_%H%M")}.csv",
                data_root=DATA_ROOT
            )
            print("Addresses to be analyzed have been successfully saved.")

    @classmethod
    def rename_gcd_cols(cls, df, suffix):

        df.rename(columns={
            "GEOCODIO_SUCCESS?": f"GEOCODIO_SUCCESS_{suffix}",
            "GCD_NUMBER": f"GCD_NUMBER_{suffix}",
            "GCD_PREDIRECTIONAL": f"GCD_PREDIRECTIONAL_{suffix}",
            "GCD_PREFIX": f"GCD_PREFIX_{suffix}",
            "GCD_STREET": f"GCD_STREET_{suffix}",
            "GCD_SUFFIX": f"GCD_SUFFIX_{suffix}",
            "GCD_POSTDIRECTIONAL": f"GCD_POSTDIRECTIONAL_{suffix}",
            "GCD_SECONDARYUNIT": f"GCD_SECONDARYUNIT_{suffix}",
            "GCD_SECONDARYNUMBER": f"GCD_SECONDARYNUMBER_{suffix}",
            "GCD_CITY": f"GCD_CITY_{suffix}",
            "GCD_COUNTY": f"GCD_COUNTY_{suffix}",
            "GCD_STATE": f"GCD_STATE_{suffix}",
            "GCD_ZIP": f"GCD_ZIP_{suffix}",
            "GCD_COUNTRY": f"GCD_COUNTRY_{suffix}",
            "GCD_X": f"GCD_XCOORD_{suffix}",
            "GCD_Y": f"GCD_YCOORD_{suffix}",
            "GCD_ACCURACY": f"GCD_ACCURACY_{suffix}",
            "GCD_FORMATTED_ADDRESS": f"GCD_FORMATTED_ADDRESS_{suffix}"
        }, inplace=True)

        return df

    @classmethod
    def merge_gcd_orgs(cls, df_main, df_merge, left_on, dup_col_drop):
        if "Unnamed: 0" in df_main.columns:
            df_main.drop(columns="Unnamed: 0", inplace=True)
        if "Unnamed: 0" in df_merge.columns:
            df_merge.drop(columns="Unnamed: 0", inplace=True)
        df_main_merged = pd.merge(df_main, df_merge, how="left", left_on=left_on, right_on="RAW_ADDRESS")
        df_main_merged = df_main_merged.drop_duplicates(subset=[dup_col_drop])
        df_main_merged = cls.rename_gcd_cols(df_main_merged, left_on)
        df_main_merged = df_main_merged.drop(columns=["RAW_ADDRESS"])
        return df_main_merged


    @classmethod
    def add_formatted_address_matching(cls, row, suffix=""):
        """
        Removes secondary unit descriptor for address matching.
        Example: 123 Oak St Ste 3B  |  123 Oak St Unit 3B  =  123 Oak St 3B
        """
        if pd.notnull(row[f"GCD_FORMATTED_ADDRESS{suffix}"]) and pd.notnull(row[f"GCD_SECONDARYUNIT{suffix}"]):
            unit = row[f"GCD_SECONDARYUNIT{suffix}"]
            gcd_formatted_split = row[f"GCD_FORMATTED_ADDRESS{suffix}"].split(",")
            gcd_formatted_split[1] = gcd_formatted_split[1][len(unit)+1:]
            gcd_formatted_match_addr = "".join(gcd_formatted_split[0:2])
            gcd_formatted_match = ",".join([gcd_formatted_match_addr] + gcd_formatted_split[2:])
            return gcd_formatted_match
        elif pd.notnull(row[f"GCD_FORMATTED_ADDRESS{suffix}"]):
            return row[f"GCD_FORMATTED_ADDRESS{suffix}"]
        else:
            return np.nan

    @classmethod
    def fix_addrs_to_exclude(cls, addrs_to_exclude):
        df_addrs = pd.DataFrame(addrs_to_exclude, columns=["GCD_FORMATTED_ADDRESS"])
        return df_addrs

    @classmethod
    def fix_xycoord(cls, row, xy):
        if xy == "x":
            x = row["GCD_X"]
            x_coord = row["GCD_XCOORD"]
            if pd.notnull(x) and pd.isnull(x_coord):
                return x
            elif pd.isnull(x) and pd.notnull(x_coord):
                return x_coord
            else:
                return np.nan
        elif xy == "y":
            y = row["GCD_Y"]
            y_coord = row["GCD_YCOORD"]
            if pd.notnull(y) and pd.isnull(y_coord):
                return y
            elif pd.isnull(y) and pd.notnull(y_coord):
                return y_coord
            else:
                return np.nan
        return np.nan

    @classmethod
    def check_secondary_number_raw(cls, row):
        raw_address = row["RAW_ADDRESS"].split(",")[0].strip()
        match = re.search(r"(\d+)$", raw_address)
        if match:
            return match.group(1)
        else:
            return np.nan

    @classmethod
    def fix_unit_num_formatted(cls, row):

        gcd_formatted_split = row[f"GCD_FORMATTED_ADDRESS"].split(",")
        gcd_sec_unit = row[f"GCD_SECONDARYUNIT"]
        gcd_sec_number = row[f"GCD_SECONDARYNUMBER"]

        gcd_city = row["GCD_CITY"]
        gcd_state = row["GCD_STATE"]
        gcd_zip = row["GCD_ZIP"]

        if pd.notnull(gcd_sec_unit):
            unit_number = f"{gcd_sec_unit.strip()} {gcd_sec_number.strip()}"
            new_formatted_addr = f"{gcd_formatted_split[0].strip()}, {unit_number}"
        else:
            new_formatted_addr = f"{gcd_formatted_split[0].strip()} {gcd_sec_number.strip()}"

        new_formatted_addr_split = new_formatted_addr.split()
        if new_formatted_addr_split[-1].strip() == new_formatted_addr_split[-2].strip():
            new_formatted_addr_split.pop()  # Remove the last element
            new_formatted_addr = " ".join(new_formatted_addr_split)

        return f"{new_formatted_addr}, {gcd_city}, {gcd_state} {gcd_zip}"





class CleanTaxpayer(BaseLandlordData):

    """
    Subclass for property taxpayer data. This class shall contain all logic for cleaning property taxpayer records,
    arranged by the use with either taxpayer names or addresses. Methods that can be used by both belong in the
    parent class (CleanBase).

    Functions in this class should be used exclusively for cleaning taxpayer records. Functions used in the taxpayer
    workflow but that have applications elsewhere should be included in the base class.

    """


    TRUSTS = [
        "ATRUS", "TRUSTLAN", "TRUSTEE", "TRUSTE", "TRUSTTRU", "TRUSTU", "TRUSTAS", "TRUST", "TRUS", "TRU", "TRST", "TR",
        "TTEE", "TT"
    ]
    TRUSTS_STRINGS = [
        "TRUSTEE", "TRUSTE", "TRUSTTRU", "TRUSTU", "TRUSTAS", "TRUST"
    ]
    TRUST_COMPANIES_IDS = [
        "CHICAGO TITLE LAND TRUST COMPANY",
        "ATG TRUST COMPANY",
        "BMO HARRIS TRUST",
        "FIRST MIDWEST BANK TRUST",
        "PARKWAY BANK TRUST",
        "OLD NATIONAL TRUST",
        "MARQUETTE BANK TRUST"
    ]

    def __init__(self):
        super().__init__()

    @classmethod
    def get_corp_abb(cls):

        corp_abb = dict()
        corp_abb.update(dict.fromkeys(["ACQUISITIONS"], "ACQUISITION"))
        corp_abb.update(dict.fromkeys(["APTS", "APTMTS", "APT"], "APARTMENTS"))
        corp_abb.update(dict.fromkeys(["ASSOC", "ASSN", "ASSOCIATIO", "ASSOCS", 'ASSOCIATES', 'ASSOCIATE'], "ASSOCIATION"))
        corp_abb.update(dict.fromkeys(["AUTH", "AUTHOR", "AUT"], "AUTHORITY"))
        corp_abb.update(dict.fromkeys(["BROS"], "BROTHERS"))
        corp_abb.update(dict.fromkeys(["COM", "CMMTY", "CMUNITY", "COMUNITY", "COMMUNITES", "COMMUNITIES"],"COMMUNITY"))
        corp_abb.update(dict.fromkeys(["CO-OPERATIVE", "CO OPERATIVE", "COOP", "CO-OP", "COOPERA"], "COOPERATIVE"))
        corp_abb.update(dict.fromkeys(["CO OPERATIVE", "COMPANY OP ", "COOPERATIVE HOUSING CORPORATION "], "COOPERATIVE"))
        corp_abb.update(dict.fromkeys(["CO", "COMANY", "COP"], "COMPANY"))
        corp_abb.update(dict.fromkeys(["CONDO T"], "CONDO TRUST"))
        corp_abb.update(dict.fromkeys(["CONDOMINIUM", "CONDOMINIUMS", "CONDOS", "CONDOS", "COND", 'CD'], "CONDO"))
        corp_abb.update(dict.fromkeys(["CTR", "CNTR"], "CENTER"))
        corp_abb.update(dict.fromkeys(["DEV", "DVLPMNT", "DEVLPMNT", "DEVELO", "DEVELOPME", "DEVLOP","DELVELOPMENT", "DEVELOP", "DEVELOPMNT", "DEVELOPMEN", "DEVE", "DEVELOPMENTS"], "DEVELOPMENT"))
        corp_abb.update(dict.fromkeys(["EXCHNG", "XCHNGE", "XCHNG", "EXCH", "EXC"],"EXCHANGE"))
        corp_abb.update(dict.fromkeys(["FAM"], "FAMILY"))
        corp_abb.update(dict.fromkeys(["FCL"], "FORECLOSURE"))
        corp_abb.update(dict.fromkeys(["GROUPS", "GRP", "GR"], "GROUP"))
        corp_abb.update(dict.fromkeys(["HOLDING", "HLDNG"], "HOLDINGS"))
        corp_abb.update(dict.fromkeys(["HOPITAL"], "HOSPITAL"))
        corp_abb.update(dict.fromkeys(["HOSP"], "HOSPITAL"))
        corp_abb.update(dict.fromkeys(["IN", "INIT", "INTV"], "INITIATIVE"))
        corp_abb.update(dict.fromkeys(["INC.", "INCOR", "INCRP", "INCORPORATED", "INCORPORTED"], "INC"))
        corp_abb.update(dict.fromkeys(["INVESMENT", "INVESTMENT", "INVES", "INVESTEMENTS", "INVESTMNT", "INVESTMNTS", "INVEST", 'INV'], "INVESTMENTS"))
        corp_abb.update(dict.fromkeys(["LIMITED PARTNERSHIP", "LIMITED PARTNER", "LIMITED PARTNERS", "LPS", "LP", "LMTD", "LTD", "LLP"], "LIMITED"))
        corp_abb.update(dict.fromkeys(["LIMITED PARTNERSHIP", "LIMITED PARTNER ", "LIMITED PARTNERS "], "LIMITED"))
        corp_abb.update(dict.fromkeys(["L L C", "L.L.C.", "PLLC", "LLLC", "LL"], "LLC"))
        corp_abb.update(dict.fromkeys(["MED"], "MEDICAL"))
        corp_abb.update(dict.fromkeys(["MNGMT", "MGMT", "MANAG", "MGT", "MNGT", "MGMNT", "MGNT", "MANAGE", "MNGMNT", "MANAGEMNT", "MANAGEMEN", "MANGEMENT"], "MANAGEMENT"))
        corp_abb.update(dict.fromkeys(["MNGRS", "MGRS", "MANAGRS", "MNGR", "MGR", "MNGMT", "MANAGR", "MANAGER"], "MANAGERS"))
        corp_abb.update(dict.fromkeys(["MORTG", "MTG", "MTGS", "MORTGAGES", "MORTGAG"], "MORTGAGE"))
        corp_abb.update(dict.fromkeys(["NAT", "NTL", "NATL"], "NATIONAL"))
        corp_abb.update(dict.fromkeys(["PARTNERSIP"], "PARTNERSHIP"))
        corp_abb.update(dict.fromkeys(["PRTNS", "PTNRS", "PTNR", "PRTN", "PRTNR", "PRTNRS", "PTN", "PTNS", "PARTNER", "PARTNERDS"], "PARTNERS"))
        corp_abb.update(dict.fromkeys(["PTY", "PTYS", "PTIES", "PROP", "PROPERTY", "PROPERT", "PROPERTI", "PRPRTY"], "PROPERTIES"))
        corp_abb.update(dict.fromkeys(["REAL EST"], "REAL ESTATE"))
        corp_abb.update(dict.fromkeys(["REALTY T", "RT"], "REALTY TRUST"))
        corp_abb.update(dict.fromkeys(["REDVLPMNT", "REDEVLPMNT", "REDEVELPMENT", "REDEVELPMNT", "REDEVEL", "REDEV"], "REDEVELOPMENT"))
        corp_abb.update(dict.fromkeys(["RENT", "RENTALS", "RNTL"],"RENTAL"))
        corp_abb.update(dict.fromkeys(["RESIDENT", "RESIDENTS", "RESID"],"RESIDENTIAL"))
        corp_abb.update(dict.fromkeys(["RLTY", "RE", "REL"], "REALTY"))
        corp_abb.update(dict.fromkeys(["SERV"], "SERVICES"))
        corp_abb.update(dict.fromkeys(["SOC"], "SOCIETY"))
        corp_abb.update(dict.fromkeys(["SYST"], "SYSTEM"))
        corp_abb.update(dict.fromkeys(["THE"], ""))
        corp_abb.update(dict.fromkeys(["TR", "TRUSTS", "TRU", "TRUS", "TS", "TRS", "TRSTS", "TRST"], "TRUST"))
        corp_abb.update(dict.fromkeys(["TRSTEES"], "TRUSTEES"))
        corp_abb.update(dict.fromkeys(["VENTURES"], "VENTURE"))

        return corp_abb

    @classmethod
    def get_other_abb(cls):
        abb = dict()
        abb.update(dict.fromkeys(["REALTY T "], "REALTY TRUST"))
        abb.update(dict.fromkeys(["LIMITED PARTNERSHIP ", "LIMITED PARTNER ", "LIMITED PARTNERS "], "LIMITED"))
        abb.update(dict.fromkeys(["CONDO T "], "CONDO TRUST"))
        abb.update(dict.fromkeys(["CO OPERATIVE ", "COMPANY OP ", "COOPERATIVE HOUSING CORPORATION "], "COOPERATIVE"))
        abb.update(dict.fromkeys(["BOSTON HOUSING AUTH ", " BHA "], "BOSTON HOUSING AUTHORITY"))
        abb.update(dict.fromkeys(["ROMAN CATH "], "ROMAN CATHOLIC"))
        abb.update(dict.fromkeys(["MASSACHUSETTS CORPORATION "], ""))
        abb.update(dict.fromkeys([" N "], "NORTH"))
        abb.update(dict.fromkeys([" S "], "SOUTH"))
        abb.update(dict.fromkeys([" E "], "EAST"))
        abb.update(dict.fromkeys([" W "], "WEST"))
        abb.update(dict.fromkeys([" 1 "], "I"))
        abb.update(dict.fromkeys(['DEVELOPMENTILIMITED'], 'DEVELOPMENT LIMITED'))
        return abb

    @classmethod
    def get_banks(cls):

        banks = dict()
        banks.update(dict.fromkeys(
            [
                "ALBANY BANK CONTR DEPT",
                "ALBANY BANK NO",
                "ALBANY BANK",
            ],
            "ALBANY BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "ALBANY BANK & TRUSTAS",
                "ALBANY BANK & TRUSTLAN",
                "ALBANY BANK & TRUSTTRU",
                "ALBANY BANK & TRUSTU T",
                "ALBANY BANK & TRUST AS",
                "ALBANY BANK & TRUST CO",
                "ALBANY BANK & TRUST NA",
                "ALBANY BANK & TRUST TR",
                "ALBANY BANK & TRUST N",
                "ALBANY BANK & TRUST",
                "ALBANY BANK AND TRUST NA",
                "ALBANY BANK TRUST BENE",
                "ALBANY BANK TRUST CO",
                "ALBANY BANK TRUST",
                "ALBANY BANK TR",
                "ALBANY BANK TUT",
                "ALBANY BANK AND TRUST",
                "ALBANK TRUST",
                "ALBANK TR",
                "ALBANY BK TR"
            ],
            "ALBANY BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "BANK FINANCIAL FSB",
                "BANK FINANCIAL NATIONA",
                "BANK FINANCIAL NA AS T",
                "BANK FINANCIAL NA LAND",
                "BANK FINANCIAL NA TD",
                "BANK FINANCIAL NA",
                "BANK FINANCIAL 101",
                "BANK FINANCIAL",
                "BANK FINANACIAL",
                "BANKFINANCIAL NATIONAL",
                "BANKFINANCIAL NA"
            ],
            "BANKFINANCIAL"
        ))
        banks.update(dict.fromkeys(
            [
                "BANK FINANCIAL FSB TRU",
                "BANK FINANCIAL NA AS T",
                "BANK FINANCIAL TRUST",
                "BANK FINANCIAL TR",
                "BANKFINANCIAL TRUST",
                "BANKFINANCIAL TR",
                "BANKFINANCIAL AS TRUST"
            ],
            "BANKFINANCIAL TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "BANK OF AMERICA NATION",
                "BANK OF AMERICA NA FOR",
                "BANK OF AMERICA NA",
                "BK OF AMER",
                "BK OF AM"
            ],
            "BANK OF AMERICA"
        ))
        banks.update(dict.fromkeys(
            [
                "BMO BANK NA",
                "BMO HARRIS BANK NA",
            ],
            "BMO HARRIS"
        ))
        banks.update(dict.fromkeys(
            [
                "BMO HARRIS BANK TR",
                "HARRIS BK TR UT"
            ],
            "BMO HARRIS TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "BRIDGEVIEW BANK & TR",
                "BRIDGEVIEW BANK GROUP",
                "BRIDGEVIEW BANK GRP"
            ],
            "BRIDGEVIEW BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "BRIDGEVIEW BANK & TR",
            ],
            "BRIDGEVIEW BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                'CITIBANK NATIONAL ASSOCIATION',
                "CITIBANK NA",
                "CIT BANK NA",
            ],
            'CITIBANK'
        ))
        banks.update(dict.fromkeys(
            [
                "CITIBANK NA AS TRUSTEE"
                'CITIBANK NA TRUST'
            ],
            'CITIBANK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                "COMMUNITY SAV BK LT",
                "COMMUNITY SAV BK TR",
                "COMMUNITY SAV BK",
                "COMMUNITY SAVINGS BANK",
                "COMMUNITY SAV BANK",
                "COMMUNITY BK TR LT",
                "COMM SAVINGS BK LT",
                "COMM SAVGS BK LT",
            ],
            "COMMUNITY SAVINGS BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "COMMUNITY SAV BK TR",
                "COMMUNITY BK TR LT",
            ],
            "COMMUNITY SAVINGS BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "COOK COUNTY LAND BANK",
                "COOK COUNTY LANK BANK",
            ],
            "COOK COUNTY LAND BANK"
        ))
        banks.update(dict.fromkeys(
            [
                'DEUTSCHE BANK NATIONAL ASSOCIATION',
                'DEUTSCHE BANK NATIONAL',
                "DUETSCHE BANK NATIOAL",
                "DEUTSCHE BANK PHH MTG"
            ],
            'DEUSTCHE BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'DEUTSCHE NATIONAL BANK TRUST COMPANY TRUST',
                'DEUSTCHE BANK NATIONAL TRUST COMPANY',
                'DEUTSCHE BANK NATIONAL TRUST COMPANY',
                'DEUTSCHE BANK NATIONAL TRUST TRUST',
                'DEUTSCHE BANK NATIONAL TRUST',
                'DEUTSCHE BANK NATN L TRUST COMPANY',
                'DEUTSCHE BANK NATNL TRUST COMPANY',
                'DEUTSCHE BANK TRUST COMPANY AMERICAS',
                'DEUTSCHE BANK TRUST COMPANY TRUST',
                'DEUTSCHE BANK TRUST COMPANY',
                'DEUTSCHE BANK TRUST NATIONAL',
                "DEUTSCHE BANK TRUST",
                "DEUTSCHE BANK NTL TRST",
                "DEUTSCHE BANK TRUST AM",
                "DEUTSCHE BK NATL TR CO",
                "DEUTSCHE BK NATL TRUST",
                'DEUTSCHE BNK NATIONAL TRUST COMPANY',
                'DEUTCHE BANK NATIONAL TRUST COMPANY TRUST',
                'DEUTCHE BANK NATIONAL TRUST COMPANY',
                'DEUTCHE BANK TRUST COMPANY AMERICAS',
                'DEUTCH BANK NATIONAL TRUST COMPANY'
            ],
            'DEUSTCHE BANK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                "DEVON BANK AN ILLINOIS",
                "DEVON BANK LOAN DEPT"
            ],
            "DEVON BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "DEVON BANK LAND TRUST",
                "DEVON BANK NA TRUSTEE",
                "DEVON BANK TR",
                "DEVON BANK TRUST DEPT",
                "DEVON BANK TRUST",
                "DEVON BK TRUST"
                "DEVON BK TR",
            ],
            "DEVON BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "FIFTH THIRD BANK NATIO",
                "FIFTH THIRD BANK NA NO",
                "FIFTH THIRD BANK NA",
            ],
            "FIFTH THIRD BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST MIDWEST BANK AS",
                "FIRST MIDWEST BANK ITS",
                "FIRST MIDWEST BANK K W",
                "FIRST MIDWEST BANK LAN",
                "FIRST MIDWEST BANK OF",
                "FIRST MIDWEST BK TRUST"
            ],
            "FIRST MIDWEST BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST MIDWEST BANK & T",
                "FIRST MIDWEST BANK LAN",
                "FIRST MIDWEST BANK TRS",
                "FIRST MIDWEST BANK TRU",
                "FIRST MIDWEST BANK U T",
                "FIRST MIDWEST BK TRUST",
            ],
            "FIRST MIDWEST BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "1 NATIONAL BK",
                "1ST NATL BK IL",
                "FIRST NAT BK OF MRT GR",
                "FIRST NATIONAL BK IL",
                "FIRST NATL BK IL",
            ],
            "FIRST NATIONAL BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "1 NATIONA BK T",
                "1ST NATL BK IL TR",
                "FIRST NATL BK TRUST"
            ],
            "FIRST NATIONAL BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST AMERICAN BANK 11",
                "FIRST AMERICAN BANK AS",
                "FIRST AMERICAN BANK LA",
            ],
            "FIRST AMERICAN BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST AMERICAN BANK TR"
            ],
            "FIRST AMERICAN BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST NATIONS BANK AS"
            ],
            "FIRST NATIONS BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST NATIONS BANK A T",
            ],
            "FIRST NATIONS BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                'HSBC MORTGAGE CORPORATION',
                'HSBC MORTGAGE SERVICES INC',
                'HSBC BANK NATIONAL ASSOCIATION',
                'HSBC BANK USA NA',
                'HSBC BANK USA',
                'HSBC BANK USA NATIONAL ASSOCIATION INC',
                'HSBC BANK USA NATIONAL ASSOCIATION',
                "HSBC BANK USA NATIONAL",
                "HSBC BANK NA",
                "HSBC BANK USA NA AS IN",
                "HSBC BANK USA NA",
                "HSBC BANK USA B150",
                "HSBC BANK USA PHH MTG",
                "HSCB BANK USA",
                "HSBC BK USA",
                "HSBC BK"
            ],
            'HSBC BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'HSBC BANK USA NA AS TRUSTEE',
                'HSBC BANK USA NATIONAL ASSOCIATION TRUST',
                'HSBC BANK USA TRUST',
                "HSBC BANK USA NA ATRUS",
            ],
            'HSBC BANK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                "ITASCA BANK & TRUSTAS",
                "ITASCA BANK & TRUSTTRU",
                "ITASCA BANK & TRUST AS",
                "ITASCA BANK & TRUST CO",
                "ITASCA BANK & TRUST TR",
                "ITASCA BANK TR"
            ],
            "ITASCA BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                'JP MORGAN CHASE BANK',
                'JPMORGAN CHASE BANK NA',
                'JPMORGAN CHASE BANK',
                "JPMORGAN CHASE BK"
                "JPM CHASE BANK",
            ],
            'JP MORGAN CHASE'
        ))
        banks.update(dict.fromkeys(
            [
                'JP MORGAN CHASE BANK TRUST',
            ],
            'JP MORGAN CHASE TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                "LA SALLE NATIONAL BANK",
                'LASALLE BANK NATIONAL ASSOCIATION',
                'LASALLE BANK NATIONAL',
                'LASALLE BANK NA',
                "LASALLE BANK",
                "LASALLE BK"
            ],
            "LA SALLE BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "LA SALLE NATIONAL BANK",
                'LASALLE BANK NATIONAL ASSOCIATION TRUST',
                "LASALLE BANK TRUST"
            ],
            "LA SALLE BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "MARQUETTE BANK AN ILLI",
                "MARQUETTE BANK F K A M",
                "MARQUETTE BANK FINANCE",
                "MARQUETTE BANK IN IT C",
                "MARQUETTE NAT BK",
                "MARQUETTE BK"
            ],
            "MARQUETTE BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "MARQUETTE BANK TRUSTEE",
                "MARQUETTE BANK TRUST N",
                "MARQUETTE BANK TRUST",
                "MARQUETTE BANK TR",
                "MARQUETTE BANK & TRUST",
                "MARQUETTE BANK AS TRUS",
                "MARQUETTE BANK LAND TR",
                "MARQUETTE BANK LAND TT",
                "MARQUETTE BANK LT",
                "MARQUETTE BANK MBTRUST",
                "MARQUETTE BANKTR",
                "MARQUETTE BK LT",
                "MARQUETTE BK TRUST",
                "MARQUETTE BK TR",
                "MARQUETE BK TRUST",
                "MARQUETE BK TRST",
            ],
            "MARQUETTE BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "MB LAND TRUST",
                "MB TRUST",
                "MB BANK ACCOUNTING",
                "MB FINANCIAL BANK NA N",
                "MB FINANCIAL"
            ],
            "MB FINANCIAL BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "MIDLAND STATES BANK A"
            ],
            "MIDLAND STATES BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "FIRST MID W BANK TRUST",
                "FIRST MID W TRUST"
                "FIRST MID W B AND T TRUST"
            ],
            "FIRST MIDWEST BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "OLD NATIONAL BANK AS G",
                "OLD NATIONAL BANK ITS",
                "OLD NATIONAL BANK"
            ],
            "OLD NATIONAL BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "OXFORD BANK & TRUST AS",
                "OXFORD BANK & TRUST TR",
                "OXFORD BANK & TRUST TT",
                "OXFORD BANK & TRUST",
                "OXFORD BANK AS TRUSTEE"
            ],
            "OXFORD BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "PAN AMERICAN BANK & TR",
                "PAN AMERICAN BANK TRUS"
            ],
            "PANAMERICAN BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "PARKWAY BANK P",
                "PARKWAY BK FACILITIES",
                "PARKWAY BK UT",
            ],
            "PARKWAY BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "PAKRWAY BANK TRUST",
                # "PARKWAY BANK & TRUST T",
                # "PARKWAY BANK & TRUST N",
                # "PARKWAY BANK & TRUST A",
                # "PARKWAY BANK & TRUST C",
                # "PARKWAY BANK & TRUST O",
                # "PARKWAY BANK & TRUST U",
                "PARKWAY BANK & TRUST",
                "PARKWAY BANK AND TRUST",
                "PARKWAY BANK AS TRUSTE",
                "PARKWAY BANK LAND TRUS",
                "PARKWAY BANK & TRUST O",
                "PARKWAY BANK TRUST NO",
                "PARKWAY BANK TRUSTA",
                "PARKWAY BANK TRUST",
                "PARKWAY BANK TR",
                "PARKWAY BK & TR",
                "PARKWAY BK TR CO",
                "PARKWAY BK TR",
                "PARKWAYBK TR",
                "PARKWAY B AND T COMPANY TRUST",
                "PARKWAY B AND T TRUST",
            ],
            "PARKWAY BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                'PNC BANK NATIONAL ASSOCIATION',
                "PNC BANK NATIONAL",
                'PNC BANK NA'
            ],
            'PNC BANK'
        ))
        banks.update(dict.fromkeys(
            [
                "REPUBLIC BANK OF CHICA"
            ],
            "REPUBLIC BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "SEAWAY BANK & TRUST F",
                "SEAWAY BANK & TRUST",
                "SEAWAY BANK AND TRUST"
            ],
            "SEAWAY BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "SOUTHCENTRALBANK"
            ],
            "SOUTH CENTRAL BANK"
        ))
        banks.update(dict.fromkeys(
            [
                "STAMDARD BANK & TRUST",
                "STANDARD BANK & TRUST",
                "STANDARD BANK TR"
            ],
            "STANDARD BANK TRUST"
        ))
        banks.update(dict.fromkeys(
            [
                "TCF NATIONAL BANK"
            ],
            "TCF BANK"
        ))
        banks.update(dict.fromkeys(
            [
                'US BANK NATIONAL ASSOCIATION',
                'US BANK NATIONAL ASSCO',
                'US BANK NATIONAL',
                'US BANK NA',
                "US BK",
                "US BANK TAX DEPT",
                "US BANK CORP RE",
                "US BANK FACILITY MGMT",
                'US BANK ASSOCIATION'
            ],
            'US BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'US BANK NATIONAL ASSOCIATION TRUST',
                'US BANK NATIONAL ASSOCIATION T',
                'US BANK NA TRUSTEE',
                "US BK TRS NA TTEE",
                "US BK TR NA TTEE",
                "US BK TR",
                'US BANK TRUST NA',
                'US BANK TRUST',
                "US BK NATL ASSN TR",
                "US BK TRUSTEE",
                'US BANK AND TRUST'
            ],
            'US BANK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                'WELLS FARGO BANK',
                'WELLS FARGO BANK NA',
                'WELLS FARGO BANK NA F/B/O',
                'WELLS FARGO BANK NATIONAL ASSOCIATION',
                'WELLS FARGO BANK NATIONAL',
                "WELL FARGO BANK NA",
                "WELLS FARGO BANK NA AS",
                "WELLS FARGO BANK NATIO",
                "WELLS FARGO BK",
                "WELLS FARGO HOME MORTG",
                "WELLS FARGO MORTGAGES",
                "WELLS FARGO BK"
            ],
            'WELLS FARGO BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'WELLS FARGO BANK NA TRUST',
                'WELLS FARGO BANK TRUSTEES',
                "WELLS FARGO BANK NA TR"
            ],
            'WELLS FARGO BANK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                'FEDERAL NATIONAL MORTGAGE ASSOCIATION',
                'FEDERAL NATIONAL MORTGAGE',
                'FEDERAL MATIONAL MORTGAGE',
                'FEDERAL MORTGAGE ASSOCIATION',
                'FEEDERAL HOME LOAN MORTGAGE',
                'FANNIE MAE FNMA',
                'FEDERAL NATIONAL MTG A',
                'FEDERAL NATIONAL MTG W',
                'FEDERAL NATIONAL MTG',
                'FEDERAL NATIONAL MORTG',
                'FEDERAL NATIONAL MORT',
                'FEDERAL NATL MTG ASSOC',
                'FEDERAL NATL MTG ASSN',
                'FEDERAL NATL MTG'
            ],
            'FANNIE MAE'
        ))
        banks.update(dict.fromkeys(
            [
                'FEDERAL HOME LOAN MORTGAGE',
                'FEEDERAL HOME LOAN MORTGAGE',
                'FEDERAL HOME MORTGAGE''FEDERAL HOME LOAN',
                'FEDERAL HOME LOAN MANAGEMENT CORPORATION',
                'FEDERAL HOME LOAN MG CORPORATION',
                'FEDERAL HOME LOAN MORT CORPORATION',
                'FEDERAL HOME LOAN MORTGAGE',
                'FEDERAL HOME LOAN MORTGAGE ASSOCIATION',
                'FEDERAL HOME LOAN MORTGAGE COMPANY',
                'FEDERAL HOME LOAN MORTGAGE CORPORATION',
                'FEDERAL HOME LOAN MTGE CORPORATION',
                'FEEDERAL HOME LOAN MORTGAGE',
                'FEDERAL HOME LOAN MTG',
                'FEDERAL HOME LOAN MTGR',
                'FEDERAL HOMELOAN MTG A'
            ],
            'FREDDIE MAC'
        ))
        banks.update(dict.fromkeys(
            [
                'BANK OF NEW YORK MELLON CORPORATION',
                'BANK OF NEW YORK MELLON',
                'BANK OF NEW YORK'
            ],
            'BANK OF NEW YORK'
        ))
        banks.update(dict.fromkeys(
            [
                'BANK OF NEW YORK TRUSTEES'
                'BANK OF NEW YORK TRUST COMPANY NA',
                'BANK OF NEW YORK TRUST COMPANY',
                'BANK OF NEW YORK TRUST',
                'BANK OF NEW YORK AS TRUSTEE'
            ],
            'BANK OF NEW YORK TRUST'
        ))
        banks.update(dict.fromkeys(
            [
                'BANK UNITED',
                'BANKUNITED'
            ],
            "BANK UNITED"
        ))
        banks.update(dict.fromkeys(
            [
                'COUNTRYWIDE BANK FSB',
                'COUNTRYWIDE BANK'
            ],
            'COUNTRYWIDE BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'FLAGSTAR BANK FSB',
                'FLAGSTAR BANK'
            ],
            'FLAGSTAR BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'INDYMAC BANK FSB',
                'INDYMAC FEDERAL BANK FSB'
            ],
            'INDYMAC BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'M AND T BANK',
                'MANDT BANK SBM',
                'MANDT BANK'
            ],
            'M AND T BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'SANTANDER BANK NA',
                'SANTANDER BANK'
            ],
            'SANTANDER BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'SHAWMUT BANK NATIONAL ASSOCIATION',
                'SHAWMUT BANK OF BOS NA TRUST'
            ],
            'SHAWMUT BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'SOVEREIGN BANK NA',
                'SOVEREIGN BANK'
            ],
            'SOVEREIGN BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'TD BANK N NA',
                'TD BANK NA'
            ],
            'TD BANK'
        ))
        banks.update(dict.fromkeys(
            [
                'WACHOVIA BANK NA TRUST',
                'WACHOVIA BANK NA'
            ],
            'WACHOVIA BANK'
        ))

        # Chicago-specific
        banks.update(dict.fromkeys(
            [
                "THE CHICAGO TRUSTNA",
                "THE CHICAGO TRUST NA A",
                "THE CHICAGO TRUST NA I",
                "THE CHICAGO TRUST NA T",
                "THE CHICAGO TRUST  NA",
                "THE CHICAGO TRUST CO",
                "THE CHICAGO TRUST TR D",
                "THE CHICAGO TRUST TR",
                "CHICAGO TRUSTNATRUSTEE",
                "CHICAGO TRUST ADMIN",
                "CHICAGO TRUST COMAPNY",
                "CHICAGO TRUST COMPANY",
                "CHICAGO TRUST CO",
                "CHICAGO TRUST TRUST",
                "CHICAGO TRUST KNOWN AS",
                "CHICAGO TRUST AS S",
                "CHICAGO TRUST",
                "CHICAGO TITLE LAND TRUST COMPANY TRUST",
                "CHICAGO TITLE LAND TRUST COMPANY COMPANY"
                "CHICAGO TITLE LAND TRUST COMPANY SUCCSR TTEE",
                "CHICAGO TITLE LAND TRUST COMPANY LAND TRUST",
                "CHICAGO TITLE LAND TRUST COMPANY AND T",
                "CHICAGO TITILE LAND TRUST",
                "CHICAGO TITLE LAND TRUST",
                "CHICAGO TITLE LAND TRUS",
                "CHICAGO TITLE LAND TRU",
                "CHICAGO TITLE LAND TRT",
                "CHICAGO TITLE LAND TRS",
                "CHICAGO TITLE LAND TR",
                "CHICAGO TITLE LAND AS",
                "CHICAGO TITLE LAND CO",
                "CHICAGO TITLE LAND",
                "CHICAGO TITLE LND TRST",
                "CHIICAGO TITLE & LAND",
                "CHICAGO TITLE & TRUSTE",
                "CHICAGO TITLE & TRUST",
                "CHICAGO TITLE & LAND T",
                "CHICAGO TITLE AS TRUST",
                "CHICAGO TITLE TRUST AS",
                "CHICAGO TITLE  TRUST",
                "CHICAGO TITLE TRUST AG",
                "CHICAGO TITLE TRUSTEE",
                "CHICAGO TITLE TRUST",
                "CHICAGO TITLE TR",
                "CHGO TITLE LAND TRUST",
                "CHGO TITLE LAND TR",
                "CHICAGO TITLE  LAND TR",
                "CHICAGO TITLE LAN TRUS",
                "CHICAGO TITLE AND",
                "CHICAGO TITLE LSND TRU",
                "CHICAGO TITLE",
                "C T L T COMPANY",
                "CHICAGOTRST",
                "CHICAGO LAND TRUST COMPANY",
                "CHICAGO LAND TRUST CO",
                "CHICAGO LAND TRUST TRUST",
                "CHICAGO LAND TRUST TRU",
                "CHICAGO LAND TRUST AS",
                "CHICAGO LANDTRUST",
                "CHICAGO LAND TRUST",
                "CT LAND TRUST",
                "CT LND TRUST",
                "CT T TRUST",
                "CT TRUST",
                "CTANTT TRUST",
                "CTCTL",
                "CTLTC SUCCSR TTEE",
                "CTLTC TRUST NO",
                "CTLTC TRUST NUMBER",
                "CTLTC TRUST",
                "CTLTC TRST",
                "CTLTC TR",
                "CTLTC NO",
                "CTLTC  NO",
                "CTLTC",
                "CTLT TRUST",
                "CTLT AS TRUSTEE",
                "CTLT NO",
                "CTLT COMPANY",
                "CTLT CO",
                "CTLT",
                "CTLCT TRUST",
                "CTL TR",
                "CTT LAND TRUST",
                "CTT TRUSTEE",
                "CTT TRUST"
                "CTTRUST COMPANY",
            ],
            "CHICAGO TITLE LAND TRUST COMPANY"
        ))
        banks.update(dict.fromkeys(
            [
                "ATG LAND TRUST COMPANY",
                "ATG LAND TRUST CO",
                # "ATG TRUST AS TRUSTEE U",
                "ATG TRUST  AS TRUSTEE",
                "ATG TRUST AS TRUSTEE",
                "ATG TRUST A TRUST AGRE",
                "ATG LAND TRUST",
                "ATG TRUST ITS SUCCESSO",
                "ATG TRUSTUTA",
                "ATG TRUST TRUSTEE TRUST",
                "ATG TRUST TRUST",
                "ATG TRUST TR",
                "ATG TRUST COMPANY",
                "ATG TRUST CO",
                "ATG TRUST AN ILLINOISR",
                "ATG TRUST AN",
                "ATG TRUST AS",
                "ATG TRUST A",
                "ATG TRUST",
            ],
            "ATG TRUST COMPANY"
        ))
        banks.update(dict.fromkeys(
            [
                "REAL ESTATE TAXPAYER",
                "TAXPAYER OF",
                "TAX PAYER OF",
                "CURRENT OWNER"
            ],
            "UNKNOWN"
        ))
        banks.update(dict.fromkeys(
            [
                "COM ED TAX DEPT",
                "COMED TAX DEPARTMENT",
                "COMMONWEALTH EDISON CO"
            ],
            "COMED"
        ))
        banks.update(dict.fromkeys(
            [
                "COUNTY OF COOK D B A C",
                "COUNTY OF COOK DBA C",
                "COUNTY OF COOK DBAC"
            ],
            "COOK COUNTY"
        ))

        return banks


    #----------------------------
    #----ROW-LEVEL OPERATIONS----
    #----------------------------

    @classmethod
    def fix_banks(cls, text, banks):

        # Obtain a list of keys from the "banks" dictionary
        bank_keys = list(banks.keys())

        # Iterate through each key and check if it's found within the "text" string
        for key in bank_keys:
            if key in text:
                # If found, replace that text with banks[key], adding a space at the end
                text = text.replace(key, banks[key] + " ")
                break

        # Split the text by spaces, strip each item of all whitespace, and return the joined list with equal spacing between everything
        fixed_text = " ".join(item.strip() for item in text.split())

        return fixed_text

    @classmethod
    def set_is_bank(cls, clean_name, banks):
        bank_values = list(set(banks.values()))
        if pd.notnull(clean_name):
            for key in bank_values:
                if key in clean_name:
                    return True
        return False

    @classmethod
    def set_is_trust(cls, clean_name):
        if pd.notnull(clean_name):
            name_split = clean_name.split()
            for name in name_split:
                if name in cls.TRUSTS:
                    return True
            for t in cls.TRUSTS_STRINGS:
                if t in clean_name:
                    return True
        return False

    @classmethod
    def convert_corp_abbreviations(cls, text, corp_abb):
        try:
            return corp_abb[text]
        except:
            return text

    @classmethod
    def convert_abbreviations_spaces(cls, text, other_abb):
        try:
            text = text + ' '
            for key in other_abb.keys():
                text = re.sub(r'{}'.format(key), str(other_abb[key]), text)
            if text.strip()[-5:] == "NORTH":
                text = text.strip()[:-5] + ' N'
            if text.strip()[-5:] == "SOUTH":
                text = text.strip()[:-5] + ' S'
            if text.strip()[-4:] == "EAST":
                text = text.strip()[:-4] + ' E'
            if text.strip()[-4:] == "WEST":
                text = text.strip()[:-4] + ' W'
            return text.strip()
        except:
            return text

    @classmethod
    def core_name(cls, text):
        try:
            text = text + ' '
            for key in cls.UNIQUE_KEYS:
                text = re.sub(r'{}'.format(key), '', text)
            return text.strip()
        except:
            return text

    @classmethod
    def identify_num(cls, text):
        try:
            all_words = []
            for item in text.split():
                try:
                    t = int(item)
                    all_words.append(True)
                except:
                    all_words.append(False)
            if True in all_words:
                return True
            else:
                return False
        except:
            return False

    @classmethod
    def identify_name_pattern(cls, text, names_list):
        try:
            text_list = text.split()

            if len(text_list) > 2 and text_list[-1] in ['JR', 'SR']:
                if all(word in names_list for word in text_list[:-1]):
                    return True

            if all(word in names_list for word in text_list):
                return True

            return False
        except:
            return False

    @classmethod
    def identify_single(cls, text):
        try:
            te = len(text.split())
            if te == 1:
                return True
            else:
                return False
        except:
            return False

    @classmethod
    def identify_person_name(cls, text, names_list):
        try:
            text_list = text.split()
            if 1 < len(text_list):
                for name in text_list:
                    if name in names_list:
                        return True

                return False
            else:
                return False
        except:
            return False


    #-------------------------------
    #----MAIN CLEANING FUNCTIONS----
    #-------------------------------

    @classmethod
    def clean_name(cls, text, banks, corp_abb):

        try:
            # Punc/Spaces/Symbols
            text = cls.fix_banks(text, banks)
            text = cls.delete_symbols_spaces(text)

            # # # Abbreviations
            text_list = text.split()
            text_list = [cls.convert_corp_abbreviations(str(x).upper(), corp_abb) for x in text_list]
            text = ' '.join([str(x) for x in text_list]).upper()

            # text = text.replace('A%A', '')
            text = cls.delete_symbols_spaces(text)
            text = text.replace(' / ', '/')

            # After abbreviations
            text =  re.sub(r'(?<=\b[A-Z]) (?=[A-Z]\b)', '', text)

            # Misc
            # Switch The to the front
            text = cls.switch_the(text)
            text = cls.convert_nesw(text)

            return text.strip()
        except Exception as e:
            return np.nan

    @classmethod
    def clean_address(cls, text, corp_abb, other_abb):

        try:
            # Punc/Spaces/Symbols
            text = text.strip()
            text = cls.convert_st(text)
            text = cls.delete_symbols_spaces(text)

            # Numbers
            text_list = cls.combine_numbers([str(cls.words_to_num(x)) for x in text.split()])
            text_list = [cls.convert_mixed(x) for x in text_list]
            text_list = [cls.words_to_num(x) for x in ' '.join(text_list).split()]
            text_list = cls.combine_numbers([cls.words_to_num(x) for x in text_list])
            text_list = [cls.convert_ordinals(x) for x in text_list]

            # Abbreviations
            text_list = [cls.convert_corp_abbreviations(x, corp_abb) for x in text_list]
            text = ' '.join([str(x) for x in text_list]).upper()
            text = text.replace(' / ', '/')

            # After abbreviations
            text =  re.sub(r'(?<=\b[A-Z]) (?=[A-Z]\b)', '', text)

            # Misc
            # Switch The to the front
            text = cls.switch_the(text)
            text = cls.convert_abbreviations_spaces(text, other_abb)
            text = cls.convert_nesw(text)
            text = cls.dedup_words(text)
            text = cls.take_first(text)
            text = text.replace('-','')
            text = cls.drop_floors(text)
            text = cls.drop_letters(text)
            text = cls.drop_floors(text)


            return text
        except Exception as e:
            print(e)
            return np.nan

    # @classmethod
    # def process_text_parallel_addresses(cls, df):
    #     df['MAIL_ADD'] = df['MAIL_ADDRESS'].apply(lambda x: cls.clean_address(x))
    #     df['MAIL_ADD_CS'] = df['MAIL_CITY_STATE'].apply(lambda x: cls.clean_address(x))
    #     df['MAIL_ADD_CS'] = df['MAIL_ADD_CS'].apply(lambda x : cls.change_NESW(x))
    #     df['MAIL_ZIPCODE'] = df['MAIL_ZIP'].apply(lambda x : cls.convert_zip(x))
    #     df['CleanAddress'] = df['MAIL_ADD'].apply(str) + ' ' + df['MAIL_ADD_CS'].apply(str)
    #     return df

    @classmethod
    def process_text_parallel_names(cls, df):
        banks = cls.get_banks()
        corp_abb = cls.get_corp_abb()
        df["CLEAN_NAME"] = df["TAXPAYER_NAME"].apply(lambda x: cls.clean_name(x, banks, corp_abb))
        df["CORE_NAME"] = df["CLEAN_NAME"].apply(lambda x: cls.core_name(x))
        return df

    @classmethod
    def process_text_parallel_banks_trusts(cls, df):
        banks = cls.get_banks()
        df["IS_BANK"] = df["CLEAN_NAME"].apply(lambda x: cls.set_is_bank(x, banks))
        df["IS_TRUST"] = df["CLEAN_NAME"].apply(lambda x: cls.set_is_trust(x))
        df["TRUST_ID"] = df.apply(lambda row: cls.set_trust_id(row), axis=1)
        df["TRUST_INSTITUTION"] = df["CLEAN_NAME"].apply(lambda x: cls.set_trust_institution(x))
        return df

    @classmethod
    def set_trust_institution(cls, clean_name):
        if pd.notnull(clean_name):
            for trust in cls.TRUST_COMPANIES_IDS:
                if trust in clean_name:
                    return trust
        return np.nan

    @classmethod
    def set_trust_id(cls, row):
        if row["IS_TRUST"] == True:
            for trust in cls.TRUST_COMPANIES_IDS:
                if trust in row["CLEAN_NAME"]:
                    id = row["CLEAN_NAME"].replace(trust, "").replace(" ", "").strip()
                    id = id.replace("PROPERTIES", "").strip()
                    if id != "" and len(id) > 3:
                        return id
        return np.nan


class AddressValidator(BaseLandlordData):

    PO_BOXES = [
        "POSTOFFICEBOX", "POSTBOX", "POBBOX", "PO BOX", "POBOC", "POBOX", "PBOX", "POST", "POBX", "BOX", "POB", "PO"
    ]
    PO_BOXES_DEPT = ["TAXDEPT", "TAXDEP", "TXDEPT", "DEPT"]
    PO_BOXES_REMOVE = [
        "LOUISAVE", "SSWANTST", "CHICAGO", "6890S2300E", "REDEPT", "RAVSTN", "RAVNIASTA", "RAVINIASTA", "RAVINIAST",
        "RAVINIA", "REVINIAST"
    ]

    def __init__(self):
        super().__init__()

    @classmethod
    def check_street_num_equality(cls, validated_addr, raw_addr):

        if validated_addr.startswith("PO BOX"):
            validated_stripped = validated_addr[7:]
            validated_addr = validated_stripped.split(",")[0]
            validated_num = validated_addr.split()[0]
        else:
            validated_street_addr = validated_addr.split(",")
            validated_num = validated_street_addr[0].split()[0]

        raw_addr_cleaned = CleanTaxpayer.delete_symbols_spaces(raw_addr)

        for po in cls.PO_BOXES:
            if raw_addr_cleaned.startswith(po):
                raw_addr = raw_addr_cleaned[len(po):].strip()
                break

        # Extract the street number from the cleaned raw address
        raw_num = raw_addr.split()[0] if raw_addr else ""

        # Compare the street numbers directly
        return raw_num == validated_num

        # return raw_addr[:len(validated_num)] == validated_num

    @classmethod
    def check_zip_equality(cls, validated_addr, raw_addr):
        if raw_addr.endswith("00000"):
            return True
        return validated_addr[-5:] == raw_addr.strip()[-5:]

    @classmethod
    def check_city_equality(cls, validated_addr, raw_addr):
        validated_city = validated_addr.split(",")[-2].strip()
        raw_city = raw_addr.split(",")[-2].strip()
        return validated_city == raw_city

    @classmethod
    def set_is_pobox(cls, raw_addr):
        raw_addr_cleaned = raw_addr.replace(" ", "").strip()
        for po in cls.PO_BOXES:
            if raw_addr_cleaned.startswith(po):
                return True
        return False

    @classmethod
    def fix_pobox(cls, raw_addr):

        raw_addr_split = raw_addr.split(",")

        if "#" in raw_addr_split[0]:
            raw_addr_no_spaces = raw_addr_split[0].replace(" ", "")
            match = re.search(r"([a-zA-Z])#", raw_addr_no_spaces)
            if match:
                raw_addr_cleaned = raw_addr.replace("#", "").strip().replace(" ", "")
            else:
                raw_addr_cleaned = raw_addr.replace("#", "DEPT ").strip().replace(" ", "")
        else:
            raw_addr_cleaned = raw_addr.replace(" ", "").strip()

        for po in cls.PO_BOXES:

            if raw_addr_cleaned.startswith(po):

                raw_addr_stripped = raw_addr_cleaned[len(po):].strip()
                pobox_num = raw_addr_stripped.split(",")[0].strip()

                for remove in cls.PO_BOXES_REMOVE:
                    if remove in pobox_num:
                        pobox_num = pobox_num.replace(remove, "")

                dep = ""
                dep_start = len(pobox_num)
                for dep_string in cls.PO_BOXES_DEPT:
                    if dep_string in pobox_num:
                        dep_start = pobox_num.find(dep_string)
                        dep_stripped = pobox_num[dep_start + len(dep_string):].strip()
                        digits = ''.join(filter(str.isdigit, dep_stripped))
                        dep = f"TAX DEPT {digits}"  # Insert space before department string
                        break

                # Add a space between the PO box number and department part
                raw_addr_split[0] = f"PO BOX {pobox_num[:dep_start].strip()} {dep}".strip()
                raw_addr_fixed = ",".join(raw_addr_split)
                return raw_addr_fixed

    @classmethod
    def create_validated_pobox_row(cls, df_subset, raw_addr, raw_addr_cleaned):
        raw_addr_validated = (f"{raw_addr_cleaned.split(',')[0].strip()}, "
                              f"{df_subset['GCD_CITY'].iloc[0]}, "
                              f"{df_subset['GCD_STATE'].iloc[0]} {df_subset['GCD_ZIP'].iloc[0]}")

        validated_row = df_subset.iloc[0].to_dict()
        validated_row["GCD_NUMBER"] = raw_addr_cleaned.split(',')[0].strip().split()[2]
        validated_row["GCD_STREET"] = "PO BOX"
        validated_row["GCD_SUFFIX"] = np.nan
        validated_row["GCD_PREDIRECTIONAL"] = np.nan
        validated_row["RAW_ADDRESS"] = raw_addr
        validated_row["GCD_FORMATTED_ADDRESS"] = raw_addr_validated.replace("  ", " ").strip()

        # Handle secondary unit and number
        if "TAX DEPT" in raw_addr_validated:
            validated_row["GCD_SECONDARYUNIT"] = "TAX DEPT"
            validated_row["GCD_SECONDARYNUMBER"] = raw_addr_cleaned.split(',')[0].strip().split()[-1]

        return validated_row


    @classmethod
    def validate_pobox(cls, raw_addr, df_pobox):

        gcd_cities = list(df_pobox["GCD_CITY"].dropna().unique())
        raw_addr_cleaned = df_pobox["RAW_ADDRESS_CLEANED"].iloc[0]

        if len(gcd_cities) == 1:
            return cls.create_validated_pobox_row(df_pobox, raw_addr, raw_addr_cleaned)

        elif df_pobox["TAXPAYER_CITY"].iloc[0] in gcd_cities:
            df_subset = df_pobox[df_pobox["GCD_CITY"] == df_pobox["TAXPAYER_CITY"].iloc[0]]
            return cls.create_validated_pobox_row(df_subset, raw_addr, raw_addr_cleaned)

        else:
            return None


    @classmethod
    def run_pobox_validator(cls, df_unvalidated_poboxes):

        validated_rows = []

        for raw_addr in list(df_unvalidated_poboxes["RAW_ADDRESS"].unique()):

            df_pobox = df_unvalidated_poboxes[df_unvalidated_poboxes["RAW_ADDRESS"] == raw_addr]
            if len(df_pobox) == 0:
                print(raw_addr)
            else:
                validated_row = cls.validate_pobox(raw_addr, df_pobox)
                if validated_row is not None:
                    validated_rows.append(validated_row)

        df_pobox_validated = pd.DataFrame(validated_rows)

        return df_pobox_validated

    @classmethod
    def get_validated_string_match_row(cls, raw_addr, gcd_validated, df_addr):
        df_addr_filtered = df_addr[df_addr["GCD_FORMATTED_ADDRESS"] == gcd_validated]
        if len(df_addr_filtered) > 0:
            validated_row = df_addr_filtered.iloc[0].to_dict()
            validated_row["RAW_ADDRESS"] = raw_addr
            return validated_row
        else:
            return None


    @classmethod
    def validate_string_match(cls, raw_addr, df_addr_match, df_addr):

        # df_addr: all rows from original unvalidated address dataset where raw_address equals raw_addr
        # df_addr_match: all rows from string match results where original_doc equals raw_addr

        df_match_filtered = df_addr_match[df_addr_match["MATCH_NUM"] == True]

        if raw_addr[-5:] != "00000":
            df_match_filtered = df_match_filtered[df_match_filtered["MATCH_ZIP"] == True]

        if len(df_match_filtered) == 1:
            gcd_validated = df_match_filtered["MATCHED_DOC"].iloc[0]
            validated_row = cls.get_validated_string_match_row(raw_addr, gcd_validated, df_addr)
            return validated_row
        elif len(df_match_filtered) > 1:
            df_match_ordered = df_match_filtered.sort_values(by="CONF1", ascending=False)
            gcd_validated = df_match_ordered["MATCHED_DOC"].iloc[0]
            validated_row = cls.get_validated_string_match_row(raw_addr, gcd_validated, df_addr)
            return validated_row
        else:
            return None

    @classmethod
    def run_string_match_validator(cls, df_unvalidated_addrs):

        # set string matching params
        ref_docs = list(df_unvalidated_addrs["GCD_FORMATTED_ADDRESS"].dropna().unique())
        query_docs = list(df_unvalidated_addrs["RAW_ADDRESS"].dropna().unique())
        nmslib_opts = {
            "method": "hnsw",
            "space": "cosinesimil_sparse_fast",
            "data_type": nmslib.DataType.SPARSE_VECTOR
        }
        query_batch_opts = {
            "num_threads": 8,
            "K": 3
        }
        match_threshold = .75

        # run matching algorithm
        df_matches = StringMatching.match_strings(ref_docs, query_docs, nmslib_opts, query_batch_opts, match_threshold)

        df_matches["MATCH_NUM"] = df_matches[["MATCHED_DOC", "ORIGINAL_DOC"]].apply(lambda x: cls.check_street_num_equality(x[0], x[1]), axis = 1)
        df_matches["MATCH_ZIP"] = df_matches[["MATCHED_DOC", "ORIGINAL_DOC"]].apply(lambda x: cls.check_zip_equality(x[0], x[1]), axis = 1)

        validated_rows = []

        for raw_addr in list(df_matches["ORIGINAL_DOC"].dropna().unique()):

            df_addr = df_unvalidated_addrs[df_unvalidated_addrs["RAW_ADDRESS"] == raw_addr]
            df_addr_matched = df_matches[df_matches["ORIGINAL_DOC"] == raw_addr]

            validated_row = cls.validate_string_match(raw_addr, df_addr_matched, df_addr)

            if validated_row is not None:
                validated_rows.append(validated_row)

        df_string_match_validated = pd.DataFrame(validated_rows)

        return df_string_match_validated





class CleanCorpLLC(CleanTaxpayer):

    CORP_WORDS = [
        'LLC', 'PROPERTIES', 'CHEMICAL', 'PC', 'MD', 'SALON', 'GOOD', 'INSTITUTE', 'HOSPITAL', 'WELLESLEY', 'PROGRAM', 'SHORE', 'TRUSTEES', 'APPLIED',
        'COASTAL', 'WORLD', 'VENTURES', 'PLYMOUTH', 'HARVARD', 'END', 'BUILDING', 'DELIVERY', 'TOWN', 'YOUTH',
        'FOODS', 'BLUE', 'GOVERNMENT', 'POST', 'SERVICE', 'EXPORT', 'PACKAGING', 'ISLAND', 'WEALTH', 'ALPHA', '80TH',
        'CATERING', 'COUNSELING', '50TH', 'ADVISORS', 'ESSEX', 'INDUSTRIAL', 'SOCIAL', 'WESTERN', 'CENTERS', 'FENWAY',
        'CHESTNUT', 'HOME', 'LINE', 'UNITED', 'SAFETY', 'BACK', 'MATERIALS', 'WEST', 'SUN', 'SCIENCE', 'HOLDINGS',
        'UNION', '40TH', 'TRANSPORT', 'FLOOR', '90TH', 'BUSINESS', 'ENTERTAINMENT', 'ST', 'FOR', 'STORAGE',
        'SYSTEMS', 'INVESTORS', 'ENGINEERS', '30TH', 'HOTEL', 'FINE', 'CONSULTANTS', 'PATRIOT', 'SPECIALTY', 'HOSPITALITY',
        'TRANSPORTATION', 'SUPPLY', 'TRAINING', 'NEW ENGLAND', 'PUBLIC', 'BIG', 'HIGHWAY', 'AGENCY', 'REAL', 'MERRIMACK',
        'NORTH END', 'HEALTH', 'STATES', 'PARTNERSHIP', 'PLAZA', 'MASSACHUSETTS', 'MUNICIPAL', 'COFFEE', 'COMMONWEALTH',
        'RESTAURANT', 'WORLDWIDE', 'EYE', 'WINE', 'PLUMBING', 'STRATEGIES', 'PIONEER', 'TIME', 'ALL', 'VINEYARD', 'ROCK',
        'TRINITY', 'COMPANY', 'CENTRAL', 'COUNCIL', 'CLEAN', 'PARK', 'CABLE', 'EXCHANGE', 'TECHNOLOGIES', 'EDUCATIONAL',
        'APARTMENTS', 'MATTAPAN', 'SOCCER', 'ELITE', 'GOLF', 'PREMIER', 'ART', 'OCEAN', 'EASTERN', 'STAR', 'FRANKLIN', 'ACADEMY',
        'GREATER', 'SPECIALISTS', 'CLEANING', 'SMART', 'CAB', 'LOGISTICS', 'CONTRACTORS', 'BROKERAGE', 'LANDSCAPE',
        'HIGHLAND', 'BEAUTY', 'DIGITAL', 'ADVISORY', 'NETWORKS', 'SILVER', 'MEDICAL', 'CALIFORNIA', 'SALEM', 'TECHNOLOGY',
        'SALES', 'SYSTEM', 'PASS', 'CONVENIENCE', 'FASHION', 'RESOURCE', 'ESTATE', 'MANUFACTURING', 'CHARITABLE',
        'PERFORMANCE', 'COMMUNITY', 'MAINTENANCE', 'STUDIO', 'BRIDGE', 'LABS', 'POWER', 'PRODUCTS', 'COUNTRY', 'CAR',
        'PAINTING', 'AND', 'VALLEY', 'PLLC', 'COLLABORATIVE', 'OAK', 'GOLD', 'BEACON', 'REVOCABLE', 'SOURCE',
        'BROCKTON', 'PHOTOGRAPHY', 'SOUTH', 'TRI', '3RD', 'WATER', 'LEARNING', 'COM', 'NATIONAL', 'SUPPORT', 'CITY',
        'DRIVE', 'UNLIMITED', 'LIGHT', 'SOLUTIONS', 'VETERANS', 'DRYWALL', 'POND', 'TAX', 'INVESTMENTS',
        'RECOVERY', 'ORGANIZATION', 'CONNECTION', 'EXPRESS', 'DISTRIBUTION', 'ARCHDIOCESE', 'GARAGE', 'THERAPEUTICS',
        'COAST', 'BUILDERS', 'PRECISION', 'RENTALS', 'TELECOMMUNICATIONS', 'KITCHEN', 'RETAIL', 'LABORATORIES',
        'COMMUNICATIONS', 'MINISTRIES', 'RESIDENTIAL', 'INTERACTIVE', 'AIR', 'NORTHEAST', 'DATA', 'THROUGH',
        'FOOD', 'BENEFITS', 'NURSING', 'DORCHESTER', 'SCIENTIFIC', 'PLASTERING', 'PET', 'CARPENTRY', 'MDPC',
        'METAL', 'DRIVEWAY', 'STATE', 'PARTS', 'BOSTON HOUSING AUTHORITY', 'BOULEVARD', 'ASSOCIATES', 'PLEASANT',
        'SPRINGFIELD', 'TOURS', 'MOTORS', 'PACIFIC', 'FUND', 'AVIATION', 'STRATEGIC', 'PHYSICAL', 'INNOVATIVE',
        'ENGINEERING', 'NANTUCKET', 'LLC', 'CORNER', 'ATLANTIC', 'CHARLESTOWN', 'CENTER', 'SEAFOOD', 'ELECTRIC', 'GRILL',
        'ENTERPRISE', 'WALTHAM', 'ENGLAND', 'QUALITY', 'NEWTON', 'CORPORATE', 'PLUS', 'IMPORTS', 'INFORMATION', 'CLASSIC',
        'EAGLE', 'NET', 'TITLE', 'CREDIT', 'RESOURCES', 'SCHOLARSHIP', 'HILL', 'GRANITE', 'FARMS', 'REAL ESTATE',
        'FAMILY', 'FITNESS', 'FLOORING', 'COMPANIES', 'EQUIPMENT', 'CONCORD', 'VENTURE', 'GENERAL', 'ROYAL', 'COLLEGE',
        'DONUTS', 'MEMORIAL', 'SECURITY', 'MIDDLESEX', 'REPAIR', 'GREAT', 'NEWBURY', '70TH', 'NORTH', 'FUNDS', 'INCOME',
        'THERAPY', 'PRESS', 'NATURAL', 'TERRACE', 'YOUR', 'OIL', 'CHURCH', 'AUTHORITY', 'PRODUCTIONS', 'CROSSWAY',
        'CONTINENTAL', 'ADVANCED', 'TECHNICAL', 'NEW', 'TRUCK', 'ARCHITECTS', 'CONCEPTS', 'SERIES', 'LEASING', 'CAFE', 'BAY',
        'HOUSING', '1ST', 'EDGE', 'YORK', 'GLOBAL', 'CONTRACTOR', 'PRINTING', 'FURNITURE', 'HAIR', 'IGLESIA', 'WHOLESALE',
        'AUTO', 'GARDEN', 'CR', 'CIR', 'SON', 'CARE', 'FRIENDS', 'WORCESTER', 'PROJECT', 'WAY', 'FORECLOSURE', 'BAR',
        'MOBILE', 'PUBLISHING', 'PRIME', 'FIRE', 'FUNDING', 'PIZZA', 'MANAGEMENT', 'NORTHERN', 'DENTAL', 'NETWORK',
        'LIBERTY', 'FINANCE', 'STEEL', 'ENVIRONMENTAL', 'REMODELING', 'BOSTON', 'STOP', 'STUDIOS', 'CONDO', 'THEATRE',
        'MECHANICAL', 'TRADING', 'CONDO TRUST', 'UNIVERSAL', 'HIGH', 'BEST', 'INSURANCE', 'MOTOR', 'GOD', 'METRO',
        'COLONIAL', 'CONSTRUCTION', 'GAS', 'PHARMACY', 'CHIROPRACTIC', 'VILLAGE', '100TH', 'PRIVATE', 'INC', 'MOUNTAIN',
        'WOOD', 'MARINE', 'ASSOCIATION', 'SOUTH END', 'EQUITY', 'ACQUISITION', 'CHAPTER', 'PINE', 'IMPROVEMENT', 'BAKERY',
        'BROADWAY', 'MUSIC', 'LENDING', 'INDEPENDENT', 'PARKWAY', 'CHILDREN', 'CORPORATION', 'LIVING', 'BROTHERS',
        'SONS', 'SUB', 'REALTY TRUST', 'TRAVEL', 'INTERNATIONAL', 'RECORDS', 'REDEVELOPMENT', 'AVE', 'PLACE', 'CAPE',
        'DMD', 'WARF', 'ANDOVER', 'INDUSTRIES', 'COMPUTER', 'DIRECT', 'CONTROL', 'COMMERCIAL', 'HALL', 'RESEARCH',
        'COD', 'SHOP', 'PACKAGE', 'EXECUTIVE', 'PARTNERS', 'COMMITTEE', 'JEWELRY', 'LEAGUE', 'TRADE',
        'FISHERIES', 'ATHLETIC', 'CLEANERS', 'HOUSE', 'SOLAR', 'HEATING', 'INN', 'ARTS', 'HOCKEY', 'SUMMIT',
        'DESIGNS', 'TRANS', 'LN', 'SOFTWARE', 'OFFICE', 'HARBOR', 'ENERGY', 'WOODS', 'UNIVERSITY',
        'WORKS', 'CAMBRIDGE', 'FARM', 'MAIN', '60TH', 'INTERIORS', 'TOP', 'SHOE', 'FISHING', 'PAPER',
        'FOUNDATION', 'FALL', 'MANAGERS', '20TH', 'TIRE', 'LIFE', 'HOMES', 'IMAGING', 'ROMAN CATHOLIC',
        '5TH', 'CHOICE', 'TRUCKING', 'ADVERTISING', 'STORES', 'SPORTS', 'STORE', 'DANCE', 'ROSLINDALE',
        'BACK BAY', 'KIDS', 'RESTORATION', 'DAY', 'GROUP', 'GLASS', 'LIABILITY', 'PROPERTIES', 'BEACH',
        'WASHINGTON', 'MARKETING', 'LOWELL', 'TOTAL', 'BODY', 'LAND', 'SOCIETY', 'SECURITIES', 'PLANNING',
        'ROXBURY', 'AMERICAN', 'LIQUORS', 'LANDSCAPING', 'WIRELESS', 'CONSULTING', 'TEAM', 'ICE', 'ACTION',
        'LLP', 'LIMOUSINE', 'CAPITAL', 'GRAPHICS', 'COVE', 'MASONRY', 'GALLERY', 'CARPET', 'GRACE', 'RD',
        'ENTERPRISES', 'DESIGN', 'ALLIANCE', 'ADVANTAGE', 'AMERICA', 'SEA', 'EAST', 'PHOENIX', 'DISTRIBUTORS',
        'MEDIA', 'TOOL', 'TAXI', 'ESTABLISHED', 'DEVELOPMENT', 'BEDFORD', 'CREATIVE', 'LEGAL', 'ELECTRICAL',
        'EXTENSION', 'COURT', 'DELI', 'BROKERS', 'SQUARE', 'FISH', 'PROFESSIONAL', 'FUEL', 'COLONY', 'PROTECTION',
        'LIMITED', 'ACCESS', 'CUSTOM', 'FOREST', 'HERITAGE', 'BURLINGTON', 'VIEW', 'INTEGRATED', 'COOPERATIVE',
        'FRAMINGHAM', 'SPA', 'AUTOMOTIVE', 'SCHOOL', 'OWNER', 'PHARMACEUTICALS', 'OLD', 'COUNTY', 'CONCRETE',
        'REALTY', 'CLUB', 'WOMEN', 'BERKSHIRE', 'GOLDEN', 'FINANCIAL', 'ALLEY', 'TREE', 'PETROLEUM', 'TECH',
        'TELECOM', 'RIVER', 'DOG', 'VALUE', 'OFFICES', 'USA', 'PRESIDENT', 'VIDEO', 'RECYCLING', 'RENTAL',
        'ALLSTON', 'EDUCATION', 'WASTE', 'BRIGHTON', 'SUMMER', 'STAFFING', 'ELECTRONICS', 'ROOFING', 'ASSET',
        'VISION', 'SERVICES', 'CONTRACTING', 'DMDPC', 'MACHINE', 'FREE', 'WELLNESS', 'PRO', 'ESTATES', 'STATION',
        'HEALTHCARE', 'MORTGAGE', 'MARKET', 'TOWING', '2ND', 'TILE', 'PORTFOLIOS', 'ASSETS', 'RESERVES'
    ]

    def __init__(self):
        super().__init__()

    #--------------------------------------------------
    #----ROW-LEVEL OPERATIONS (BOOLEAN IDENTIFIERS)----
    #--------------------------------------------------

    @classmethod
    def identify_corp(cls, text):
        try:
            all_words = []
            for item in text.split():
                if item in cls.CORP_WORDS:
                    all_words.append(True)
                else:
                    all_words.append(False)
            if True in all_words:
                return True
            else:
                return False
        except:
            return False


    @classmethod
    def set_clean_merge(cls, row):
        """
        Sets boolean column on the merged dataframe indicating whether the corp/llc match was made based on
        merging clean names (as opposed to a match based on the raw name, core name merge or fuzzy matching results).
        """
        if pd.notnull(row["ENTITY_NAME"]):
            return True
        return np.nan

    @classmethod
    def set_core_merge(cls, row):
        """
        Sets boolean column on the merged dataframe indicating whether the corp/llc match was made based on
        merging core names (as opposed to a match based on the raw name, clean name merge or fuzzy matching results).
        """
        if pd.notnull(row["ENTITY_NAME"]):
            if pd.isnull(row["CLEAN_MERGE_ORG"]):
                return True
            else:
                return False
        return np.nan

    @classmethod
    def set_fuzzy_merge(cls, row):
        """
        Sets boolean column on the merged dataframe indicating whether the corp/llc match was made based on
        string matching results (as opposed to a match based on the raw name, clean name merge or core name merge).
        """
        if pd.notnull(row["ENTITY_NAME"]):
            if pd.isnull(row["CLEAN_MERGE_ORG"]) and pd.isnull(row["CORE_MERGE_ORG"]):
                return True
            else:
                return False
        return np.nan


    #--------------------------------------------
    #----ROW-LEVEL OPERATIONS (TEXT CLEANING)----
    #--------------------------------------------

    @classmethod
    def remove_inc_llc(cls, text):
        if "INC" in text:
            text = text.replace("INC", "")
        if "LLC" in text:
            text = text.replace("LLC", "")
        return text.strip()

    @classmethod
    def check_core_matches(cls, df_merge_core, common_names):

        """
        Checks the results of the merge based on core name, and filters out invalid matches.
        """

        columns_to_nan = ["ENTITY_NAME", "ENTITY_CLEAN_NAME", "ENTITY_CORE_NAME", "CLEAN_MERGE_ORG", "CORE_MERGE_ORG"]
        indices_to_remove = []

        for i, row in df_merge_core.iterrows():

            if row["CORE_MERGE_ORG"] == True:

                if row["CLEAN_NAME"] in common_names:
                    indices_to_remove.append(i)
                    continue

                clean_taxpayer = cls.remove_inc_llc(row["CLEAN_NAME"]).split()
                clean_entity = cls.remove_inc_llc(row["ENTITY_CLEAN_NAME"]).split()
                for word in clean_taxpayer:
                    if word not in clean_entity:
                        indices_to_remove.append(i)
                        break

                for word in clean_entity:
                    if word not in clean_taxpayer:
                        indices_to_remove.append(i)
                        break

        df_merge_core.loc[indices_to_remove, columns_to_nan] = np.nan

        return df_merge_core


    #-------------------------------
    #----MAIN CLEANING FUNCTIONS----
    #-------------------------------

    @classmethod
    def clean_names_corp(cls, text):

        try:
            # Punc/Spaces/Symbols
            text = cls.delete_symbols_spaces(text)

            # Abbreviations
            text_list = text.split()
            # text_list = [convert_abbreviations(str(x).upper(), corp_abb) for x in text_list]
            text = ' '.join([str(x) for x in text_list]).upper()

            # text = text.replace('A%A', '')
            text = cls.delete_symbols_spaces(text)
            text = text.replace(' / ', '/')

            # After abbreviations
            text =  re.sub(r'(?<=\b[A-Z]) (?=[A-Z]\b)', '', text)

            # Misc
            # Switch The to the front
            text = cls.switch_the(text)
            text = cls.convert_nesw(text)

            return text.strip()
        except Exception as e:
            print(e)
            return np.nan

    @classmethod
    def ident_parallel(cls, all_info, names_list):

        all_info['CORP_WORDS'] = all_info["CLEAN_NAME"].apply(lambda x : cls.identify_corp(x))
        all_info['CORP_NUM'] = all_info["CLEAN_NAME"].apply(lambda x : cls.identify_num(x))
        all_info['PEOPLE_STRUCTURE'] = all_info["CLEAN_NAME"].apply(lambda x : cls.identify_name_pattern(x, names_list))
        all_info['CORP_SINGLE'] = all_info["CLEAN_NAME"].apply(lambda x : cls.identify_single(x))
        all_info['PEOPLE_NAMES'] = all_info["CLEAN_NAME"].apply(lambda x : cls.identify_person_name(x, names_list))

        all_info['CORP'] = np.nan
        all_info.loc[all_info['PEOPLE_STRUCTURE'] == True, 'CORP'] = False
        all_info.loc[all_info['PEOPLE_NAMES'] == True, 'CORP'] = False
        all_info.loc[all_info['CORP_WORDS'] == True, 'CORP'] = True
        all_info.loc[all_info['CORP_NUM'] == True, 'CORP'] = True
        all_info.loc[all_info['CORP_SINGLE'] == True, 'CORP'] = True
        all_info.loc[(all_info['PEOPLE_STRUCTURE'] == True) & all_info['PEOPLE_NAMES'] == True, 'CORP'] = False

        return all_info

    @classmethod
    def corp_data_parallel(cls, df):
        df['ENTITY_CLEAN_NAME'] = df["ENTITY_NAME"].apply(lambda x : cls.clean_names_corp(x))
        df['ENTITY_CORE_NAME'] = df["ENTITY_CLEAN_NAME"].apply(lambda x : cls.core_name(x))
        return df



class StringMatching(BaseLandlordData):

    def __init__(self):
        super().__init__()

    @classmethod
    def ngrams(cls, string, n=3):

        # convert string into ascii encoding
        string = string.encode("ascii", errors="ignore").decode()

        # converts letters in string to lower case
        string = string.lower()

        # removes unwanted characters
        chars_to_remove = [')', '(', '.', '|', '[', ']', '{', '}', "'"]
        rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
        string = re.sub(rx, '', string) # remove the list of chars defined above

        # replace various symbols with non-symbols
        string = string.replace('&', 'and')
        string = string.replace(',', ' ').replace('-', ' ')

        string = string.title() # Capital at start of each word
        string = re.sub(' +',' ',string).strip() # combine whitespace
        string = ' ' + string + ' ' # pad
        #string = re.sub(r'[,-./]', r'', string)

        # core N-gram generation
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]


    @classmethod
    def match_strings(cls, ref_docs, query_docs, nmslib_opts, query_batch_opts, match_threshold=None):

        # set up vectorizer and index for string matching
        vectorizer = TfidfVectorizer(min_df=1, analyzer=cls.ngrams)

        tf_idf_matrix = vectorizer.fit_transform(ref_docs)
        messy_tf_idf_matrix = vectorizer.transform(query_docs)
        data_matrix = tf_idf_matrix

        index = nmslib.init(method=nmslib_opts["method"], space=nmslib_opts["space"], data_type=nmslib_opts["data_type"])
        index.addDataPointBatch(data_matrix)
        index.createIndex()

        # execute query and store good matches
        print("Executing TFIDF matrix query...")
        query_matrix = messy_tf_idf_matrix
        nbrs = index.knnQueryBatch(query_matrix, k=query_batch_opts["K"], num_threads=query_batch_opts["num_threads"])

        mts =[]
        with tqdm(total=len(nbrs), desc="Fetching matches...") as pbar:
            for i in range(len(nbrs)):
                original_nm = query_docs[i]
                for row in list(range(len(nbrs[i][0]))):
                    try:
                        matched_nm = ref_docs[nbrs[i][0][row]]
                        conf = abs(nbrs[i][1][row])
                    except:
                        matched_nm = "no match found"
                        conf = None
                    mts.append([original_nm, matched_nm, conf])
                pbar.update(1)

        print("Generating results dataframe...")
        df_matches = pd.DataFrame(mts,columns=["ORIGINAL_DOC", "MATCHED_DOC", "CONF"])
        df_matches["LDIST"] = df_matches[["MATCHED_DOC", "ORIGINAL_DOC"]].apply(lambda x: lev.distance(x[0], x[1]), axis=1)
        df_matches["CONF1"] = 1- df_matches["CONF"]

        if match_threshold is not None:
            df_good_matches = df_matches[(df_matches["LDIST"] > 0) & (df_matches["CONF1"] > match_threshold) & (df_matches["CONF1"] < 1)].sort_values(by=["CONF1"])
            return df_good_matches  # does this return duplicates? like if there are multiple matches above the threshhold it needs to pick the highest one, NOT include all of them
        else:
            return df_matches


class NetworkAnalysis(BaseLandlordData):

    def __init__(self):
        super().__init__()

    @classmethod
    def set_network_name(cls, ntwk_id, df_rentals_component, final_component_column, network_name_column):
        df_networked = df_rentals_component.copy()
        df_networked[network_name_column] = np.nan
        unique_networks = df_networked[final_component_column].unique()

        for ntwk in unique_networks:
            df_subset = df_networked[df_networked[final_component_column] == ntwk]
            unique_names = list(df_subset["CLEAN_NAME"].dropna())
            name_counts = Counter(unique_names)
            sorted_names = [name for name, count in name_counts.most_common()]

            if sorted_names:
                network_name_short = f"{sorted_names[0]} Etc."
            else:
                network_name_short = f"Network {ntwk_id} - {ntwk}"

            concatenated_names = " -- ".join(sorted_names[:3])
            concatenated_names += f" -- {ntwk} -- ({ntwk_id})"

            # Handle case where there are no clean names
            df_networked.loc[
                df_networked[final_component_column] == ntwk, network_name_column
            ] = concatenated_names

            df_networked.loc[
                df_networked[final_component_column] == ntwk, f"{network_name_column}_SHORT"
            ] = network_name_short

        return df_networked

    @classmethod
    def set_network_text(
            cls,
            g,
            df_rentals_component: pd.DataFrame,
            network_name_column: str,
            final_component_column: str
    ):
        df_networked = df_rentals_component.copy()
        df_networked[f"{network_name_column}_TEXT"] = np.nan
        unique_networks = df_networked[final_component_column].unique()
        components = list(nx.connected_components(g))

        for ntwk in unique_networks:
            if ntwk == None or np.isnan(ntwk): continue
            nodes = list(g.subgraph(components[int(ntwk)]).nodes())
            edges = list(g.subgraph(components[int(ntwk)]).edges())
            if len(edges) == 0:
                network_text = json.dumps(nodes)
            else:
                network_text = json.dumps(edges)
            df_networked.loc[
                df_networked[final_component_column] == ntwk, f"{network_name_column}_TEXT"
            ] = network_text

        return df_networked



    @classmethod
    def set_component(
            cls,
            row: pd.Series,
            combosgMatches: dict,
            string_match_column: str,
            clean_core_column: str,
            clean_core_column_entity: str
    ):
        keys_to_check = [
            row[clean_core_column],
            row["GCD_FORMATTED_MATCH"],
            row[string_match_column],
            row[clean_core_column_entity],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"]
        ]
        for key in keys_to_check:
            if key in combosgMatches.keys():
                return combosgMatches[key]
        # If no match is found, print debug info and return np.nan
        # print(f"KeyError for CleanName: {row['CLEAN_NAME']} and ADDRESS: {row['CLEAN_ADDRESS']}")
        return np.nan

    @classmethod
    def check_name(cls, row):
        """
        Returns "True" if the row SHOULD be included in the network analysis, and False if it should be ignored.
        """
        return pd.isna(row["IS_COMMON_NAME"]) or row["IS_COMMON_NAME"] is False

    @classmethod
    def check_entity_name(cls, row, common_names):
        """
        Returns "True" if the row SHOULD be included in the network analysis, and False if it should be ignored.
        """
        return pd.isna(row["IS_COMMON_NAME"]) or row["IS_COMMON_NAME"] is False

    # processes nodes and edges for clean_name and clean_address ONLY
    @classmethod
    def process_row_network(
            cls,
            g: nx.Graph,
            row: pd.Series,
            df_analysis: pd.DataFrame,
            clean_core_column: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:
        """
        Adds nodes and edges for taxpayer name and taxpayer address. Uses clean address to check for inclusion
        detection and matching_address for the network graph

        Names that are NOT indicated as being common names are included. Addresses that pass the check_address test are
        included. If both the name and address pass the name and address checks, add them as nodes AND edges

        If the name passes the check_name test but the address does not pass the check_address test, add ONLY the name
        as a node.
        """

        name = row[clean_core_column]
        clean_address = row["GCD_FORMATTED_ADDRESS"]
        matching_address = row["GCD_FORMATTED_MATCH"]

        if cls.check_address(clean_address, df_analysis, include_orgs, include_unresearched) and cls.check_name(row):
            if pd.notnull(name) and not cls.is_encoded_empty(name) and pd.notnull(clean_address) and not cls.is_encoded_empty(clean_address):
                g.add_edge(name, matching_address)
        elif cls.check_name(row) and pd.notnull(name) and not cls.is_encoded_empty(name):
            g.add_node(name)

    # processes nodes and edges related EXCLUSIVELY to fuzzy_match_combo
    @classmethod
    def process_row_network_string_match(
            cls,
            g: nx.Graph,
            row: pd.Series,
            string_match_column: str,
            df_analysis: pd.DataFrame,
            clean_core_column: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:

        fuzzy_match_combo = row[string_match_column]
        clean_address = row["GCD_FORMATTED_ADDRESS"]
        name = row[clean_core_column]
        matching_address = row["GCD_FORMATTED_MATCH"]

        if cls.check_address(clean_address, df_analysis, include_orgs, include_unresearched) and cls.check_name(row):
            if pd.notnull(name) and pd.notnull(clean_address):
                g.add_edge(name, fuzzy_match_combo)
                g.add_edge(matching_address, fuzzy_match_combo)
        elif cls.check_name(row) and pd.notnull(name):
            g.add_edge(name, fuzzy_match_combo)
        else:
            g.add_node(fuzzy_match_combo)

    # processes nodes and edges related EXCLUSIVELY to entity_name and entity_address
    @classmethod
    def process_row_network_entity(
            cls,
            g: nx.Graph,
            row: pd.Series,
            df_analysis: pd.DataFrame,
            string_match_column: str,
            clean_core_column: str,
            clean_core_column_entity: str,
            include_orgs: bool,
            include_unresearched: bool
    ) -> None:

        entities_to_ignore: List[str] = ["CHICAGO TITLE LAND TRUST COMPANY", "CHICAGO TITLE LAND"]

        taxpayer_name = row[clean_core_column]
        entity_name = row[clean_core_column_entity]
        fuzzy_match_combo = row[string_match_column]
        entity_addresses = [
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3"],
        ]
        entity_matching_addresses = [
            row["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"],
            row["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"],
        ]

        for i, address in enumerate(entity_addresses):
            if cls.check_address(address, df_analysis, include_orgs, include_unresearched):
                if pd.notnull(address) and entity_name not in entities_to_ignore:
                    g.add_edge(entity_name, entity_matching_addresses[i])
                    g.add_edge(taxpayer_name, entity_matching_addresses[i])
                    if pd.notnull(fuzzy_match_combo):
                        g.add_edge(entity_name, fuzzy_match_combo)
                        g.add_edge(entity_matching_addresses[i], fuzzy_match_combo)
                elif pd.notnull(address):
                    g.add_edge(taxpayer_name, entity_matching_addresses[i])
                    if pd.notnull(fuzzy_match_combo):
                        g.add_edge(entity_matching_addresses[i], fuzzy_match_combo)



    @classmethod
    def rentals_network(
            cls,
            network_id: int,
            df_rentals: pd.DataFrame,
            df_analysis: pd.DataFrame,
            string_match_column: str,
            clean_core_column: str,
            clean_core_column_entity: str,
            include_orgs: bool = False,
            include_unresearched: bool = False
    ):

        start_time = time.time()

        df_unique = df_rentals.drop_duplicates(subset=["NAME_ADDRESS_CLEAN"])
        gMatches = nx.Graph()

        print("Creating nodes and edges...")
        for i, row in df_unique.iterrows():

            # 1. Add nodes & edges for taxpayer name and address
            cls.process_row_network(gMatches, row, df_analysis, clean_core_column, include_orgs, include_unresearched)

            # 2. Add nodes & edges for fuzzy match combo name (if present)
            if pd.notnull(row[string_match_column]):
                cls.process_row_network_string_match(
                    gMatches,
                    row,
                    string_match_column,
                    df_analysis,
                    clean_core_column,
                    include_orgs,
                    include_unresearched
                )

            # 3. Add nodes & edges for entity name and entity address (if present)
            if pd.notnull(row[clean_core_column_entity]):
                cls.process_row_network_entity(
                    g=gMatches,
                    row=row,
                    df_analysis=df_analysis,
                    string_match_column=string_match_column,
                    clean_core_column=clean_core_column,
                    clean_core_column_entity=clean_core_column_entity,
                    include_orgs=include_orgs,
                    include_unresearched=include_unresearched
                )


        print(f"(Elapsed time: {time.time() - start_time})")
        print("Mapping components to their corresponding properties...")
        taxpayer_names_set = list(set(df_unique[clean_core_column].dropna().unique()))
        fuzzy_matches_set = list(set(df_unique[string_match_column].dropna().unique()))
        clean_addresses_set = list(set(
                list(set(df_unique["GCD_FORMATTED_MATCH"].dropna().unique())) +
                list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_1_MATCH"].dropna().unique()))  #+
                # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_2_MATCH"].dropna().unique())) +
                # list(set(df_unique["GCD_FORMATTED_ADDRESS_ADDRESS_3_MATCH"].dropna().unique()))
        ))

        entity_names_set = list(set(df_unique[clean_core_column_entity].dropna().unique()))

        combosgMatches = {}
        for i, connections in enumerate(list(nx.connected_components(gMatches))):
            for component in connections:
                if component in taxpayer_names_set:
                    combosgMatches[component] = i
                elif component in fuzzy_matches_set:
                    combosgMatches[component] = i
                elif component in clean_addresses_set:
                    combosgMatches[component] = i
                elif component in entity_names_set:
                    combosgMatches[component] = i

        print(f"(Elapsed time: {time.time() - start_time})")
        print("Merging mapped components to rental dataset...")
        df_unique[f"FINAL_COMPONENT_{network_id+1}"] = df_unique.apply(
            lambda row: cls.set_component(
                row,
                combosgMatches,
                string_match_column,
                clean_core_column,
                clean_core_column_entity
            ),
            axis=1
        )
        print(f"(Elapsed time: {time.time() - start_time})")
        print("Done")
        return df_unique, gMatches
        # df_rentals = pd.merge(df_rentals, df_unique[["NAME_ADDRESS", "FINAL_COMPONENT"]], how="left", on="NAME_ADDRESS")
        # df_rentals = cls.combine_columns_parallel(df_rentals)

    @classmethod
    def build_edge(cls, g, node_a, node_b, common_names=None, common_addrs=None, row=None):
        if (common_names is None or node_a not in common_names) and (common_addrs is None or node_b not in common_addrs):
            g.add_edge(node_a, node_b)

    @classmethod
    def name_addr_network(cls, df_matches):
        gMatches = nx.Graph()
        print("Building network graph for string matching results...")
        with tqdm(total=len(df_matches)) as pbar:
            for i, row in df_matches.iterrows():
                cls.build_edge(gMatches, row["ORIGINAL_DOC"], row["MATCHED_DOC"])
                pbar.update(1)
        return gMatches

    @classmethod
    def process_name_addr_network(
            cls,
            match_count: int,
            df_filtered,  # this should be either filtered or unfiltered (ex: if we want to exclude org addrs from string matching but include them in the network analysis
            df_matches,
            gMatches,
            name_address_column
    ):

        """
        Network graph calculator used in the string matching workflow.
        :param match_count: Iteration variable "i" generated from looping through the cartesian product of the
        parameters using enumerate()
        :param df_filtered:
        :param df_matches:
        :param gMatches:
        :param name_address_column:
        :return:
        """

        # loop through each connected component
        combosgMatches = {}
        combosgMatchesNames = {}
        for i, connections in enumerate(list(nx.connected_components(gMatches))):

            # pull out name with the shortest length as representative "canonical" name for network
            shortest = min(connections, key=len)

            # store key/value pair for original name and new name in dictionary
            for component in connections:
                combosgMatches[component] = shortest

            shortest_two = sorted(connections, key=len)[:3]
            shortest_names = []
            for name in shortest_two:
                name_addr_split = name.split("-")
                shortest_names.append(name_addr_split[0].strip())

            # concatenate the two shortest names with " -- " as the separator
            canonical_name = ' -- '.join(shortest_names)

            # store key/value pair for original name and new name in dictionary
            for component in connections:
                combosgMatchesNames[component] = f"{canonical_name} -- {i}"

        # add new column for landlord network name
        df_matches["FUZZY_MATCH_NAME"] = df_matches["ORIGINAL_DOC"].apply(lambda x: combosgMatches[x])  # this is likely the redundant column
        df_matches["FUZZY_MATCH_COMBO"] = df_matches["ORIGINAL_DOC"].apply(lambda x: combosgMatchesNames[x])

        # merge clean name and clean address columns based on the simplified, calculated network name
        # df_matches = pd.merge(df_matches, df_filtered[['CLEAN_NAME', 'CLEAN_ADDRESS']], how='left', left_on='FUZZY_MATCH_NAME', right_on='CLEAN_NAME')

        # merge clean name and clean address columns based on the raw NameAddress string concatenation
        # df_matches = pd.merge(df_matches, df_filtered[['CLEAN_NAME', 'CLEAN_ADDRESS', 'NAME_ADDRESS']], how='left', left_on='FUZZY_MATCH_NAME', right_on='NAME_ADDRESS')

        # remove redundant columns
        df_matches = cls.combine_columns_parallel(df_matches)

        # the "clean names" here are the
        # df_matches.rename(columns={'CLEAN_NAME':'FUZZY_NAME', 'CLEAN_ADDRESS':'FUZZY_ADDRESS'}, inplace=True)

        # Keep good matches and join back to data
        # df_matches.drop_duplicates(subset=['FUZZY_NAME', 'FUZZY_ADDRESS', 'ORIGINAL_DOC'], inplace=True)
        df_filtered = pd.merge(df_filtered, df_matches[["ORIGINAL_DOC", "FUZZY_MATCH_COMBO"]], how="left", left_on=name_address_column, right_on="ORIGINAL_DOC")

        # fill in empty rows with the name of their corresponding CleanName and CleanAddress values from the new dataframe
        # df_filtered['FUZZY_NAME'].fillna(df_filtered['CLEAN_NAME'], inplace=True)
        # df_filtered['FUZZY_ADDRESS'].fillna(df_filtered['CLEAN_ADDRESS'], inplace=True)
        df_filtered = df_filtered.rename(columns={"FUZZY_MATCH_COMBO": f"STRING_MATCHED_NAME_{match_count+1}"})

        return df_filtered
