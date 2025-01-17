from classes.base_data import BaseData
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from constants.landlord_fields import LandlordFields
from constants.constants import DATA_ROOT
from constants.constants import CookCounty as cc
import os

class BaseScrape(BaseData):

    sc = LandlordFields.PropertyScrapeFields

    def __init__(self):
        super().__init__()

    @classmethod
    def rename_columns(cls, df):
        for col_name_old, col_name_new in cc.SCRAPE_FIELDS.items():
            df.rename(columns={col_name_old: col_name_new}, inplace=True)
        return df

    @classmethod
    def save_partial_scrape(cls, records):
        timestamp = cls.get_timestamp()
        df_partial = pd.DataFrame(records)
        df_partial = cls.rename_columns(df_partial)
        partial_scrape_path = f"{DATA_ROOT}/scrape/partials/partial_scrape_{timestamp}.csv"
        cls.save_df(df_partial, partial_scrape_path, DATA_ROOT)

    @classmethod
    def set_raw_address(cls, row):
        address = row[cls.sc.TAXPAYER_ADDRESS].strip() if isinstance(row[cls.sc.TAXPAYER_ADDRESS], str) else ""
        city = row[cls.sc.TAXPAYER_CITY].strip() if isinstance(row[cls.sc.TAXPAYER_CITY], str) else ""
        state = row[cls.sc.TAXPAYER_STATE].strip() if isinstance(row[cls.sc.TAXPAYER_STATE], str) else ""
        zip_code = row[cls.sc.TAXPAYER_ZIP].strip() if isinstance(row[cls.sc.TAXPAYER_ZIP], str) else ""
        return f"{address}, {city}, {state} {zip_code}"

    # ----------------------------–
    # ----MAIN SCRAPER FUNCTION----
    # -----------------------------
    @classmethod
    def scrape(cls, pin):

        data = {"PIN": str(pin)}

        try:

            time.sleep(0.01)
            # total_requests += 1

            # BUILDING PROFILE PAGE - general info for building
            page = requests.get(f"{cc.COOK_COUNTY_SCRAPE}{pin}", timeout=10)
            soup = BeautifulSoup(page.content, "html.parser")
            table = soup.find("table", {"id": "PIN Info"})

            # field names for scraper output - correspond to HTML associated with each field
            fields = cc.SCRAPE_FIELDS

            if table:
                for row in table.find_all("tr"):
                    columns = row.find_all("td")
                    if len(columns) == 2:
                        field_name = columns[0].text.strip()
                        val = columns[1].text.strip()
                        if field_name in fields.keys():
                            data[field_name] = val
            else:
                data["Success?"] = False
                data["Error"] = "Building profile data_root table not found"
                return data

            # PROPERTY TAX PAYER PAGE - taxpayer info for building
            taxpayer_data_url = None
            for link in soup.find_all("a", href=True):
                span = link.find("span")
                if span and "Taxpayer Data" in span.text:
                    taxpayer_data_url = link["href"]
                    break

            if taxpayer_data_url:
                taxpayer_data_url = taxpayer_data_url[2:]
                if not taxpayer_data_url.startswith("http"):
                    taxpayer_data_url = f"https://assessorpropertydetails.cookcountyil.gov{taxpayer_data_url}"

                taxpayer_page = requests.get(taxpayer_data_url)
                taxpayer_soup = BeautifulSoup(taxpayer_page.content, "html.parser")
                taxpayer_div = taxpayer_soup.find("div", {"id": "datalet_div_0"})

                if taxpayer_div:
                    tables = taxpayer_div.find_all("table")
                    if len(tables) > 1:
                        taxpayer_table = tables[1]
                        for row in taxpayer_table.find_all("tr"):
                            columns = row.find_all("td")
                            if len(columns) == 2:
                                field_name = columns[0].text.strip()
                                val = columns[1].text.strip()
                                if field_name in fields.keys():
                                    data[field_name] = val
                else:
                    data["Success?"] = False
                    data["Error"] = "Building Taxpayer data_root table not found"
                    return data

            data["Error"] = ""
            data["Success?"] = True
            # successful_responses += 1
            return data

        # return empty row with error message to keep track of failed/bunk PINs
        except Exception as e:
            data["Success?"] = False
            data["Error"] = str(e)
            return data

    # ----------------------------–---------
    # ----UPDATE SCRAPER STATS FUNCTIONS----
    # --------------------------------------
    @classmethod
    def get_pins_zip_count(cls, df_pins, zip_):
        df_pins_zip = df_pins[df_pins["Post_Code"] == zip_]
        return len(df_pins_zip)

    @classmethod
    def get_pins_total_count(cls, path_to_dir, zip_):
        file_names = os.listdir(path_to_dir)
        for file in file_names:
            file_name = os.path.splitext(file)[0]
            if not file_name.startswith("."):
                zip_code = int(file_name[-5:])
                if zip_code == zip_:
                    file_to_read = f"{path_to_dir}/{file_name}.csv"
        if file_to_read:
            df = pd.read_csv(file_to_read)
            return len(df)
        else:
            raise

    @classmethod
    def get_count_no_data(cls, path_to_failures, zip_):
        file_names = os.listdir(path_to_failures)
        for file in file_names:
            file_name = os.path.splitext(file)[0]
            if not file_name.startswith("."):
                zip_code = int(file_name[-5:])
                if zip_code == zip_:
                    file_to_read = f"{path_to_failures}/{file_name}.csv"
        if file_to_read:
            df = pd.read_csv(file_to_read)
            df_no_data = df[df["Error"] == "Building profile data_root table not found"]
            return len(df_no_data)
        else:
            raise

    @classmethod
    def update_stats(cls, path_to_successes, path_to_failures, path_to_stats_csv, zip_, df_pins):
        df_stats = pd.read_csv(path_to_stats_csv)
        zips = df_stats["zip_code"].tolist()
        if zip_ in zips:
            df_stats = df_stats[df_stats["zip_code"] != zip_]
        count_total_pins = cls.get_pins_zip_count(df_pins, zip_)
        count_successes = cls.get_pins_total_count(path_to_successes, zip_)
        count_failures = cls.get_pins_total_count(path_to_failures, zip_)
        count_failures_no_data = cls.get_count_no_data(path_to_failures, zip_)
        row = {
            "zip_code": zip_,
            "total_pins": count_total_pins,
            "success_rate": round(count_successes/count_total_pins, 2) * 100,
            "successes": count_successes,
            "failures": count_failures,
            "failures_no_data": count_failures_no_data,
            "failure_rate_no_data": round(count_failures_no_data/count_failures, 2) * 100
        }
        df_stats = pd.concat([df_stats, pd.DataFrame([row])], ignore_index=True)
        df_stats.to_csv(path_to_stats_csv, index=False)
        print(f"Scraper stats updated: {path_to_stats_csv}")

