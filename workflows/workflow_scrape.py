import time
import traceback

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from constants.landlord_fields import LandlordFields as lf

# from tqdm.notebook import tqdm
from tqdm import tqdm

from classes.base_data import BaseData
from classes.base_scrape import BaseScrape
from constants.constants import DATA_ROOT, CookCounty
from constants.property_fields import PropertyFields


# python3 -m workflows.workflow_scrape

class WkflScrape(BaseData):

    """
    Runs scraper on the Cook County Assessor's site to fetch property taxpayer records. Fetches properties based on
    their PINs. Saves partial scrapes to not lose progress in the "partials" directory, so that it can be stopped and
    started back up at a later time and pick up where it left off.

    By default, the scraper only scrapes for pins that are NOT found in the "partials" directory. This behavior can be
    easily modified by either adjusting the get_pins_to_scrape() function, or by creating a new function that scrapes
    a specified selection of pins.
    """

    MASTER_SCRAPE = f"{DATA_ROOT}/scrape/scrape_2025_01_10_11_38_09.csv"
    ADDRESS_POINTS = f"{DATA_ROOT}/property_data_clean/address_points_all.csv"
    SCRAPE_PARTIALS = f"{DATA_ROOT}/scrape/partials"
    SCRAPE_SUCCESS = f"{DATA_ROOT}/scrape/scrape_2025_01_10_11_38_09_success.csv"
    SCRAPE_FAILURE = f"{DATA_ROOT}/scrape/scrape_failure"

    def __init__(self, interval=100):
        super().__init__()
        self.ps = lf.PropertyScrapeFields
        self.ap = PropertyFields.AddressPointsFields
        self.df_addrs = self.get_df(self.ADDRESS_POINTS, self.ap.DTYPES)
        self.interval = interval
        self.scrape_out_path = ""

    def get_pins_to_scrape(self):
        """
        Returns list of PINs that have NOT been scraped yet by combining the partial scrape files contained
        in the partials directory.
        """
        pins_all = list(self.df_addrs["PIN"].unique())
        df_combined = self.combine_partials(self.SCRAPE_PARTIALS, self.ps.DTYPES)
        if df_combined is None: return pins_all
        pins_scraped = list(df_combined["PIN"].unique())
        pins_to_scrape = list(set(pins_all) - set(pins_scraped))
        return pins_to_scrape

    def run_scrape(self, pins_to_scrape):

        try:

            records = []
            start_time = time.time()

            # initiate thread pool
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for i, pin in enumerate(pins_to_scrape):
                    future = executor.submit(BaseScrape.scrape, str(int(pin)))
                    futures[future] = i

                # initiate tqdm for progress bar tracking
                with tqdm(total=len(pins_to_scrape), desc="Scraping property data...") as pbar:
                    for future in as_completed(futures):
                        i = futures[future]
                        result = future.result()
                        if result:
                            records.append(result)
                        pbar.update(1)
                        if self.interval is not None:
                            records = self.save_partial(records, self.interval, BaseScrape.save_partial_scrape)
                    if self.interval is not None and records:
                        BaseScrape.save_partial_scrape(records)

            # if self.interval is None:
            #     timestamp = self.get_timestamp()
            #     df_final = pd.DataFrame(records)
            #     df_final = BaseScrape.rename_columns(df_final)
            #     self.scrape_out_path = f"{DATA_ROOT}/scrape/scrape_{timestamp}"
            #     self.save_df(df_final, self.scrape_out_path, DATA_ROOT)

            # log time
            end_time = time.time()
            print(f"Elapsed time: {round((end_time - start_time), 2)} minutes")

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def combine_partials_save(self):
        timestamp = self.get_timestamp()
        df_combined = self.combine_partials(self.SCRAPE_PARTIALS, self.ps.DTYPES)
        df_combined = BaseScrape.rename_columns(df_combined)
        df_combined[self.ps.RAW_ADDRESS] = df_combined.apply(
            lambda row: BaseScrape.set_raw_address(row), axis=1
        )
        self.scrape_out_path = f"{DATA_ROOT}/scrape/scrape_{timestamp}.csv"
        self.save_df(df_combined, self.scrape_out_path, DATA_ROOT)

    # def concatenate_to_master(self):
    #     df_master = self.get_df(self.MASTER_SCRAPE, self.ps.DTYPES)
    #     df_scrape = self.get_df(self.scrape_out_path, self.ps.DTYPES)
    #     df_scrape_success = df_scrape[df_scrape[self.ps.SCRAPE_SUCCESS] == True]
    #     df_out = pd.concat([df_master, df_scrape_success], ignore_index=True)
    #     self.save_df(df_out, self.MASTER_SCRAPE, DATA_ROOT)

    def separate_success_failure(self):
        df_scrape = self.get_df(self.MASTER_SCRAPE, self.ps.DTYPES)
        self.save_df(
            df_scrape[df_scrape[self.ps.SCRAPE_SUCCESS] == True],
            f"{self.MASTER_SCRAPE[:-4]}_success.csv",
            DATA_ROOT
        )
        self.save_df(
            df_scrape[df_scrape[self.ps.SCRAPE_SUCCESS] == False],
            f"{self.MASTER_SCRAPE[:-4]}_failure.csv",
            DATA_ROOT
        )

    def separate_chicago(self):
        """
        Saves copy of scrape dataset containing only successfully scraped properties with Chicago zip codes
        """
        df_scrape = self.get_df(self.SCRAPE_SUCCESS, self.ps.DTYPES)
        df_chicago = self.df_addrs[self.df_addrs[self.ap.PROP_ZIP_CODE].isin(CookCounty.CHICAGO_ZIPS)]
        chicago_pins = list(df_chicago[self.ap.PIN].unique())
        self.save_df(
            df_scrape[df_scrape[self.ps.PIN].isin(chicago_pins)],
            f"{self.SCRAPE_SUCCESS[:-4]}_chicago.csv",
        )

    def workflow(self):
        pins_to_scrape = self.get_pins_to_scrape()
        self.run_scrape(pins_to_scrape)
        self.combine_partials_save()
        self.separate_success_failure()
        self.separate_chicago()
        # if self.interval is not None:
        #     self.combine_partials_save()
        # self.concatenate_to_master()

# wkfl_scrape = WkflScrape()
# wkfl_scrape.workflow()