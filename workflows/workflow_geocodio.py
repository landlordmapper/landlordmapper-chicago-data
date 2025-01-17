import time
import traceback
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from tqdm.notebook import tqdm

from classes.base_data import BaseData
from classes.base_geocodio import BaseGeocodio, GeocodioResults, GeocodioTaxpayer
from constants.constants import DATA_ROOT, PREDIRECTIONALS
from constants.landlord_fields import LandlordFields


class WkflGeocodio(BaseData):

    """
    Dataframe passed as parameter shall contain all addresses to be run through Geocodio.

    TO-DO: Add flag for whether or not the addresses being processed are property taxpayers or corporations/LLCs. Or,
    alternatively, adjust the process_results function to be standardized and address type-agnostic.
    """

    MASTER_SCRAPE = f"{DATA_ROOT}/scrape/scrape_2025_01_10_11_38_09_success_chicago.csv"
    PROP_VALIDATED = f"{DATA_ROOT}/addresses/prop_validated.csv"
    PROP_UNVALIDATED = f"{DATA_ROOT}/addresses/prop_unvalidated.csv"
    GCD_PARTIALS = f"{DATA_ROOT}/geocodio/partials2"

    def __init__(self, interval=100):
        super().__init__()
        self.interval = interval
        self.va = LandlordFields.ValidatedAddressFields
        self.ps = LandlordFields.PropertyScrapeFields
        self.gcd_out_path = ""
        self.gcd_processed_validated_path = ""
        self.gcd_processed_unvalidated_path = ""
        self.gcd_processed_failed_path = ""

    def get_addrs_to_validate(self):
        # fetch all unique addresses from master geocodio validation list
        df_validated = self.get_df(self.PROP_VALIDATED, self.va.DTYPES)
        validated_addrs = list(df_validated[self.va.RAW_ADDRESS].unique())
        df_unvalidated = self.get_df(self.PROP_VALIDATED, self.va.DTYPES)
        unvalidated_addrs = list(df_unvalidated[self.va.RAW_ADDRESS].unique())
        # fetch all unique addresses from partial scrapes
        df_partials = self.combine_partials(self.GCD_PARTIALS, self.va.DTYPES)
        if df_partials != None:
            validated_partials = list(df_partials[self.va.RAW_ADDRESS].unique())
        else:
            validated_partials = []
        # fetch all unique addresses from scrape
        df_scrape = self.get_df(self.MASTER_SCRAPE, self.ps.DTYPES)
        df_addrs = df_scrape[~df_scrape[self.ps.RAW_ADDRESS].isin(validated_addrs+unvalidated_addrs+validated_partials)]
        # set self.df_addrs
        self.df_addrs = df_addrs.drop_duplicates(subset=[self.ps.RAW_ADDRESS])

    def run_geocodio(self):

        try:

            gcd_results = []
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=10) as executor:

                # store all unique addresses from dataframe into future object
                futures = {}
                for i, row in self.df_addrs.iterrows():
                    future = executor.submit(BaseGeocodio.call_geocodio, row[self.va.RAW_ADDRESS])
                    futures[future] = (i, row)

                # loop through futures object, executing geocodio calls for each one
                with tqdm(total=len(self.df_addrs), desc="Executing geocodio API calls...") as pbar:
                    for future in as_completed(futures):
                        try:
                            i, row = futures[future]
                            results = future.result()
                            if results and len(results) > 0:
                                # creates a new row per geocodio result
                                # ex: if one address search returned 5 results, this creates 5 rows with the same raw
                                    # address but with each result field
                                for result in results:
                                    new_row = GeocodioTaxpayer.process_results(row, result)
                                    gcd_results.append(new_row)
                            else:
                                new_row = GeocodioTaxpayer.process_results_none(row)
                                gcd_results.append(new_row)
                            pbar.update(1)
                            # save geocodio partial and empty out gcd_results
                            if self.interval is not None and len(gcd_results) >= self.interval:
                                BaseGeocodio.save_partial_geocodio(gcd_results)
                                gcd_results = []
                        except Exception as e:
                            print(f"Error: {e}")
                            print(traceback.format_exc())
                    if self.interval is not None and gcd_results:
                        BaseGeocodio.save_partial_geocodio(gcd_results)

            if self.interval is None:
                timestamp = self.get_timestamp()
                df_final = pd.DataFrame(gcd_results)
                self.gcd_out_path = f"{DATA_ROOT}/geocodio/geocodio_results_{timestamp}.csv"
                self.save_df(df_final, self.gcd_out_path, DATA_ROOT)

            # log time
            end_time = time.time()
            print(f"Elapsed time: {round((end_time - start_time), 2)} minutes")

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())

    def combine_partials_save(self):
        """Combines geocodio results partials into a single dataframe and saves."""
        timestamp = self.get_timestamp()
        df_combined = self.combine_partials(self.GCD_PARTIALS, self.va.DTYPES)
        self.gcd_out_path = f"{DATA_ROOT}/geocodio/geocodio_results_{timestamp}.csv"
        self.save_df(df_combined, self.gcd_out_path, DATA_ROOT)

    def process_geocodio_results(self, path):

        validated_rows = []
        unvalidated_rows = []
        no_results = []

        df_gcd_results = self.get_df(path, self.va.DTYPES)
        unique_addrs = list(df_gcd_results[self.va.RAW_ADDRESS].unique())

        with tqdm(total=len(unique_addrs), desc=f"Filtering geocodio results...") as pbar:
            for addr in unique_addrs:
                GeocodioResults.process_address(df_gcd_results, addr, validated_rows, unvalidated_rows, no_results)
                pbar.update(1)

        df_validated = pd.DataFrame(validated_rows)
        df_unvalidated = pd.DataFrame(unvalidated_rows)
        df_failed = pd.DataFrame(no_results)

        if not df_validated.empty:
            df_validated = df_validated.astype(
                {self.va.TAXPAYER_ZIP: str, self.va.GCD_NUMBER: str, self.va.GCD_ZIP: str}
            )
        if not df_unvalidated.empty:
            df_unvalidated = df_unvalidated.astype(
                {self.va.TAXPAYER_ZIP: str, self.va.GCD_NUMBER: str, self.va.GCD_ZIP: str}
            )
        if not df_failed.empty:
            df_failed = df_failed.astype(
                {self.va.TAXPAYER_ZIP: str, self.va.GCD_NUMBER: str, self.va.GCD_ZIP: str}
            )

        timestamp = self.get_timestamp()
        self.gcd_processed_validated_path = f"{DATA_ROOT}/geocodio/geocodio_results_validated_{timestamp}.csv"
        self.gcd_processed_unvalidated_path = f"{DATA_ROOT}/geocodio/geocodio_results_unvalidated_{timestamp}.csv"
        self.gcd_processed_failed_path = f"{DATA_ROOT}/geocodio/geocodio_results_failed_{timestamp}.csv"
        self.save_df(df_validated, self.gcd_processed_validated_path, DATA_ROOT)
        self.save_df(df_unvalidated, self.gcd_processed_unvalidated_path, DATA_ROOT)
        self.save_df(df_failed, self.gcd_processed_failed_path, DATA_ROOT)

    def concatenate_to_master(self, path_validated, path_unvalidated):
        df_valid_props = self.get_df(self.PROP_VALIDATED, self.va.DTYPES)
        df_unvalid_props = self.get_df(self.PROP_UNVALIDATED, self.va.DTYPES)
        df_valid_results = self.get_df(path_validated, self.va.DTYPES)
        df_valid_results.drop_duplicates(subset=[self.va.RAW_ADDRESS], inplace=True)
        df_unvalid_results = self.get_df(path_unvalidated, self.va.DTYPES)
        df_valid_out = pd.concat([df_valid_props, df_valid_results], ignore_index=True)
        df_valid_out.drop_duplicates(subset=[self.va.RAW_ADDRESS], inplace=True)
        df_unvalid_out = pd.concat([df_unvalid_props, df_unvalid_results], ignore_index=True)
        self.save_df(df_valid_out, f"{self.PROP_VALIDATED[:-4]}_new.csv", DATA_ROOT)
        self.save_df(df_unvalid_out, f"{self.PROP_UNVALIDATED[:-4]}_new.csv", DATA_ROOT)

    def workflow(self):
        self.get_addrs_to_validate()
        self.run_geocodio()
        if self.interval is not None:
            self.combine_partials_save()
        self.process_geocodio_results(self.gcd_out_path)
        self.concatenate_to_master(self.gcd_processed_validated_path, self.gcd_processed_unvalidated_path)
