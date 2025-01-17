class Geocodio:
    API_KEY = ""
    GEOCODIO_URL = "https://api.geocod.io/v1.7/geocode?api_key=" + API_KEY + "&q="

class CookCounty:
    COOK_COUNTY_SCRAPE = "https://assessorpropertydetails.cookcountyil.gov/Datalets/Datalet.aspx?UseSearch=no&pin="
    SCRAPE_FIELDS = {
        "Parcel #": "PARCEL_NUM",
        "Neighborhood": "NEIGHBORHOOD",
        "Tax District": "TAX_DISTRICT",
        "Key PIN": "KEY_PIN",
        "PIN:": "PIN",
        "Town Name": "TOWN_NAME",
        "Tri-Town": "TRI-TOWN",
        "Tax Year": "TAX_YEAR",
        "Property Address": "PROP_ADDRESS",
        "Building/Unit #:": "PROP_UNIT_NUM",
        "City & Zip Code": "PROP_CITY_ZIP",
        "Multiple Addresses:": "MULTIPLE_ADDRESSES",
        "Class": "CLASS",
        "Taxpayer Name": "TAXPAYER_NAME",
        "Taxpayer Name 2": "TAXPAYER_NAME_2",
        "Address Type:": "ADDRESS_TYPE",
        "Address:": "TAXPAYER_ADDRESS",
        "City:": "TAXPAYER_CITY",
        "State": "TAXPAYER_STATE",
        "Zip": "TAXPAYER_ZIP",
        "Success?": "SCRAPE_SUCCESS?",
        "Error": "ERROR"
    }
    ALL_ZIPS = [60004, 60005, 60007, 60008, 60010, 60013, 60015, 60016, 60018, 60022, 60025, 60026, 60029, 60035, 60043, 60053, 60056, 60062, 60067, 60068, 60070, 60074, 60076, 60077, 60082, 60089, 60090, 60091, 60093, 60095, 60103, 60104, 60107, 60109, 60120, 60126, 60130, 60131, 60133, 60141, 60153, 60154, 60155, 60160, 60162, 60163, 60164, 60165, 60169, 60171, 60172, 60173, 60176, 60179, 60192, 60193, 60194, 60195, 60196, 60197, 60198, 60199, 60200, 60201, 60202, 60203, 60204, 60205, 60206, 60207, 60208, 60209, 60210, 60211, 60212, 60213, 60214, 60215, 60216, 60217, 60218, 60219, 60220, 60221, 60222, 60223, 60224, 60225, 60226, 60227, 60228, 60229, 60230, 60231, 60232, 60233, 60234, 60235, 60301, 60302, 60304, 60305, 60340, 60402, 60406, 60409, 60411, 60415, 60418, 60419, 60422, 60423, 60425, 60426, 60428, 60429, 60430, 60438, 60439, 60443, 60445, 60449, 60452, 60453, 60455, 60456, 60457, 60458, 60459, 60461, 60462, 60463, 60464, 60465, 60466, 60467, 60469, 60471, 60472, 60473, 60475, 60476, 60477, 60478, 60480, 60482, 60487, 60501, 60513, 60521, 60523, 60525, 60526, 60527, 60534, 60546, 60558, 60666, 60706, 60712, 60714, 60803, 60804, 60805, 60827]
    CHICAGO_ZIPS = [
        "60106",
        "60601",
        "60602",
        "60603",
        "60604",
        "60605",
        "60606",
        "60607",
        "60608",
        "60609",
        "60610",
        "60611",
        "60612",
        "60613",
        "60614",
        "60615",
        "60616",
        "60617",
        "60618",
        "60619",
        "60620",
        "60621",
        "60622",
        "60623",
        "60624",
        "60625",
        "60626",
        "60628",
        "60629",
        "60630",
        "60631",
        "60632",
        "60633",
        "60634",
        "60636",
        "60637",
        "60638",
        "60639",
        "60640",
        "60641",
        "60642",
        "60643",
        "60644",
        "60645",
        "60646",
        "60647",
        "60649",
        "60651",
        "60652",
        "60653",
        "60654",
        "60655",
        "60656",
        "60657",
        "60659",
        "60660",
        "60661",
        "60706",
        "60707",
        "60827"
    ]


DATA_ROOT = "/Users/dpederson/Library/CloudStorage/ProtonDrive-director@landlordmapper.org-folder/data/datasets/chi2"

NUM_APTS = {
    "One": "1",
    "Two": "2",
    "Three": "3",
    "Four": "4",
    "Five": "5",
    "Six": "6"
}

PREDIRECTIONALS = {
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "NORTHEAST": "NE",
    "NORTHWEST": "NW",
    "SOUTHEAST": "SE",
    "SOUTHWEST": "SW"
}

LANDLORD_DTYPES = {
    "GCD_NUMBER": str,
    "GCD_ZIP": str,
    "GCD_SECONDARYNUMBER": str,
    "ZIP": str,
    "TAXPAYER_ZIP": str,
    "PIN": str,
    "CLASS": str,
    "KEYPIN": str,
    "CLASS(ES)": str,
    "CORP-STATUS": int,
    "CORP-FILE-NUMBER": str,
    "LL-STATUS-CODE": int,
    "LL-FILE-NUMBER": str,
    "GCD_NUMBER_ADDRESS_1": str,
    "GCD_SECONDARYNUMBER_ADDRESS_1": str,
    "GCD_ZIP_ADDRESS_1": str,
    "GCD_NUMBER_ADDRESS_2": str,
    "GCD_SECONDARYNUMBER_ADDRESS_2": str,
    "GCD_ZIP_ADDRESS_2": str,
    "GCD_NUMBER_ADDRESS_3": str,
    "GCD_SECONDARYNUMBER_ADDRESS_3": str,
    "GCD_ZIP_ADDRESS_3": str,
    "TRUST_ID": str
}