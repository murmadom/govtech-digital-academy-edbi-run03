"""POI Pipeline Utilities

Shared functions for the POI data pipeline (full load & incremental).
Used by: POI EDA & QA, POI Data Transformation, POI Incremental Load.
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 1. CLEAN NULLS
# ══════════════════════════════════════════════════════════════════════

def clean_nulls(df):
    """Replace string null equivalents with NaN in all object columns.

    Handles: 'null', 'nil', 'none', 'n/a', 'na', 'nan', '', ' '
    """
    df = df.copy()
    null_strings = ["null", "nil", "none", "n/a", "na", "nan", "", " "]
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].astype(str).str.strip().str.lower().isin(null_strings)
        df.loc[mask, col] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. QA POI
# ══════════════════════════════════════════════════════════════════════

def qa_poi(df):
    """Apply QA rules to raw POI data.

    Steps:
        1. Replace string nulls with NaN
        2. Remove rows where NRIC is null
        3. Rename 'Integer' column to 'Year of Profile'
        4. Dedup by NRIC keeping latest Year of Profile
        5. Remove rows where birth year > Year of Profile

    Returns:
        Cleaned pandas DataFrame
    """
    rows_before = len(df)
    df = clean_nulls(df)

    # Remove null NRICs
    df = df[df["NRIC"].notna()]
    print(f"  After removing null NRICs: {len(df)} rows")

    # Rename Integer → Year of Profile
    df = df.rename(columns={"Integer": "Year of Profile"})

    # Dedup by NRIC keeping latest Year of Profile
    df = (df
        .sort_values(["NRIC", "Year of Profile"], ascending=[True, False])
        .drop_duplicates(subset=["NRIC"], keep="first"))
    print(f"  After dedup by NRIC: {len(df)} rows")

    # DOB validation: birth year must be <= Year of Profile
    df["_year_of_birth"] = pd.to_datetime(df["DOB"], dayfirst=True, errors="coerce").dt.year
    df = df[df["_year_of_birth"] <= df["Year of Profile"]]
    df = df.drop(columns=["_year_of_birth"])
    print(f"  After DOB validation: {len(df)} rows")

    print(f"  QA complete: {rows_before} → {len(df)} rows ({rows_before - len(df)} removed)")
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. ENRICH POI
# ══════════════════════════════════════════════════════════════════════

# ── Mapping dictionaries (module-level constants) ──

DISTRICT_MAP = {
    "01": "D01 - Raffles Place, Marina", "02": "D02 - Tanjong Pagar, Chinatown",
    "03": "D03 - Queenstown, Tiong Bahru", "04": "D04 - Telok Blangah, HarbourFront",
    "05": "D05 - Pasir Panjang, Clementi", "06": "D06 - City Hall, Clarke Quay",
    "07": "D07 - Beach Road, Bugis", "08": "D08 - Little India, Farrer Park",
    "09": "D09 - Orchard, River Valley", "10": "D10 - Bukit Timah, Holland",
    "11": "D11 - Newton, Novena", "12": "D12 - Balestier, Toa Payoh",
    "13": "D13 - Macpherson, Potong Pasir", "14": "D14 - Geylang, Paya Lebar",
    "15": "D15 - East Coast, Katong", "16": "D16 - Bedok, Upper East Coast",
    "17": "D17 - Changi, Loyang", "18": "D18 - Tampines, Pasir Ris",
    "19": "D19 - Serangoon, Hougang", "20": "D20 - Bishan, Ang Mo Kio",
    "21": "D21 - Clementi, Upper Bukit Timah", "22": "D22 - Jurong, Boon Lay",
    "23": "D23 - Bukit Batok, Bukit Panjang", "24": "D24 - Lim Chu Kang, Tengah",
    "25": "D25 - Woodlands, Admiralty", "26": "D26 - Mandai, Upper Thomson",
    "27": "D27 - Yishun, Sembawang", "28": "D28 - Seletar, Yio Chu Kang",
}

REGION_MAP = {
    # Southeast Asia
    "Bruneian": "Southeast Asia", "Cambodian": "Southeast Asia", "East Timorese": "Southeast Asia",
    "Filipino": "Southeast Asia", "Indonesian": "Southeast Asia", "Laotian": "Southeast Asia",
    "Malaysian": "Southeast Asia", "Myanmar": "Southeast Asia", "Singapore Citizen": "Southeast Asia",
    "Thai": "Southeast Asia", "Timorense": "Southeast Asia", "Vietnamese": "Southeast Asia",
    # East Asia
    "Chinese": "East Asia", "Chinese/hongkong Sar": "East Asia", "Chinese/macau Sar": "East Asia",
    "Chinese/taiwanese": "East Asia", "Japanese": "East Asia", "Mongolian": "East Asia",
    '"""korean"': "East Asia",
    # South Asia
    "Bangladeshi": "South Asia", "Bhutanese": "South Asia", "Indian": "South Asia",
    "Maldivian": "South Asia", "Nepalese": "South Asia", "Pakistani": "South Asia",
    "Sri Lankan": "South Asia",
    # Central Asia
    "Kazakhstani": "Central Asia", "Kyrghis": "Central Asia", "Kyrgyzstan": "Central Asia",
    "Tajikistani": "Central Asia", "Turkmen": "Central Asia", "Uzbekistan": "Central Asia",
    # Middle East
    "Afghan": "Middle East", "Bahraini": "Middle East", "Iranian": "Middle East",
    "Iraqi": "Middle East", "Israeli": "Middle East", "Jordanian": "Middle East",
    "Kuwaiti": "Middle East", "Lebanese": "Middle East", "Omani": "Middle East",
    "Palestinian": "Middle East", "Qatari": "Middle East", "Saudi Arabian": "Middle East",
    "Syrian": "Middle East", "Turk": "Middle East", "United Arab Emirates": "Middle East",
    "Yemeni": "Middle East",
    # Europe
    "Albanian": "Europe", "Andorran": "Europe", "Armenian": "Europe", "Austrian": "Europe",
    "Azerbaijani": "Europe", "Belarussian": "Europe", "Belgian": "Europe", "Bosnian": "Europe",
    "British": "Europe", "British National Overseas": "Europe", "British Overseas Citizen": "Europe",
    "British Overseas Territories Citizen": "Europe", "British Protected Person": "Europe",
    "British Subject": "Europe", "Bulgarian": "Europe", "Croatian": "Europe", "Cypriot": "Europe",
    "Czech": "Europe", "Danish": "Europe", "Estonian": "Europe", "Finnish": "Europe",
    "French": "Europe", "Georgian": "Europe", "German": "Europe", "Greek": "Europe",
    "Hungarian": "Europe", "Icelander": "Europe", "Irish": "Europe", "Italian": "Europe",
    "Kosovar": "Europe", "Latvian": "Europe", "Liechtensteiner": "Europe", "Lithuanian": "Europe",
    "Luxembourger": "Europe", "Macedonian": "Europe", "Maltese": "Europe", "Moldavian": "Europe",
    "Monacan": "Europe", "Montenegrin": "Europe", "Netherlands": "Europe", "Norwegian": "Europe",
    "Polish": "Europe", "Portuguese": "Europe", "Romanian": "Europe", "Russian": "Europe",
    "Sammarinese": "Europe", "Serbian": "Europe", "Slovak": "Europe", "Slovenian": "Europe",
    "Spanish": "Europe", "Swedish": "Europe", "Swiss": "Europe", "Ukrainian": "Europe",
    "Vatican City State (holy See)": "Europe", "Yugoslavians": "Europe",
    # Africa
    "Algerian": "Africa", "Angolan": "Africa", "Beninese": "Africa", "Botswana": "Africa",
    "Burkinabe": "Africa", "Burundian": "Africa", "Cameroonian": "Africa",
    "Cape Verdean": "Africa", "Central African Republic": "Africa", "Chadian": "Africa",
    "Comoran": "Africa", "Congolese": "Africa", "Democratic Republic Of The Congo": "Africa",
    "Djiboutian": "Africa", "Egyptian": "Africa", "Equatorial Guinea": "Africa",
    "Eritrean": "Africa", "Ethiopian": "Africa", "Gabon": "Africa", "Gambian": "Africa",
    "Ghanaian": "Africa", "Guinean": "Africa", "Guinean (bissau)": "Africa",
    "Ivory Coast": "Africa", "Kenyan": "Africa", "Lesotho": "Africa", "Liberian": "Africa",
    "Libyan": "Africa", "Madagasy": "Africa", "Malawian": "Africa", "Malian": "Africa",
    "Mauritanean": "Africa", "Mauritian": "Africa", "Moroccan": "Africa",
    "Mozambican": "Africa", "Namibian": "Africa", "Niger": "Africa", "Nigerian": "Africa",
    "Reunionese": "Africa", "Rwandan": "Africa", "Sao Tomean": "Africa",
    "Senegalese": "Africa", "Seychellois": "Africa", "Sierra Leone": "Africa",
    "Somali": "Africa", "South African": "Africa", "Sudanese": "Africa", "Swazi": "Africa",
    "Tanzanian": "Africa", "Togolese": "Africa", "Tunisian": "Africa", "Ugandan": "Africa",
    "Zambian": "Africa", "Zimbabwean": "Africa",
    # North America
    "American": "North America", "Canadian": "North America", "Mexican": "North America",
    "American Samoan": "North America", "Guamanian": "North America", "Puerto Rican": "North America",
    # Caribbean & Central America
    "Antiguan": "Caribbean & Central America", "Barbados": "Caribbean & Central America",
    "Belizean": "Caribbean & Central America", "Costa Rican": "Caribbean & Central America",
    "Cuban": "Caribbean & Central America", "Dominican": "Caribbean & Central America",
    "Dominican (republic)": "Caribbean & Central America", "Grenadian": "Caribbean & Central America",
    "Guatemalan": "Caribbean & Central America", "Guadeloupian": "Caribbean & Central America",
    "Haitian": "Caribbean & Central America", "Honduran": "Caribbean & Central America",
    "Jamaican": "Caribbean & Central America", "Kittian & Nevisian": "Caribbean & Central America",
    "Nicaraguan": "Caribbean & Central America", "Panamanian": "Caribbean & Central America",
    "Salvadoran": "Caribbean & Central America", "St. Lucia": "Caribbean & Central America",
    "St. Vincentian": "Caribbean & Central America", "Trinidadian & Tobagonian": "Caribbean & Central America",
    "Aruban": "Caribbean & Central America", "Netherlands Antil.": "Caribbean & Central America",
    # South America
    "Argentinian": "South America", "Bolivian": "South America", "Brazilian": "South America",
    "Chilean": "South America", "Colombian": "South America", "Ecuadorian": "South America",
    "French Guianese": "South America", "Guyanese": "South America", "Paraguayan": "South America",
    "Peruvian": "South America", "Surinamer": "South America", "Uruguayan": "South America",
    "Venezuelan": "South America",
    # Oceania
    "Australian": "Oceania", "Fijian": "Oceania", "Kiribati": "Oceania",
    "Marshallese": "Oceania", "Micronesian": "Oceania", "Nauruan": "Oceania",
    "New Zealander": "Oceania", "Ni-vanuatu": "Oceania", "Niuean": "Oceania",
    "Palauan": "Oceania", "Papua New Guinean": "Oceania", "Samoan": "Oceania",
    "Solomon Islander": "Oceania", "Tokelauan": "Oceania", "Tongan": "Oceania",
    "Tuvalu": "Oceania", "French Polynesian": "Oceania", "New Caledonia": "Oceania",
    "Pacific Is Trust T": "Oceania",
    # Unknown / Stateless
    "N.A.": "Unknown", "Others": "Unknown", "Unknown": "Unknown",
    "Stateless": "Stateless", "Nationality": "Unknown",
}

OCCUPATION_SECTOR_MAP = {
    # Management
    "Administrative & Commercial Managers": "Management",
    "Production and Specialised Services Managers": "Management",
    "Hospitality, Retail & Related Services Managers": "Management",
    "Legislators, Senior Officials & Chief Executives": "Management",
    # Professional
    "Science & Engineering Professionals": "Professional",
    "Information & Communications Technology Professionals": "Professional",
    "Health Professionals": "Professional",
    "Teaching & Training Professionals": "Professional",
    "Business & Administration Professionals": "Professional",
    "Legal, Social, Religious & Cultural Professionals": "Professional",
    # Associate Professional & Technician
    "Business & Administration Associate Professionals": "Associate Professional",
    "Physical & Engineering Science Associate Professionals": "Associate Professional",
    "Information & Communications Technicians": "Associate Professional",
    "Health Associate Professionals": "Associate Professional",
    "Teaching Associate Professionals": "Associate Professional",
    "Legal, Social, Cultural & Related Associate Professionals": "Associate Professional",
    "Other Associate Professionals Not Elsewhere Classified": "Associate Professional",
    # Clerical & Administrative
    "General & Keyboard Clerks": "Clerical",
    "Customer Services Officers & Clerks": "Clerical",
    "Numerical & Material-Recording Clerks": "Clerical",
    "Other Clerical Support Workers": "Clerical",
    "Clerical Supervisors": "Clerical",
    # Service & Sales
    "Sales Workers": "Service & Sales",
    "Personal Service Workers": "Service & Sales",
    "Protective Services Workers": "Service & Sales",
    "Personal Care Workers": "Service & Sales",
    # Trades & Craft
    "Building & Related Trades Workers, Excluding Electricians": "Trades & Craft",
    "Electrical And Electronic Trades Workers": "Trades & Craft",
    "Metal, Machinery & Related Trades Workers": "Trades & Craft",
    "Food Processing, Woodworking, Garment, Leather & Other Craft & Related Trades Workers": "Trades & Craft",
    "Precision, Handicraft, Printing & Related Trades Workers": "Trades & Craft",
    # Plant & Machine Operators
    "Drivers & Mobile Machinery Operators": "Operators",
    "Stationary Plant & Machine Operators": "Operators",
    "Assemblers & Quality Checkers": "Operators",
    # Elementary
    "Cleaners & Related Workers": "Elementary",
    "Labourers & Related Workers": "Elementary",
    "Food Preparation & Kitchen Assistants": "Elementary",
    "Waste Collection, Recycling & Material Recovery Workers & Other Elementary Workers": "Elementary",
    "Agricultural & Fishery Workers": "Elementary",
    "Agricultural, Fishery & Related Labourers": "Elementary",
    # Unemployed
    "Unemployed": "Unemployed",
}

SKILL_LEVEL_MAP = {
    "Management": 4,
    "Professional": 4,
    "Associate Professional": 3,
    "Clerical": 2,
    "Service & Sales": 2,
    "Trades & Craft": 2,
    "Operators": 2,
    "Elementary": 1,
}


def enrich_poi(df_poi, df_employer, df_industry):
    """Enrich POI data with employer, industry, and derived features.

    Pipeline:
        1. Standardise POI column names (Title Case → snake_case)
        2. Left join POI → Employer (on uen_identifier = uen)
        3. Left join Result → Industry (on primary_ssic_code = ssic)
        4. Add flag columns (has_employer, has_industry, is_unemployed)
        5. Feature engineering: age, company_age, poi_per_employer,
           age_group, company_age_group, postal_district, full_address,
           nationality_region, occupation_sector, occupation_skill_level

    Args:
        df_poi: pandas DataFrame from silver.poi
        df_employer: pandas DataFrame from silver.employer
        df_industry: pandas DataFrame from silver.industry

    Returns:
        Enriched pandas DataFrame ready for gold layer
    """
    df_poi = df_poi.copy()

    # ── Standardise column names ──
    df_poi.columns = df_poi.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # ── Enrichment joins ──
    df = (
        df_poi
        .merge(df_employer, left_on="uen_identifier", right_on="uen", how="left", suffixes=("", "_employer"))
        .merge(df_industry, left_on="primary_ssic_code", right_on="ssic", how="left", suffixes=("", "_industry"))
    )
    df.drop(columns=["uen", "ssic"], inplace=True, errors="ignore")

    # ── Flag columns ──
    df["has_employer"] = df["entity_name"].notna()
    df["has_industry"] = df["section"].notna()
    df["is_unemployed"] = df["occupation"].str.lower() == "unemployed"

    # ── 1. Parse dates ──
    df["dob_parsed"] = pd.to_datetime(df["dob"], format="%d/%m/%Y", errors="coerce")
    df["reg_date_parsed"] = pd.to_datetime(
        df["registration_incorporation_date"], format="%Y-%m-%d", errors="coerce"
    )

    # ── 2. Age (relative to year_of_profile) ──
    df["age"] = df["year_of_profile"] - df["dob_parsed"].dt.year
    df.loc[df["age"] < 0, "age"] = np.nan

    # ── 3. Company age ──
    df["company_age"] = df["year_of_profile"] - df["reg_date_parsed"].dt.year
    df.loc[df["company_age"] < 0, "company_age"] = np.nan

    # ── 4. POI count per employer ──
    poi_counts = (
        df[df["has_employer"]]
        .groupby("uen_identifier")["nric"]
        .transform("count")
    )
    df["poi_per_employer"] = np.nan
    df.loc[df["has_employer"], "poi_per_employer"] = poi_counts.values

    # ── 5. Cast float columns to nullable Int64 ──
    for col in ["id", "primary_ssic_code", "no_of_officers",
                "division_code", "group_code", "class_code", "sub_class_code"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    # ── 6. Age bins ──
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 25, 35, 45, 55, 65, 100],
        labels=["<18", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        right=True,
    )

    # ── 7. Company age bins ──
    df["company_age_group"] = pd.cut(
        df["company_age"],
        bins=[0, 5, 10, 20, 30, 50, 150],
        labels=["0-5 (Startup)", "6-10 (Growth)", "11-20 (Established)",
                "21-30 (Mature)", "31-50 (Legacy)", "50+ (Heritage)"],
        right=True,
    )

    # ── 8. Postal district ──
    df["postal_district"] = (
        df["postal_code"].astype(str).str.zfill(6).str[:2].map(DISTRICT_MAP)
    )

    # ── 9. Full address (vectorised) ──
    has_data = df["has_employer"]
    addr_block = ("Blk " + df["block"].astype(str)).where(df["block"].notna(), "")
    addr_street = df["street_name"].fillna("")
    addr_unit = (
        "#" + df["level_no"].astype(str) + "-" + df["unit_no"].astype(str)
    ).where(df["level_no"].notna() & df["unit_no"].notna(), "")
    addr_building = df["building_name"].fillna("")
    addr_postal = ("S(" + df["postal_code"].astype(str) + ")").where(df["postal_code"].notna(), "")

    addr_parts = pd.DataFrame({
        "blk": addr_block, "street": addr_street, "unit": addr_unit,
        "bldg": addr_building, "postal": addr_postal,
    })
    df["full_address"] = (
        addr_parts
        .apply(lambda row: ", ".join(p for p in row if p), axis=1)
        .where(has_data, None)
    )

    # ── 10. Nationality region ──
    df["nationality_region"] = df["nationality"].map(REGION_MAP).fillna("Unknown")

    # ── 11. Occupation sector ──
    df["occupation_sector"] = df["occupation"].map(OCCUPATION_SECTOR_MAP).fillna("Unknown")

    # ── 12. Occupation skill level ──
    df["occupation_skill_level"] = (
        df["occupation_sector"].map(SKILL_LEVEL_MAP).astype("Int64")
    )

    print(f"  Enriched: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Employer matched: {df['has_employer'].sum()} / {len(df)}")
    print(f"  Industry matched: {df['has_industry'].sum()} / {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════
# 4. PREPARE FOR SPARK
# ══════════════════════════════════════════════════════════════════════

def prepare_for_spark(df, drop_parsed_dates=True):
    """Prepare a pandas DataFrame for spark.createDataFrame().

    Handles:
        - Category columns → string (with 'nan' → None)
        - Datetime columns → string '%Y-%m-%d' (with NaT → None)
        - Optionally drops intermediate parsed date columns

    Returns:
        Cleaned pandas DataFrame ready for Spark conversion
    """
    df = df.copy()
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str).replace("nan", None)
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d").where(df[col].notna())
    if drop_parsed_dates:
        df = df.drop(columns=["dob_parsed", "reg_date_parsed"], errors="ignore")
    return df
