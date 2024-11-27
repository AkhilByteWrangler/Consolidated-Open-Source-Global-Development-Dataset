# pandas: The one library that's always there for you, even when your GPA isn't.
import pandas as pd

# requests: Because sometimes you just need to borrow data from the internet... politely.
import requests

# BeautifulSoup: Not food, but a tool to dig through messy HTML like it’s your dorm room.
from bs4 import BeautifulSoup

# warnings: For all those times Python says, "Are you sure about this?" We're like, "Yes. Quiet, please."
import warnings 
warnings.filterwarnings('ignore')  # Like muting that one group chat during finals.


# File paths: AKA the syllabus for this data project. Let's try to stay organized... for once.
FILE_PATHS = {
    "whr": "data/world_happiness_report.xls",  # Because we all want to know who's acing happiness.
    "tedi": "data/TheEconomistDemocracyIndex.xlsx",  # Democracy, ranked. Like your GPA, but for countries.
    "energy": "data/energy.csv",  # Renewable, non-renewable, and that one time you tried to "renew" your life energy.
    "food": "data/food_supply.csv",  # Fuel for both humans and datasets.
    "deaths": "data/deaths-in-armed-conflicts-based-on-where-they-occurred.csv",  # Dark, but necessary data.
    "air_pollution": "data/long-run-air-pollution.csv",  # Breathing isn’t free after all.
    "hdi": "data/human-development-index.csv",  # Because it’s not just about grades; it’s about growth (or so they say).
    "rule_of_law": "data/rule-of-law-index.csv",  # Order in the court! Or in the dataset, at least.
    "median_age": "data/median-age.csv",  # Who’s aging gracefully? Spoiler: Not your finals week self.
    "urban_population": "data/share-urban-and-rural-population.csv",  # City life vs. the countryside – stats edition.
    "tax_revenue": "data/tax-revenues-as-a-share-of-gdp-unu-wider.csv",  # Paying taxes is inevitable. Knowing about them is optional.
}

# Function to load data. Think of it as assembling your study notes before a big exam (No ChatGPT, in our days, Chegg maybe!).
def load_data():
    """Load all datasets into a dictionary of DataFrames."""
    data = {
        "whr": pd.read_excel(FILE_PATHS["whr"]),  
        "tedi": pd.read_excel(FILE_PATHS["tedi"]),  
        "energy": pd.read_csv(FILE_PATHS["energy"]), 
        "food": pd.read_csv(FILE_PATHS["food"]),  
        "deaths": pd.read_csv(FILE_PATHS["deaths"]),
        "air_pollution": pd.read_csv(FILE_PATHS["air_pollution"]), 
        "hdi": pd.read_csv(FILE_PATHS["hdi"]), 
        "rule_of_law": pd.read_csv(FILE_PATHS["rule_of_law"]),  
        "median_age": pd.read_csv(FILE_PATHS["median_age"]), 
        "urban_population": pd.read_csv(FILE_PATHS["urban_population"]),
        "tax_revenue": pd.read_csv(FILE_PATHS["tax_revenue"]), 
    }
    return data  

def process_whr_dataset(whr):
    """Process the World Happiness Report dataset."""
    # Rename columns for consistency
    whr.rename(columns={'Country name': 'Country', 'year': 'Year'}, inplace=True)
    
    # Remove non-alphanumeric characters from 'Country' column
    whr['Country'] = whr['Country'].str.replace(r'[^\w\s]', '', regex=True)
    
    # Strip any leading/trailing spaces after cleaning
    whr['Country'] = whr['Country'].str.strip()
    
    # Return the processed WHR DataFrame
    return whr

def process_tedi_dataset(tedi):
    """Transform and clean the Economist Democracy Index dataset."""
    # Reshape the data to be more analysis-friendly. Long format, strong format.
    tedi_long = tedi.melt(
        id_vars=["Country", "Regime type"], var_name="Year", value_name="Democracy_Index"
    )
    tedi_long["Year"] = tedi_long["Year"].astype(int)
    tedi_long = tedi_long[tedi_long["Year"] != 2006]  # 2006 didn’t make the cut.

    # Clean up country names and get everything in order.
    tedi_long['Country'] = tedi_long['Country'].str.replace(r'[^\w\s]', '', regex=True).str.strip()
    tedi_long.sort_values(by=['Country', 'Year'], inplace=True)
    return tedi_long  # Democracy, prepped for data justice.

def process_energy_dataset(energy_df):
    """
    Process and clean the energy dataset by combining key metrics
    and removing unnecessary columns. Let's make this dataset as
    efficient as renewable energy should be.
    """
    # Combine various energy consumption and production metrics.
    # Because who needs a hundred columns when you can just sum it up?
    energy_df['renewables_consumption'] = (
        energy_df['biofuel_consumption'] + energy_df['hydro_consumption'] +
        energy_df['solar_consumption'] + energy_df['wind_consumption'] +
        energy_df['other_renewable_consumption']
    )
    energy_df['non_renewables_consumption'] = (
        energy_df['coal_consumption'] + energy_df['gas_consumption'] +
        energy_df['oil_consumption'] + energy_df['nuclear_consumption']
    )
    energy_df['renewables_production'] = (
        energy_df['biofuel_electricity'] + energy_df['hydro_electricity'] +
        energy_df['solar_electricity'] + energy_df['wind_electricity'] +
        energy_df['other_renewable_electricity']
    )
    energy_df['non_renewables_production'] = (
        energy_df['coal_electricity'] + energy_df['gas_electricity'] +
        energy_df['oil_electricity'] + energy_df['nuclear_electricity']
    )

    # Add total consumption and production for good measure – more data, more fun.
    energy_df['total_consumption'] = (
        energy_df['renewables_consumption'] + energy_df['non_renewables_consumption']
    )
    energy_df['total_production'] = (
        energy_df['renewables_production'] + energy_df['non_renewables_production']
    )

   # Drop the original columns that have been combined, along with other unused columns
    columns_to_drop = [
        'biofuel_consumption', 'coal_consumption', 'gas_consumption', 'nuclear_consumption',
        'oil_consumption', 'hydro_consumption', 'renewables_consumption', 'other_renewable_consumption',
        'biofuel_electricity', 'coal_electricity', 'gas_electricity', 'nuclear_electricity',
        'oil_electricity', 'hydro_electricity', 'renewables_electricity', 'other_renewable_electricity',
        
        'iso_code', 'population', 'gdp', 'biofuel_cons_change_pct', 'biofuel_cons_change_twh', 
        'biofuel_cons_per_capita', 'biofuel_share_elec', 'biofuel_share_energy', 'carbon_intensity_elec',
        'coal_cons_change_pct', 'coal_cons_change_twh', 'coal_cons_per_capita', 'coal_prod_change_pct', 
        'coal_prod_change_twh', 'coal_prod_per_capita', 'coal_share_elec', 'coal_share_energy', 
        'electricity_demand', 'electricity_generation', 'electricity_share_energy', 'energy_cons_change_pct',
        'energy_cons_change_twh', 'energy_per_capita', 'energy_per_gdp', 'fossil_cons_change_pct', 
        'fossil_cons_change_twh', 'fossil_elec_per_capita', 'fossil_electricity', 'fossil_energy_per_capita', 
        'fossil_fuel_consumption', 'fossil_share_elec', 'fossil_share_energy', 'gas_cons_change_pct', 
        'gas_cons_change_twh', 'gas_consumption', 'gas_elec_per_capita', 'gas_electricity', 
        'gas_energy_per_capita', 'gas_prod_change_pct', 'gas_prod_change_twh', 'gas_prod_per_capita', 
        'gas_share_elec', 'gas_share_energy', 'greenhouse_gas_emissions', 'hydro_cons_change_pct', 
        'hydro_cons_change_twh', 'hydro_consumption', 'hydro_elec_per_capita', 'hydro_energy_per_capita', 
        'hydro_share_elec', 'hydro_share_energy', 'low_carbon_cons_change_pct', 'low_carbon_cons_change_twh',
        'low_carbon_consumption', 'low_carbon_elec_per_capita', 'low_carbon_energy_per_capita', 
        'low_carbon_share_elec', 'low_carbon_share_energy', 'net_elec_imports', 'net_elec_imports_share_demand', 
        'nuclear_cons_change_pct', 'nuclear_cons_change_twh', 'nuclear_consumption', 'nuclear_elec_per_capita',
        'nuclear_energy_per_capita', 'nuclear_share_elec', 'nuclear_share_energy', 'oil_cons_change_pct', 
        'oil_cons_change_twh', 'oil_consumption', 'oil_elec_per_capita', 'oil_energy_per_capita', 
        'oil_prod_change_pct', 'oil_prod_change_twh', 'oil_prod_per_capita', 'oil_share_elec', 
        'oil_share_energy', 'other_renewable_consumption', 'other_renewable_electricity', 
        'other_renewable_exc_biofuel_electricity', 'other_renewables_cons_change_pct', 
        'other_renewables_cons_change_twh', 'other_renewables_elec_per_capita', 
        'other_renewables_share_elec', 'other_renewables_share_energy', 'per_capita_electricity', 
        'primary_energy_consumption', 'renewables_cons_change_pct', 'renewables_cons_change_twh', 
        'renewables_consumption', 'renewables_elec_per_capita', 'renewables_energy_per_capita', 
        'renewables_share_elec', 'renewables_share_energy', 'solar_cons_change_pct', 'solar_cons_change_twh', 
        'solar_consumption', 'solar_elec_per_capita', 'solar_energy_per_capita', 'solar_share_elec', 
        'solar_share_energy', 'wind_cons_change_pct', 'wind_cons_change_twh', 'wind_consumption', 
        'wind_elec_per_capita', 'wind_energy_per_capita', 'wind_share_elec', 'wind_share_energy', 'solar_electricity', 
        'wind_electricity', 'non_renewables_consumption', 'total_consumption'
    ]

    # Dropping unnecessary columns because we don’t need extra baggage in life or datasets.
    energy_df.drop(columns=columns_to_drop, inplace=True)

    # Clean up country names and year column – no weird characters allowed.
    energy_df['Country'] = energy_df['country'].str.replace(r'[^\w\s]', '', regex=True)
    energy_df['Year'] = energy_df['year'].astype(int)

    # Drop redundant columns and get everything sorted for analysis.
    energy_df.drop(columns=['country', 'year'], inplace=True)
    energy_df.sort_values(by=['Country', 'Year'], inplace=True)

    return energy_df  # Energy data, streamlined and ready to go!

# Functions to process individual datasets. These are the "cleaning crew" 
# making sure every dataset is neat, organized, and analysis-ready.

def process_food_dataset(food):
    """Select only relevant columns from the food dataset."""
    return food[['Country', 'Year', 
                 'Food supply (kcal per capita per day)',
                 'Food supply (Protein g per capita per day)',
                 'Food supply (Fat g per capita per day)']]

def process_deaths_dataset(deaths):
    """Clean up and rename columns for the deaths dataset."""
    deaths = deaths[['Entity', 'Year', 
                     'Deaths in ongoing conflicts in a country (best estimate) - Conflict type: all']]
    deaths.rename(columns={'Entity': 'Country', 
                           'Deaths in ongoing conflicts in a country (best estimate) - Conflict type: all': 'Deaths'}, 
                  inplace=True)
    return deaths

def process_air_pollution_dataset(air_pollution):
    """Calculate total emissions and keep only the essentials."""
    air_pollution['Total_Emissions'] = (
        air_pollution['Nitrogen oxide (NOx)'] +
        air_pollution['Sulphur dioxide (SO₂) emissions'] +
        air_pollution['Carbon monoxide (CO) emissions'] +
        air_pollution['Black carbon (BC) emissions'] +
        air_pollution['Ammonia (NH₃) emissions'] +
        air_pollution['Non-methane volatile organic compounds (NMVOC) emissions']
    )
    air_pollution = air_pollution[['Entity', 'Year', 'Total_Emissions']]
    air_pollution.rename(columns={'Entity': 'Country'}, inplace=True)
    return air_pollution

def process_hdi_dataset(hdi):
    """Simplify the Human Development Index dataset."""
    hdi = hdi[['Entity', 'Year', 'Human Development Index']]
    hdi.rename(columns={'Entity': 'Country'}, inplace=True)
    return hdi

def process_rule_of_law_dataset(rule_of_law):
    """Clean and rename columns for the rule of law dataset."""
    rule_of_law = rule_of_law[['Entity', 'Year', 
                               'Rule of Law index (best estimate, aggregate: average)']]
    rule_of_law.rename(columns={'Entity': 'Country', 
                                'Rule of Law index (best estimate, aggregate: average)': 'Rule_of_Law_Index'}, 
                       inplace=True)
    return rule_of_law

def process_median_age_dataset(median_age):
    """Keep key columns from the median age dataset."""
    median_age = median_age[['Entity', 'Year', 
                             'Median age - Sex: all - Age: all - Variant: estimates']]
    median_age.rename(columns={'Entity': 'Country', 
                               'Median age - Sex: all - Age: all - Variant: estimates': 'Median Age'}, 
                      inplace=True)
    return median_age

def process_urban_population_dataset(urban_population):
    """Extract urban population percentages and clean column names."""
    urban_population = urban_population[['Entity', 'Year', 
                                         'Urban population (% of total population)']]
    urban_population.rename(columns={'Entity': 'Country', 
                                     'Urban population (% of total population)': 'Urban Population (%)'}, 
                            inplace=True)
    return urban_population

def process_tax_revenue_dataset(tax_revenue):
    """Clean tax revenue data by renaming and selecting the important stuff."""
    tax_revenue = tax_revenue[['Entity', 'Year', 
                               'Taxes including social contributions (as a share of GDP)']]
    tax_revenue.rename(columns={'Entity': 'Country', 
                                'Taxes including social contributions (as a share of GDP)': 'Tax_Revenue'}, 
                       inplace=True)
    return tax_revenue

# Main script logic.
def main():
    """
    Load, process, and merge all datasets into one comprehensive DataFrame.
    This is where the magic happens.
    """
    data = load_data()  # Load all raw datasets.

    # Process datasets one by one. It’s like assembling IKEA furniture but with data.
    whr = process_whr_dataset(data["whr"])
    tedi = process_tedi_dataset(data["tedi"])
    energy = process_energy_dataset(data["energy"])
    food = process_food_dataset(data["food"])
    deaths = process_deaths_dataset(data["deaths"])
    air_pollution = process_air_pollution_dataset(data["air_pollution"])
    hdi = process_hdi_dataset(data["hdi"])
    rule_of_law = process_rule_of_law_dataset(data["rule_of_law"])
    median_age = process_median_age_dataset(data["median_age"])
    urban_population = process_urban_population_dataset(data["urban_population"])
    tax_revenue = process_tax_revenue_dataset(data["tax_revenue"])

    # Merge all datasets sequentially – because teamwork makes the dataset dream work.
    merged = pd.merge(whr, tedi, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, energy, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, food, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, deaths, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, air_pollution, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, hdi, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, rule_of_law, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, median_age, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, urban_population, on=["Country", "Year"], how="inner")
    merged = pd.merge(merged, tax_revenue, on=["Country", "Year"], how="inner")

    # Drop leftover unnecessary columns (if any).
    merged.drop(columns=['other_renewables_elec_per_capita_exc_biofuel',
                         'other_renewables_energy_per_capita',
                         'other_renewables_share_elec_exc_biofuel'], 
                inplace=True)

    return merged  # The final, all-star dataset.

if __name__ == "__main__":
    # Save the final dataset and admire your data wizardry.
    final_dataset = main()
    final_dataset.to_csv("Money_vs_Happiness_dataset.csv", index=False)
    print("Final dataset saved to 'Money_vs_Happiness_dataset.csv'")
    print(final_dataset)


