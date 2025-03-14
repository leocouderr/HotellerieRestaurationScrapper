from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import unicodedata
import re
import requests
import asyncio
import httpx
import nest_asyncio  # Allows running async code in Jupyter

# Set up Selenium options (headless mode for efficiency)
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# List of categories to scrape
categories = [
    "cuisine",
    "salle-bar-cafe-room-service",
    "reception-reservation",
    "service-etage-housekeeping",
    "direction",
    "restauration-rapide",
    "restauration-collective",
]

base_url = "https://www.lhotellerie-restauration.fr/emplois/"
max_pages = 5
job_urls = []

for category in categories:
    for page in range(1, max_pages + 1):
        url = f"{base_url}{category}?Page={page}"
        driver.get(url)  # Open the page
        
        # Wait for job items to be present (Optional: You can add WebDriverWait if needed)
        job_links = driver.find_elements(By.CSS_SELECTOR, "a.list-group-item.job-item")
        
        for link in job_links:
            href = link.get_attribute("href")
            if href and "/emploi/" in href:
                job_urls.append(href)

driver.quit()  # Close the browser when done

print(f"Collected {len(job_urls)} job URLs")

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# Set up Selenium options (headless mode for efficiency)
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Initialize an empty list to store job data
job_data = []

def accepter_cookies():
    """Vérifie et accepte les cookies si le bouton est présent."""
    try:
        bouton_cookies = driver.find_element(By.ID, "popin_tc_privacy_button")
        bouton_cookies.click()
        print("Popup des cookies accepté.")
    except NoSuchElementException:
        print("Aucun popup de cookies trouvé.")

for job_url in job_urls:
    driver.get(job_url)
    accepter_cookies()

    def get_text(selector, multiple=False):
        """Helper function to extract text from an element."""
        try:
            if multiple:
                return [elem.text.strip() for elem in driver.find_elements(By.CSS_SELECTOR, selector)]
            return driver.find_element(By.CSS_SELECTOR, selector).text.strip()
        except NoSuchElementException:
            return "" if not multiple else []

    # Extract job details
    title = get_text("h1.text-primary.job-offer-occupation-list")
    location = get_text("p.m-0")
    date = get_text("small.job-offer-date")
    
    # Find all <p class="m-0"> elements to distinguish between location & contract type
    p_elements = driver.find_elements(By.CSS_SELECTOR, "p.m-0")
    contract_type = p_elements[1].text.strip() if len(p_elements) > 1 else ""

    tags = get_text("div.job-tags.mt-2 p", multiple=True)
    description = get_text("div.row.py-3 p.keep-format")

    # Append extracted data to list
    job_data.append({
        "Title": title,
        "Location": location,
        "Date": date,
        "Contract Type": contract_type,
        "Tags": tags,
        "Description": description,
        "URL": job_url
    })

# Convert list to Pandas DataFrame
df_jobs = pd.DataFrame(job_data)
print("Debug - First 5 descriptions:")
print(df_jobs.Description)

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import json
import os

# Google Sheets API setup
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

credentials_info = json.loads(os.environ.get("GOOGLE_CREDENTIALS"))
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_info, scope)
client = gspread.authorize(credentials)

# Open the Google Sheet
spreadsheet = client.open('HotellerieRestaurationListings')  # Use your sheet's name
worksheet = spreadsheet.sheet1

# Read existing data from Google Sheets into a DataFrame
existing_data = pd.DataFrame(worksheet.get_all_records())

# Convert scraped results into a DataFrame
new_data = df_jobs

#Ajouter all CAPS et "FRANCE" à la localisation + clean autres colonnes
new_data.Location = (new_data.Location + ', FRANCE').str.upper()
new_data = new_data[new_data["Title"].notna() & (new_data["Title"].str.strip() != "")]
new_data["Tags"] = new_data["Tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
new_data["Tags"] = new_data["Tags"].str.replace(r"[\[\]']", "", regex=True)
new_data = new_data[new_data["Location"].str.match(r"^\d", na=False)]
new_data['Date'] = pd.to_datetime(new_data['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# Apply nest_asyncio to fix event loop issue in Jupyter
#nest_asyncio.apply()

# Data Gouv API URL
API_URL = "https://api-adresse.data.gouv.fr/search"

# Function to call API asynchronously with retries
async def get_geodata(client, address, retries=3):
    params = {"q": address, "limit": 1}

    for attempt in range(retries):
        try:
            response = await client.get(API_URL, params=params, timeout=5)

            if response.status_code == 503:  # Server overloaded
                print(f"503 Error - Retrying {address} (Attempt {attempt+1})...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

            response.raise_for_status()  # Raise error if response is bad
            data = response.json()

            if data["features"]:
                props = data["features"][0]["properties"]
                geo = data["features"][0]["geometry"]["coordinates"]

                ville = props.get("city", "")
                code_postal = props.get("postcode", "")
                longitude = geo[0] if geo else None
                latitude = geo[1] if geo else None
                contexte = props.get("context", "")

                # Extract region name (after second comma)
                region = contexte.split(", ")[-1] if contexte.count(",") >= 2 else ""

                return ville, code_postal, longitude, latitude, region
        
        except Exception as e:
            print(f"Error fetching data for {address} (Attempt {attempt+1}): {e}")
        
        await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

    return None, None, None, None, None  # Return empty values if all retries fail

# Async function to process all addresses with rate limiting
async def process_addresses(address_list, delay_between_requests=0.017):  # 1/60 = ~0.017s
    results = []
    async with httpx.AsyncClient() as client:
        for i, address in enumerate(address_list):
            result = await get_geodata(client, address)
            results.append(result)
            
            print(f"Processed {i + 1} / {len(address_list)}")

            # Respect 60 requests per second limit
            await asyncio.sleep(delay_between_requests)  

    return results

# Run API calls asynchronously
addresses = new_data["Location"].tolist()
geodata_results = asyncio.run(process_addresses(addresses))

# Assign the results to the DataFrame
new_data[["Ville", "Code Postal", "Longitude", "Latitude", "Region"]] = pd.DataFrame(geodata_results)

# Add "France Travail" column
new_data["Source"] = "Hotellerie Restauration"

# Combine and remove duplicates
if not existing_data.empty:
    combined_data = pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(
        subset=['URL']
    )
else:
    combined_data = new_data

# Debug: Print the number of rows to append
rows_to_append = new_data.shape[0]
print(f"Rows to append: {rows_to_append}")

# Handle NaN, infinity values before sending to Google Sheets
# Replace NaN values with 0 or another placeholder (you can customize this)
combined_data = combined_data.fillna(0)

# Replace infinite values with 0 or another placeholder
combined_data.replace([float('inf'), float('-inf')], 0, inplace=True)

# Optional: Ensure all float types are valid (e.g., replace any invalid float with 0)
combined_data = combined_data.applymap(lambda x: 0 if isinstance(x, float) and (x == float('inf') or x == float('-inf') or x != x) else x)

# Optional: Ensuring no invalid values (like lists or dicts) in any column
def clean_value(value):
    if isinstance(value, (list, dict)):
        return str(value)  # Convert lists or dicts to string
    return value

combined_data = combined_data.applymap(clean_value)

#add column titre de annonce sans accents ni special characters
def remove_accents_and_special(text):
    # Normalize the text to separate characters from their accents.
    normalized = unicodedata.normalize('NFD', text)
    # Remove the combining diacritical marks.
    without_accents = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Remove special characters (retain letters, digits and whitespace).
    cleaned = re.sub(r'[^A-Za-z0-9\s]', '', without_accents)
    return cleaned

# Create the new column "Titre annonce sans accent" by applying the function on "intitule".
combined_data["TitreAnnonceSansAccents"] = combined_data["Title"].apply(
    lambda x: remove_accents_and_special(x) if isinstance(x, str) else x
)

# Update Google Sheets with the combined data
worksheet.clear()  # Clear existing content
worksheet.update([combined_data.columns.tolist()] + combined_data.values.tolist())

print("New rows successfully appended to Google Sheets without duplicates!")




