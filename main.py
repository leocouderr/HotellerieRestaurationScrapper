from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
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

# List of categories to scrape
categories = [
   # "cuisine",
   # "salle-bar-cafe-room-service",
   # "reception-reservation",
   # "service-etage-housekeeping",
   # "direction",
   # "restauration-rapide",
    "restauration-collective",
]

base_url = "https://www.lhotellerie-restauration.fr/emplois/"
max_pages = 1
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

# Update Google Sheets with the combined data
worksheet.clear()  # Clear existing content
worksheet.update([combined_data.columns.tolist()] + combined_data.values.tolist())

print("New rows successfully appended to Google Sheets without duplicates!")




