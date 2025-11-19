import time
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime
from selenium import webdriver
from time import sleep
import random
import os
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import unicodedata
import re
import asyncio
import httpx
import nest_asyncio
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import json
from selenium.webdriver.chrome.options import Options
import os
import requests
from openai import OpenAI
import instructor
from enum import Enum
from pydantic import BaseModel, Field, field_validator, constr, ValidationError, model_validator
from typing import List, Optional, Any



# Set up Selenium options (headless mode for efficiency)
options = Options()
options.add_argument("--headless=new")  # modern headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--remote-debugging-port=9222")  # helps Chrome bind properly in CI
options.add_argument("--window-size=1920,1080")

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
max_pages = 3
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
        #print("Popup des cookies accepté.")
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
#new_data.Location = (new_data.Location + ', FRANCE').str.upper()
new_data = new_data[new_data["Title"].notna() & (new_data["Title"].str.strip() != "")]
new_data["Tags"] = new_data["Tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
new_data["Tags"] = new_data["Tags"].str.replace(r"[\[\]']", "", regex=True)
#new_data = new_data[new_data["Location"].str.match(r"^\d", na=False)]
new_data['Date'] = pd.to_datetime(new_data['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

print(f"Check {new_data.URL}")

# Data Gouv API URL
API_URL = "https://api-adresse.data.gouv.fr/search"

# Function to call API synchronously with retries
def get_geodata(address, retries=3):
    params = {"q": address, "limit": 1}

    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=5)

            if response.status_code == 503:  # Server overloaded
                print(f"503 Error - Retrying {address} (Attempt {attempt+1})...")
                time.sleep(2 ** attempt)  # Exponential backoff
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
        
        time.sleep(2 ** attempt)  # Exponential backoff for retries

    return None, None, None, None, None  # Return empty values if all retries fail

def clean_address(address):
    if pd.isna(address):
        return ""

    # Supprimer les codes comme "13 - " ou "75A - "
    address = re.sub(r"^[\dA-Z]{2,3}\s*-\s*", "", address)

    # Supprimer les codes postaux en fin (ex: "75001" ou "07")
    address = re.sub(r"\d{2,5}$", "", address)

    # Remplacer "St", "ST.", "St." par "Saint"
    address = re.sub(r"\bSt[\.]?\b", "Saint", address, flags=re.IGNORECASE)

    # Nettoyage final : strip, espaces doubles → simples, capitalisation
    address = address.strip()
    address = re.sub(r"\s{2,}", " ", address)
    address = address.title()

    return address


# Load addresses from DataFrame
addresses = new_data["Location"].apply(clean_address).tolist()

# Process all addresses synchronously
results = []

for i, address in enumerate(addresses):
    try:
        print(f"Processing {i + 1}/{len(addresses)}: '{address}'")

        if pd.isna(address) or not address.strip():
            results.append((None, None, None, None, None))
            continue

        result = get_geodata(address)

        if result is None or len(result) != 5:
            print(f"→ Résultat invalide pour '{address}'")
            results.append((None, None, None, None, None))
        else:
            results.append(result)

        time.sleep(0.02)
    
    except Exception as e:
        print(f"⚠️ Erreur fatale à la ligne {i} pour l’adresse '{address}' : {e}")
        results.append((None, None, None, None, None))

# Assign the results to the DataFrame
new_data[["Ville", "Code Postal", "Longitude", "Latitude", "Region"]] = pd.DataFrame(results)

# Add "France Travail" column
new_data["Source"] = "Hotellerie Restauration"

print(f"Post geo new data Check url {new_data.URL}")
print(f"Post geo new data Check length {len(new_data)}")
print(f"Post geo Check existing length {len(existing_data)}")

# -------- DEBUT CHATGPT DATA ENRICHMENT --------------------------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_ai = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))

class Loge(str, Enum):
    LOGE = "Logé"
    NON_LOGE = "Non Logé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class TypeContrat(str, Enum):
    CDD = "CDD"
    CDI = "CDI"
    STAGE = "Stage"
    APPRENTISSAGE = "Apprentissage"
    INTERIM = "Interim"
    EXTRA = "Extra"
    SAISONNIER = "Saisonnier"
    ALTERNANCE = "Alternance"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CoupleAccepte(str, Enum):
    ACCEPTE = "Couple accepté"
    NON_ACCEPTE = "Couple non accepté"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieEtablissement(str, Enum):
    GASTRONOMIQUE = "Gastronomique"
    BRASSERIE = "Brasserie"
    BAR = "Bar"
    RAPIDE = "Restauration rapide"
    COLLECTIVE = "Restauration collective"
    RESTAURANT = "Restaurant"
    HOTEL_LUXE = "Hôtel luxe"
    HOTEL = "Hôtel"
    CAMPING = "Camping"
    CAFE = "Café/Salon de thé"
    BOULANGERIE = "Boulangerie/Patisserie"
    ETOILE = "Etoile Michelin"
    PALACE = "Palace"
    TRAITEUR = "Traiteur/Événementiel/Banquet"
    SPA = "Spa"
    LABORATOIRE = "Laboratoire"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob1(str, Enum):
    RESTAURATION = "Restauration"
    HOTELLERIE = "Hôtellerie"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob2(str, Enum):
    SALLE = "Salle & Service"
    DIRECTION = "Direction & Management"
    SUPPORT = "Support & Back-office"
    CUISINE = "Cuisine"
    SPA = "Spa & Bien-être"
    ETAGES = "Étages & Housekeeping"
    BAR = "Bar & Sommellerie"
    RECEPTION = "Réception & Hébergement"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class CategorieJob3(str, Enum):
    CHEF_EXECUTIF = "Chef exécutif"
    CHEF_CUISINE = "Chef de cuisine"
    SOUS_CHEF = "Sous-chef"
    CHEF_PARTIE = "Chef de partie"
    COMMIS_CUISINE = "Commis de cuisine"
    PATISSIER = "Pâtissier"
    BOULANGER = "Boulanger"
    PIZZAIOLO = "Pizzaiolo"
    TRAITEUR = "Traiteur"
    MANAGER = "Manager / Responsable"
    EMPLOYE = "Employé polyvalent"
    PLONGEUR = "Plongeur"
    STEWARD = "Steward"
    DIRECTEUR = "Directeur"
    RESPONSABLE_SALLE = "Responsable de salle"
    MAITRE_HOTEL = "Maître d’hôtel"
    CHEF_RANG = "Chef de rang"
    COMMIS_SALLE = "Commis de salle / Runner"
    SERVEUR = "Serveur"
    SOMMELIER = "Sommelier"
    BARMAN = "Barman"
    BARISTA = "Barista"
    RECEPTIONNISTE = "Réceptionniste / Hôte d’accueil"
    CONCIERGE = "Concierge"
    BAGAGISTE = "Bagagiste / Voiturier"
    VALET = "Valet / Femme de chambre"
    MARKETING = "Marketing / Communication"
    AGENT_RESERVATIONS = "Agent de réservations"
    REVENUE_MANAGER = "Revenue manager"
    GOUVERNANT = "Gouvernant(e)"
    SPA_PRATICIEN = "Spa praticien(ne) / Ésthéticien(ne)"
    COACH = "Coach sportif"
    MAITRE_NAGEUR = "Maître-nageur"
    ANIMATION = "Animation / Événementiel"
    COMMERCIAL = "Commercial"
    RH = "RH / Paie"
    COMPTABILITE = "Comptabilité / Contrôle de gestion"
    TECHNICIEN = "Technicien / Maintenance"
    IT = "IT / Data"
    HACCP = "HACCP manager"
    CUISINIER = "Cuisinier"
    LIMONADIER = "Limonadier"
    ALLOTISSEUR = "Allotisseur"
    APPROVISIONNEUR = "Approvisionneur / Économe"
    AGENT_SECURITE = "Agent de sécurité"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Urgent(str, Enum):
    URGENT = "Urgent"
    NON_URGENT = "Non Urgent"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"

class Environnement(str, Enum):
    CENTRE_VILLE = "Centre ville"
    BORD_MER = "Bord de mer"
    MONTAGNE = "Montagne"
    BANLIEUE = "Banlieue"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class ChaineIndependant(str, Enum):
    CHAINE = "Chaine"
    INDEPENDANT = "Indépendant"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class TempsTravail(str, Enum):
    PLEIN_TEMPS = "Plein temps"
    TEMPS_PARTIEL = "Temps partiel"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class HorairesTravail(str, Enum):
    JOUR = "Jour"
    NUIT = "Nuit"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class Experience(str, Enum):
    DEBUTANT = "Débutant"
    CONFIRME = "Confirmé"
    VIDE = ""
    NON_SPECIFIE = "Non spécifié"


class DureeModel(BaseModel):
    value: str


class HeuresParSemaineModel(BaseModel):
    heures: Optional[int] = None

    # v2 field validator
    @field_validator("heures", mode="before")
    def parse_heures(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            match = re.search(r"\d+", v)
            if match:
                return int(match.group())
        return None

class DateDebutModel(BaseModel):
    value: str

class SalaireModel(BaseModel):
    value: str

# --- Base model that ties everything together ---
class JobClassification(BaseModel):
    IA_Logé: Loge
    IA_Type_de_contrat: TypeContrat
    IA_Salaire: SalaireModel
    IA_Couple_accepté: CoupleAccepte
    IA_Catégorie_établissement: CategorieEtablissement
    IA_Catégorie_Job_1: CategorieJob1
    IA_Catégorie_Job_2: CategorieJob2
    IA_Catégorie_Job_3: CategorieJob3
    IA_Urgent: Urgent
    IA_Date_de_début: DateDebutModel
    IA_Durée: DureeModel
    IA_Type_environnement: Environnement
    IA_Chaine_Indépendant: ChaineIndependant
    IA_Temps_de_travail: TempsTravail
    IA_Horaires_de_travail: HorairesTravail
    IA_Heures_par_semaine: HeuresParSemaineModel
    IA_Éxpérience: Experience

SYSTEM_PROMPT = """You are a classifier for job listings in the hospitality industry in France. You are an expert and absolutely have to respect the 
instructions. Each category can ONLY take one the value that are specified for it.
The success of my business depends on you so double check!!
    "IA_Logé": when accomodation or help with accomodation is provided "Logé" else "Non logé",
        "IA_Type_de_contrat": it MUST BE one of ["CDD", "CDI", "Stage", "Apprentissage", "Interim", "Extra", "Saisonnier", "Alternance"],
        "IA_Salaire": the highest salary offered in format "X€/heure" or "X€/mois" or "X€/an", or "" if not specified,
        "IA_Couple_accepté": "Couple accepté" or "",
    	"IA_Catégorie_établissement": it MUST BE one of the following and CANNOT be empty ["Gastronomique","Brasserie","Bar","Restauration rapide","Restauration collective","Restaurant","Hôtel luxe","Hôtel","Camping","Café/Salon de thé”,”Boulangerie/Patisserie”,”Etoile Michelin","Palace”, “Traiteur/Événementiel/Banquet”,“Spa”, “Laboratoire”],
    	"IA_Catégorie_Job_1":  it MUST BE one of the following and it cannot be empty [“Restauration”, “Hôtellerie”],
    	“IA_Catégorie_Job_2”:  it MUST BE one of and the most relevant, it cannot be empty [“Salle & Service”, “Direction & Management”, “Support & Back-office”, “Cuisine”, “Spa & Bien-être”, “Étages & Housekeeping”, “Bar & Sommellerie”, “Réception & Hébergement”],
        “IA_Catégorie_Job_3”: it has to be one of the following and the most relevant, it cannot be empty ["Chef exécutif","Chef de cuisine","Sous-chef","Chef de partie","Commis de cuisine","Pâtissier","Boulanger","Pizzaiolo","Traiteur","Manager / Responsable","Employé polyvalent","Plongeur","Steward","Directeur","Responsable de salle","Maître d’hôtel","Chef de rang","Commis de salle / Runner","Serveur","Sommelier","Barman","Barista","Réceptionniste / Hôte d’accueil","Concierge","Bagagiste / Voiturier","Valet / Femme de chambre","Marketing / Communication","Agent de réservations","Revenue manager","Gouvernant(e)","Spa praticien(ne) / Ésthéticien(ne)","Coach sportif","Maître-nageur","Animation / Événementiel","Commercial","RH / Paie","Comptabilité / Contrôle de gestion","Technicien / Maintenance","IT / Data","HACCP manager","Cuisinier","Limonadier","Allotisseur","Approvisionneur / Économe","Agent de sécurité"],
    	"IA_Urgent": "Urgent" or "", it takes "Urgent" only when the starting date is within 2 weeks of the date_scraping or when it is explicitly mentioned in the description
        "IA_Date_de_début": starting date in format YYYY-MM-DD if present, else "",
        "IA_Durée": contract duration like "N days", "N weeks", "N months", or "Indéfini",
        "IA_Type_environnement”: one of ["Centre ville","Bord de mer","Montagne","Banlieue"],
    	“IA_Chaine_Indépendant”: when the company posting the job listing is part of a group or bigger company "Chaine", else ""
        "IA_Temps_de_travail": "Plein temps" or "Temps partiel",
        "IA_Horaires_de_travail": "Jour" or "Nuit",
        "IA_Heures_par_semaine": return a number not a string ! the number of hours worked per week if available, when the contract is less than a week just put how many hours it , else “”,
    	“IA_Éxpérience” one the following [“Débutant”, “Confirmé”]

    Strictly output without explanations."""


def classify_job_listing(ticket_text: str) -> JobClassification:
    response = client_ai.chat.completions.create(
            model="gpt-4o-mini",
            max_retries=3,
            response_model=JobClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ticket_text}
            ],
            temperature=0
        )
    return response

# Convert each row into a single string with "col":"value" format
new_data["row_as_string"] = new_data.apply(
    lambda row: ", ".join([f'"{col}":"{row[col]}"' for col in new_data.columns]),
    axis=1
)

# Apply your classify_job_listing function to each row
result = new_data["row_as_string"].apply(classify_job_listing)

# If you want, convert the results (list of dicts) into a DataFrame
classified_df = pd.DataFrame(result.tolist())

base_model_columns = list(JobClassification.model_fields.keys())

def get_value(cell, column_name=None):
    if isinstance(cell, tuple) and len(cell) == 2:
        val = cell[1]

        # Special case for IA_Heures_par_semaine
        if column_name == "IA_Heures_par_semaine" and hasattr(val, "heures"):
            return val.heures  # directly the int

        # Other enums / objects
        if hasattr(val, "value"):
            return val.value
        return str(val)
    elif hasattr(cell, "value"):
        return cell.value
    return str(cell)

classified_df = pd.DataFrame([
    [get_value(cell, col) for cell, col in zip(row, base_model_columns)]
    for row in classified_df.values
], columns=base_model_columns)

new_data = new_data.drop(columns=["row_as_string"])

# -------- FIN CHATGPT DATA ENRICHMENT ----------------------------------------------------------------------------------------------

# Merge with original sample
new_data = pd.concat([new_data.reset_index(drop=True), classified_df], axis=1)

# -------- DEBUT EMBEDING OPENAI LARGE ----------------------------------------------------------------------------------------------

from tqdm import tqdm
import time

# --- Function to build a natural-language sentence per row ---
def build_sentence(row):
    return (
        f"Le titre de ce job est {row.get('titre', '')} et est à {row.get('Ville', '')} "
        f"dans le département {row.get('Code Postal', '')} de la région {row.get('Region', '')}. "
        f"Ce poste est {row.get('IA_Logé', '')} et le contrat est en {row.get('IA_Type_de_contrat', '')}. "
        f"Le salaire est de {row.get('IA_Salaire', '')}. Particularité : {row.get('IA_Couple_accepté', '')} "
        f"et {row.get('IA_Urgent', '')}. Le job est dans un {row.get('IA_Catégorie_établissement', '')} "
        f"et dans le secteur de {row.get('IA_Catégorie_Job_1', '')}. "
        f"Détails du job : {row.get('IA_Catégorie_Job_2', '')} {row.get('IA_Catégorie_Job_3', '')}. "
        f"Il commence le {row.get('IA_Date_de_début', '')} et dure {row.get('IA_Durée', '')}. "
        f"L'établissement est situé {row.get('IA_Type_environnement', '')} et est {row.get('IA_Chaine_Indépendant', '')}. "
        f"C'est un contrat à {row.get('IA_Temps_de_travail', '')} et travail de {row.get('IA_Horaires_de_travail', '')} "
        f"pendant {row.get('IA_Heures_par_semaine', '')} heures par semaine. "
        f"Le poste est pour les {row.get('IA_Éxpérience', '')}."
    )

# --- Create the combined natural-language field ---
new_data["combined_text"] = new_data.apply(build_sentence, axis=1)

# --- Define batching function ---
def embed_in_batches(texts, batch_size=1500):
    """
    Embed a list of texts in batches to avoid API rate limits and memory issues.
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        try:
            response = client_ai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error at batch {i}: {e}")
            time.sleep(5)  # Wait a bit and retry
            continue
    return embeddings

# --- Generate embeddings ---
new_data["OpenAIEmbeddedSmall"] = embed_in_batches(new_data["combined_text"].tolist(), batch_size=1500)

# --- Optional: Clean up ---
new_data.drop(columns=["combined_text"], inplace=True)

# -------- FIN EMBEDING OPENAI LARGE ----------------------------------------------------------------------------------------------



# Combine and remove duplicates
if not existing_data.empty:
    print(len(pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(subset=['URL'])))
    combined_data = pd.concat([existing_data, new_data], ignore_index=True).drop_duplicates(
        subset=['URL']
    )
else:
    combined_data = new_data

# -------- DEBUT DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------

# Select columns starting with "IA_"
ia_cols = [col for col in combined_data.columns if col.startswith("IA_")]

# Replace "" with "Non spécifié" in those columns only
combined_data[ia_cols] = combined_data[ia_cols].replace("", "Non spécifié")

# -------- FIN DATA VALIDATION EMPTY VALUES OPENAI ----------------------------------------------------------------------------------------------


print(f"Post concat Check combined_data length {len(combined_data)}")

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
    # Replace special characters (-, ') with a space.
    cleaned = re.sub(r"[-']", " ", without_accents)
    # Remove other special characters (retain letters, digits, and whitespace).
    cleaned = re.sub(r"[^A-Za-z0-9\s]", "", cleaned)
    return cleaned

# Create the new column "Titre annonce sans accent" by applying the function on "intitule".
combined_data["TitreAnnonceSansAccents"] = combined_data["Title"].apply(
    lambda x: remove_accents_and_special(x) if isinstance(x, str) else x
)

print(f"Post concat Check combined_data length {len(combined_data)}")

# Update Google Sheets with the combined data
#worksheet.clear()  # Clear existing content
#worksheet.update([combined_data.columns.tolist()] + combined_data.values.tolist())


# ==========================================
# 🚀 BATCH SEND TO GOOGLE SHEETS (500 rows)
# ==========================================

def number_to_column(n):
    """Convert 1 → A, 2 → B, ..., 27 → AA."""
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result

def update_sheet_in_batches(worksheet, df, batch_size=500):
    df = df.astype(str)
    values = [df.columns.tolist()] + df.values.tolist()
    total_rows = len(values)

    print(f"Uploading {total_rows} rows to Google Sheets (batches of {batch_size})...")

    worksheet.clear()

    num_cols = df.shape[1]
    last_col_letter = number_to_column(num_cols)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = values[start:end]

        start_row = start + 1
        end_row = start_row + len(batch) - 1

        range_name = f"A{start_row}:{last_col_letter}{end_row}"

        print(f" → Updating rows {start_row} to {end_row} into {range_name}")

        for attempt in range(5):
            try:
                worksheet.update(range_name, batch)
                break
            except Exception as e:
                print(f"   !!! Error attempt {attempt+1}: {e}")
                time.sleep(2 ** attempt)
        else:
            raise Exception("Failed after multiple retry attempts")

    print("✓ Sheets updated successfully in batches.")


# Run batch update
update_sheet_in_batches(worksheet, combined_data)


