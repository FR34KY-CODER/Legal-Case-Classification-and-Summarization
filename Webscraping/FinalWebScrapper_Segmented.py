import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

# ==== Setup Browser Options ====
options1 = uc.ChromeOptions()
options1.add_argument("--start-maximized")
options1.add_argument(r"--user-data-dir=C:\Users\FR34K\AppData\Local\Google\Chrome\User Data")
options1.add_argument("--profile-directory=Default")

# ==== HARDCODED BASE URL LIST ====
base_urls = [
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=16",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=17",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=18",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=19",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=20",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=21",
    "https://indiankanoon.org/search/?formInput=doctypes%3A%20supremecourt%20year%3A%202025&pagenum=22",

    # Add more pagination links as needed
]

# ==== Collect Case Links ====
driver = uc.Chrome(options=options1)
all_cases = []

for url in base_urls:
    print(f"Processing URL: {url}")
    driver.get(url)

    try:
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "result_title")))
    except Exception as e:
        print(f"Error loading {url}: {e}")
        continue

    while True:
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        case_elements = soup.find_all("div", class_="result_title")

        for case in case_elements:
            try:
                a_tag = case.find("a")
                title = a_tag.get_text(strip=True)
                href = a_tag.get("href")
                if not href.startswith("http"):
                    href = "https://indiankanoon.org" + href
                date_elem = case.find_next("div", class_="result_subtitle")
                date_text = date_elem.get_text(strip=True) if date_elem else ""
                all_cases.append({"Case Title": title, "Date": date_text, "Link": href})
            except Exception as ex:
                print(f"Error parsing case: {ex}")

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "a[rel='next']")
            next_button.click()
            WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, "result_title")))
        except:
            break

driver.quit()
print(f"✅ Collected {len(all_cases)} cases.")

# ==== Section-based Extraction ====
def extract_sections_from_case(soup):
    expected_sections = [
        "Issue", "Precedent Analysis", "Analysis of the law", "Fact",
        "Respondent's Argument", "Petitioner's Argument",
        "Court's Reasoning", "Conclusion"
    ]
    sections = {section: "" for section in expected_sections}

    p_tags = soup.find_all("p", title=True)
    for p in p_tags:
        title_attr = p.get("title")
        if title_attr in expected_sections:
            sections[title_attr] += p.get_text(separator="\n", strip=True) + "\n"

    for section in sections:
        sections[section] = sections[section].strip()
    return sections

# ==== Extract Data from All Case Links ====
options2 = uc.ChromeOptions()
options2.add_argument("--start-maximized")
options2.add_argument(r"--user-data-dir=C:\Users\FR34K\AppData\Local\Google\Chrome\User Data")
options2.add_argument("--profile-directory=Default")

driver2 = uc.Chrome(options=options2)
segregated_contents = []

start_all = time.time()

for idx, case in enumerate(all_cases):
    start_case = time.time()
    try:
        driver2.get(case["Link"])
        WebDriverWait(driver2, 30).until(EC.presence_of_element_located((By.TAG_NAME, "p")))
        time.sleep(1.5)
        soup_case = BeautifulSoup(driver2.page_source, "html.parser")
        sections = extract_sections_from_case(soup_case)
        segregated_contents.append(sections)
        elapsed = time.time() - start_case
        print(f"Processed case {idx+1}/{len(all_cases)} in {elapsed:.2f}s")
    except Exception as ex:

        print(f"Error processing case {idx+1}: {ex}")
        segregated_contents.append({section: "" for section in [
            "Issue", "Precedent Analysis", "Analysis of the law", "Fact",
            "Respondent's Argument", "Petitioner's Argument",
            "Court's Reasoning", "Conclusion"]})

driver2.quit()
total_elapsed = time.time() - start_all
print(f"✅ Section-based extraction complete in {total_elapsed/60:.2f} minutes")

# ==== Save Final Output ====
merged_data = []
for idx, case in enumerate(all_cases):
    merged = case.copy()
    merged.update(segregated_contents[idx])
    merged_data.append(merged)

pd.DataFrame(merged_data).to_csv("Legal_Case_Dataset_Final2.csv", index=False, encoding="utf-8")
print("✅ CSV file saved with segmented data")