#web scraping + PDF extraction automation script
import os #Work with folders and file paths
import io #Handle PDF data in memory
import time #Add waiting time (sleep)
import requests #Download PDFs from internet
import pandas as pd #Create and save CSV tables
import pdfplumber #Read/extract tables from PDFs
from selenium import webdriver #Selenium controls a browser automatically
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

#Function for scrape data for a given year
def scrape_year_data(target_year):
    base_dir = r"G:\A_FYP_PROJECT\code"
    output_file = os.path.join(base_dir, "data", f"affected_population_{target_year}.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure output directory exists otherwise create it

    #Launch Chrome browser using Selenium (automated web scraping) - this will open a browser window and perform actions as if a user is doing it
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://www.dmc.gov.lk/index.php?option=com_dmcreports&view=reports&report_type_id=1&Itemid=273&lang=en"
    driver.get(url)

    #Store the reports we want to process in a list (date, time, download link)
    reports_to_process = []

    try:
        wait = WebDriverWait(driver, 30) #Wait maximum 30 seconds for elements to appear
        from_box = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='From Date']"))) #Find the "From Date" input box using its placeholder text
        to_box = driver.find_element(By.XPATH, "//input[@placeholder='To Date']") #Find the "To Date" input box using its placeholder text

        #Auto-fill the date range for the entire target year
        driver.execute_script(f"arguments[0].value = '{target_year}-01-01';", from_box)
        driver.execute_script(f"arguments[0].value = '{target_year}-12-31';", to_box)

        #Click the search button
        search_icon = driver.find_element(By.CLASS_NAME, "icon-search")
        search_btn = search_icon.find_element(By.XPATH, "./..")
        driver.execute_script("arguments[0].click();", search_btn)

        time.sleep(10) #Gives time for reports to load.
        limit_dropdown = wait.until(EC.presence_of_element_located((By.NAME, "limit"))) #Find the dropdown that controls how many reports are shown per page (name="limit")
        Select(limit_dropdown).select_by_visible_text("All") #Select "All" from the dropdown to show all reports on one page
        time.sleep(7)

        rows = driver.find_elements(By.XPATH, "//table[contains(@class, 'table')]/tbody/tr") #Read all the rows from the reports table
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 4:
                date_val = cells[1].text.strip() #Gets report date 
                time_val = cells[2].text.strip() #Gets report time
                try:
                    download_link = row.find_element(By.PARTIAL_LINK_TEXT, "Download").get_attribute("href") #Extracts PDF URL
                    reports_to_process.append({"date": date_val, "time": time_val, "link": download_link}) #Store the report info in the list for later processing
                except:
                    continue
    finally:
        driver.quit() #browser closes after scraping is done

    all_final_data = [] #hold cleaned data for all reports before saving to CSV
    headers = {"User-Agent": "Mozilla/5.0"} #Set a user agent to mimic a real browser when downloading PDFs, some servers block requests without a user agent

    print("Starting PDF extraction...")
    for report in reports_to_process:
        try:
            resp = requests.get(report['link'], headers=headers, timeout=30) #Download the PDF file from the report link with a timeout of 30 seconds
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf: #Open the PDF in memory using pdfplumber to extract tables without saving the file to disk

                # Variables to remember values from merged cells above
                last_prov = ""
                last_dist = ""
                last_ds = ""
                last_disaster = ""

                for page in pdf.pages: #Loop through each page of the PDF to extract tables
                    table = page.extract_table() #Extracts the table from the current page.
                    if not table: continue

                    for row in table: #Loop through each row of the extracted table
                        clean = [str(c).replace('\n', ' ').strip() if c else "" for c in row] # Clean the row by removing newlines and extra spaces, and converting None to empty strings
                        if len(clean) < 8: continue

                        # Forward Fill Logic: If current cell is empty, keep the last known value
                        if clean[0] != "": last_prov = clean[0]
                        if clean[2] != "": last_dist = clean[2]
                        if clean[3] != "": last_ds = clean[3]
                        if clean[4] != "": last_disaster = clean[4]

                        # Hierarchy & Disaster Checks (Filter conditions)
                        is_western = "western" in last_prov.lower()
                        is_gampaha = "gampaha" in last_dist.lower() or "ගම්පහ" in last_dist
                        is_flood = "flood" in last_disaster.lower()

                        if is_western and is_gampaha and is_flood:
                            f_val = clean[6].replace(",", "") # Get the affected families value from the 7th column (index 6) and remove commas for conversion
                            p_val = clean[7].replace(",", "") # Get the affected people value from the 8th column (index 7) and remove commas for conversion

                            all_final_data.append({ #Append the cleaned and structured data to the list for all reports
                                "Date": report['date'],
                                "Time(24h)": report['time'],
                                "District": "Gampaha",
                                "DS_Division": last_ds,
                                "Disaster": last_disaster,
                                "Affected_Families": int(f_val) if f_val.isdigit() else 0,
                                "Affected_People": int(p_val) if p_val.isdigit() else 0
                            })
        except Exception as e:
            print(f"Error on {report['date']}: {e}") #print error message and continue next pdf without crashing the script if PDF fails.

    if all_final_data:
        df = pd.DataFrame(all_final_data) #Convert the list of dictionaries into a DataFrame for easier manipulation and saving to CSV
        # Sort to ensure chronological order in the CSV
        df = df.sort_values(by=["Date", "Time(24h)", "DS_Division"])
        df.to_csv(output_file, index=False, encoding='utf-8-sig') #creates final CSV file with UTF-8 encoding to handle any special characters in the data, and without the index column
        print(f"\n--- SUCCESS ---")
        print(f"File saved to: {output_file}")

if __name__ == "__main__": #Run the scraper for the target year (you can change this to scrape different years)
    scrape_year_data(2020) #You can change this to scrape different years by calling scrape_year_data with different year values