import time
import json
import re
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def download_judiciary_data(
    base_url='https://ara.jri.ac.ir',
    start_page=1,
    end_page=10,
    delay=1
):
    # --- Configuration ---
    STOP_PHRASES = [
        'فهرست', 'نقد رأی', 'نقد رای', 'نام', 'رایانامه', 'رای نامه',
        'توضیح', 'کدتصویری', 'کد تصویری', 'تعدادموافق', 'تعداد مخالف',
        'تعدادمخالف', 'نقدهای شما', 'تماس با ما', 'پیوندها', 'نسخه', 'Main Footer'
    ]

    # --- Setup Selenium ---
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    # options.add_argument("--headless") # Uncomment to run invisibly

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    try:
        print(f"🌍 Opening site: {base_url}/Judge/Index")
        driver.get(f"{base_url}/Judge/Index")
        time.sleep(3)

        # ==========================
        # Loop Through Pages
        # ==========================
        for page_num in range(start_page, end_page + 1):
            print(f"\n📄 Processing Page {page_num}...")
            page_data = [] # Reset data for this page

            # 1. Pagination: Type Page Number & Enter
            try:
                # Find the input box
                try:
                    page_input = driver.find_element(By.NAME, "Page")
                except:
                    page_input = driver.find_element(By.XPATH, "//input[contains(@id, 'Page')]")

                # If the number isn't already there, type it
                if page_input.get_attribute('value') != str(page_num):
                    page_input.clear()
                    page_input.send_keys(str(page_num))
                    page_input.send_keys(Keys.ENTER)
                    time.sleep(4) # Wait for table reload
            except Exception as e:
                print(f"❌ Error changing page: {e}")
                continue

            # 2. Parse List Rows
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            rows = soup.select('table tr')[1:] # Skip header
            print(f"   Found {len(rows)} rulings on this page.")

            # Extract row metadata first
            row_items = []
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 6: continue
                link_tag = cells[1].find('a')
                if not link_tag: continue
                
                href = link_tag.get('href')
                full_url = base_url + href if href.startswith('/') else href
                
                row_items.append({
                    'metadata': {
                        'title': link_tag.get_text(strip=True),
                        'date': cells[2].get_text(strip=True),
                        'case_number': cells[3].get_text(strip=True),
                        'domain': cells[4].get_text(strip=True),
                        'authority': cells[5].get_text(strip=True),
                        'language': 'fa',
                        'source_url': full_url
                    },
                    'raw_href': href # Needed for Related Laws ID
                })

            # 3. Visit Details (New Tab)
            main_window = driver.current_window_handle

            for item in row_items:
                meta = item['metadata']
                print(f"   → Scraping: {meta['title'][:30]}...")

                # Open link in new tab
                driver.execute_script("window.open(arguments[0], '_blank');", meta['source_url'])
                driver.switch_to.window(driver.window_handles[-1])

                try:
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    doc_soup = BeautifulSoup(driver.page_source, 'html.parser')

                    # --- Extract Message ---
                    message = None
                    for label in doc_soup.find_all(string=re.compile(r'پیام')):
                        parent = label.parent
                        if parent:
                            msg = parent.find_next_sibling(string=True)
                            if msg:
                                message = msg.strip()
                                break

                    # --- Extract Decision Text ---
                    collecting = False
                    decision_lines = []
                    for text in doc_soup.stripped_strings:
                        if text.startswith(('رأی', 'رای')): collecting = True
                        if not collecting: continue
                        if any(text.startswith(p) for p in STOP_PHRASES): break
                        decision_lines.append(text)
                    decision_text = '\n'.join(decision_lines).strip()

                    # --- Extract Related Laws ---
                    related_laws = []
                    decision_id = item['raw_href'].rstrip('/').split('/')[-1]
                    relations_url = f"{base_url}/Judge/Relations/{decision_id}"
                    
                    driver.get(relations_url) # Go to relations in same tab
                    time.sleep(0.5)
                    
                    rel_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    for a in rel_soup.find_all('a', href=True):
                        if 'ilaws.net' in a['href']:
                            related_laws.append({
                                'law_title': a.get_text(strip=True),
                                'law_url': a['href']
                            })
                    
                    # Deduplicate laws
                    unique_laws = {l['law_url']: l for l in related_laws}
                    
                    # Append completed record
                    page_data.append({
                        'metadata': meta,
                        'message': message,
                        'decision_text': decision_text,
                        'related_laws': list(unique_laws.values())
                    })

                except Exception as e:
                    print(f"     ❌ Error: {e}")

                # Close tab and return
                driver.close()
                driver.switch_to.window(main_window)
                time.sleep(delay)

            # ==========================
            # Save File Per Page
            # ==========================
            if page_data:
                filename = f"judgments-{page_num}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, ensure_ascii=False, indent=2)
                print(f"✅ Saved: {filename}")
            else:
                print(f"⚠️ Page {page_num} was empty, nothing saved.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
    finally:
        driver.quit()
        print("\n👋 Done.")

# =======================
# Run
# =======================
if __name__ == '__main__':
    download_judiciary_data(
        start_page=1,
        end_page=1026,  # Last page
        delay=0
    )