import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import time

chrome_options = Options()
chrome_options.add_argument("--headless")
# Set up the Selenium WebDriver (make sure to have the appropriate driver installed)
driver = webdriver.Chrome() 

def get_total_pages(book_id):
    # Fetch the first page to find out the total number of pages
    url = f"https://www.sto.cx/book-{book_id}-1.html"
    driver.get(url)

    # Wait for the content to load
    time.sleep(1)

    # Get the page source and pass it to BeautifulSoup
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    
    # Find the div that contains the book content
    book_content_div = soup.find('div', id='webPage')

    if not book_content_div:
        raise Exception("Could not find the book content on the webpage")

    # Find the select tag inside the book content div
    page_select = book_content_div.find('select')

    if not page_select:
        raise Exception("Could not find the page selector inside the book content")

    # Extract the total number of pages by looking at the option values
    options = page_select.find_all('option')
    total_pages = max(int(option.contents[0]) for option in options)

    return total_pages

def scrape_page(book_id, page_number):
    # Fetch the content of a specific page
    url = f"https://www.sto.cx/book-{book_id}-{page_number}.html"
    driver.get(url)

    # Wait for the page to load (adjust time if necessary)
    time.sleep(3)

    # Get the page source and pass it to BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract the book content
    book_content = soup.find('div', id='BookContent')

    if not book_content:
        raise Exception(f"Could not find book content on page {page_number}")

    return book_content.text.strip()

def scrape_book(book_id):
    total_pages = get_total_pages(book_id)
    print(f"Total pages found: {total_pages}")

    book_content = ""

    for page in range(1, total_pages + 1):
        print(f"Scraping page {page}/{total_pages}...")
        page_content = scrape_page(book_id, page)
        book_content += page_content + "\n\n"  # Add some spacing between pages

    return book_content

# Example usage
if __name__ == "__main__":
    book_id = 225588  # Replace with the actual book ID
    book_text = scrape_book(book_id)
    
    # Save the book text to a file
    with open(f"book_{book_id}.txt", "w", encoding="utf-8") as f:
        f.write(book_text)

    print(f"Book content saved to book_{book_id}.txt")
    driver.quit()