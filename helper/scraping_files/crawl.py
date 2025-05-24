import requests
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
import time

# Configuration
api_key = 'ce50e5669ed7408efbe51ba92ab0fd4f'
sitemap_url = 'https://www.madewithnestle.ca/sitemap.xml' 
all_urls = set()

def scrape_url(url):
    payload = {
        'api_key': api_key,
        'url': url
    }
    
    try:
        response = requests.get('https://api.scraperapi.com/', params=payload)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
        return None

def parse_sitemap(content, base_url):
    urls = set()
    
    try:
        root = ET.fromstring(content)
        namespaces = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        for url_elem in root.findall('.//sitemap:url', namespaces):
            loc_elem = url_elem.find('sitemap:loc', namespaces)
            if loc_elem is not None:
                urls.add(loc_elem.text)
        for sitemap_elem in root.findall('.//sitemap:sitemap', namespaces):
            loc_elem = sitemap_elem.find('sitemap:loc', namespaces)
            if loc_elem is not None:
                urls.add(loc_elem.text)
                
    except ET.ParseError:
        #If XML parsing fails, try parsing as HTML
        soup = BeautifulSoup(content, 'html.parser')
        for link in soup.find_all('a', href=True):
            full_url = urljoin(base_url, link['href'])
            if 'madewithnestle.ca' in full_url:
                urls.add(full_url)
    
    return urls

# Try different sitemap URLs
sitemap_urls = [
    'https://www.madewithnestle.ca/sitemap.xml',
    'https://www.madewithnestle.ca/sitemap',
    'https://www.madewithnestle.ca/sitemap.html',
]

for sitemap_url in sitemap_urls:
    print(f"Trying sitemap: {sitemap_url}")
    response = scrape_url(sitemap_url) 
    if response:
        urls_found = parse_sitemap(response.text, sitemap_url)
        all_urls.update(urls_found)
        print(f"  Found {len(urls_found)} URLs")
        
        # if this is a sitemap index, crawl the individual sitemaps
        for url in list(urls_found):
            if 'sitemap' in url and url.endswith('.xml'):
                print(f"  Crawling sub-sitemap: {url}")
                sub_response = scrape_url(url)
                if sub_response:
                    sub_urls = parse_sitemap(sub_response.text, url)
                    all_urls.update(sub_urls)
                    print(f"    Found {len(sub_urls)} additional URLs")
                time.sleep(0.5)
    
    time.sleep(1)

#Filter out sitemap URLs from final list
final_urls = {url for url in all_urls if not url.endswith('.xml') or 'sitemap' not in url}

print(f"\nTotal URLs found: {len(final_urls)}")

with open("all_madewithnestle_urls2.txt", "w", encoding="utf-8") as f:
    for url in sorted(final_urls):
        f.write(url + "\n")