import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

#Load the files from the sitemap crawl
with open("all_madewithnestle_urls.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]

api_key = 'ce50e5669ed7408efbe51ba92ab0fd4f'

#Text file to save the results for graph database processing
with open("madewithnestle_content.txt", "w", encoding="utf-8") as txt:
    # Write header information
    txt.write("WEBSITE CONTENT EXTRACTION FOR GRAPH DATABASE\n")
    txt.write("=" * 60 + "\n")
    txt.write(f"Total URLs to process: {len(urls)}\n")
    txt.write("=" * 60 + "\n\n")
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"Scraping ({i}/{len(urls)}): {url}")

            txt.write(f"{'='*100}\n")
            txt.write(f"URL {i}: {url}\n")
            txt.write(f"{'='*100}\n\n")
            
            payload = {
                'api_key': api_key,
                'url': url,
                'render': 'true'  
            }
            
            #Make request to ScraperAPI
            response = requests.get('https://api.scraperapi.com/', params=payload)
            # Raise an exception for bad status codes
            response.raise_for_status()  
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            for script in soup(["script", "style"]):
                script.extract()
            
            # Page title
            title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"
            txt.write(f"PAGE TITLE: {title}\n")
            txt.write(f"SOURCE URL: {url}\n\n")
            
            # Remove navigation, footer, and other non-content elements
            for element in soup(["nav", "footer", "header", "aside"]):
                element.extract()
            
            # Get text from main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.main-content', '.post-content', '.entry-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content found, use the whole body
            if not main_content:
                main_content = soup.body or soup
            
            # Extract headings separately for better structure
            headings = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if headings:
                txt.write("HEADINGS:\n")
                for heading in headings:
                    heading_text = heading.get_text(strip=True)
                    if heading_text:
                        txt.write(f"- {heading.name.upper()}: {heading_text}\n")
                txt.write(f"[Source: {url}]\n\n")
            
            # Extract paragraphs and other text content
            text_tags = main_content.find_all(['p', 'li', 'span', 'div', 'article', 'section'])
            
            paragraphs = []
            for tag in text_tags:
                text = tag.get_text(strip=True)
                if text and len(text) > 20:  #filter out very short texts
                    paragraphs.append(text)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_paragraphs = []
            for text in paragraphs:
                if text not in seen:
                    seen.add(text)
                    unique_paragraphs.append(text)
            
            if unique_paragraphs:
                txt.write("MAIN CONTENT:\n")
                for paragraph in unique_paragraphs:
                    txt.write(f"{paragraph}\n\n")
                txt.write(f"[Source: {url}]\n\n")
            else:
                txt.write("MAIN CONTENT: No text content found\n")
                txt.write(f"[Source: {url}]\n\n")
            
            # Gather all image URLs and descriptions
            images = []
            for img in soup.find_all('img', src=True):
                img_url = urljoin(url, img['src'])
                #Filter out small icons, logos, etc.
                if not any(skip in img_url.lower() for skip in ['icon', 'logo', 'sprite', 'pixel']):
                    alt_text = img.get('alt', 'No description')
                    images.append((img_url, alt_text))
            
            if images:
                txt.write("IMAGES:\n")
                for img_url, alt_text in images:
                    txt.write(f"Image URL: {img_url}\n")
                    txt.write(f"Description: {alt_text}\n")
                    txt.write("---\n")
                txt.write(f"[Source: {url}]\n\n")
            
            #Extract links for potential relationship mapping
            links = []
            for link in soup.find_all('a', href=True):
                link_url = urljoin(url, link['href'])
                link_text = link.get_text(strip=True)
                if link_text and len(link_text) > 3:
                    links.append((link_url, link_text))
            
            if links:
                txt.write("INTERNAL LINKS:\n")
                for link_url, link_text in links[:10]:
                    txt.write(f"Link: {link_text} -> {link_url}\n")
                if len(links) > 10:
                    txt.write(f"... and {len(links) - 10} more links\n")
                txt.write(f"[Source: {url}]\n\n")
            
            #add metadata for graph database processing
            txt.write("METADATA FOR GRAPH DB:\n")
            txt.write(f"Source URL: {url}\n")
            txt.write(f"Page Title: {title}\n")
            txt.write(f"Content Sections: Headings={len(headings)}, Paragraphs={len(unique_paragraphs)}, Images={len(images)}, Links={len(links)}\n")
            txt.write(f"{'='*100}\n\n")
            
            txt.flush()  # Ensure content is written immediately
        
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed for {url}: {e}"
            print(error_msg)
            txt.write(f"ERROR PROCESSING URL: {url}\n")
            txt.write(f"Error: {error_msg}\n")
            txt.write(f"{'='*100}\n\n")
            
        except Exception as e:
            error_msg = f"Failed to scrape {url}: {e}"
            print(error_msg)
            txt.write(f"ERROR PROCESSING URL: {url}\n")
            txt.write(f"Error: {error_msg}\n")
            txt.write(f"{'='*100}\n\n")
