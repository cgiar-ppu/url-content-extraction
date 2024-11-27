import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from io import BytesIO
import re
import urllib.parse  # Import urllib.parse to resolve URLs
# Import extract_text from pdfminer.high_level
from pdfminer.high_level import extract_text

# Function to extract text from PDF content using pdfminer.six
def extract_pdf_content(url, content):
    try:
        if not content or len(content) < 100:
            return ""
        content_stream = BytesIO(content)
        try:
            # Use pdfminer.six's extract_text function to extract text with proper spacing
            text = extract_text(content_stream)
            return text
        except Exception as e:
            return f"Error extracting PDF content from {url}: {str(e)}"
    except Exception as e:
        return f"Error extracting PDF content from {url}: {str(e)}"

# Function to extract PDF links from the HTML content
def extract_pdf_links(soup, base_url):
    links = set()

    # Extract links from meta tags
    for meta in soup.find_all('meta', {'name': re.compile("citation_(pdf|abstract)_url")}):
        content = meta.get('content')
        if content:
            links.add(content)

    # Extract PDF links from anchor tags
    for link in soup.find_all('a', href=True):
        href = link['href']
        if any(keyword in href.lower() for keyword in ['download', 'pdf']):
            if not href.startswith('http'):
                href = urllib.parse.urljoin(base_url, href)  # Use urllib.parse.urljoin to build the full URL
            links.add(href)

    return links

# Function to fetch content from a single URL
async def fetch_content(session, url):
    try:
        async with session.get(url, timeout=30) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                # Directly fetch and extract content from PDF
                content = await response.read()
                text = extract_pdf_content(url, content)
                return url, text, '', 'pdf'  # Empty string for PDF links content
            else:
                # Handle HTML content
                content = await response.text()
                # Parse and extract text using BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n')
                # Extract PDF links from the page
                pdf_links = extract_pdf_links(soup, url)
                pdf_contents = []
                for pdf_link in pdf_links:
                    try:
                        async with session.get(pdf_link, timeout=30) as pdf_response:
                            pdf_response.raise_for_status()
                            pdf_content = await pdf_response.read()
                            pdf_text = extract_pdf_content(pdf_link, pdf_content)
                            pdf_contents.append(pdf_text)
                    except Exception as e:
                        pdf_contents.append(f"Error fetching {pdf_link}: {str(e)}")
                # Combine all PDF contents
                combined_pdf_content = "\n".join(pdf_contents)
                return url, text, combined_pdf_content, 'html'
    except Exception as e:
        return url, f"Error fetching {url}: {str(e)}", '', 'error'

# Main function to process URLs asynchronously
async def process_urls(urls):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_content(session, url) for url in urls]
        progress_bar = st.progress(0)
        for idx, future in enumerate(asyncio.as_completed(tasks)):
            url, content, pdf_content, content_type = await future
            results.append((url, content, pdf_content, content_type))
            progress_bar.progress((idx + 1) / len(tasks))
        return results

# Streamlit app
def main():
    st.title("URL Content Extractor")

    uploaded_file = st.file_uploader("Upload an Excel file with a list of URLs", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(df)

        # Assume the URLs are in a column named 'URL'
        if 'URL' in df.columns:
            url_cells = df['URL'].dropna().tolist()
            # Split URLs by comma and strip whitespace
            urls = [url.strip() for cell in url_cells for url in str(cell).split(',')]
            # Get unique URLs
            urls = list(set(urls))
            if st.button("Start Extraction"):
                st.write("Processing URLs...")
                # Run the asynchronous processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(process_urls(urls))

                # Prepare output data
                output_data = []
                for url, content, pdf_content, content_type in results:
                    if content_type == 'error':
                        output_data.append({'URL': url, 'Extracted Text': content, 'PDF Content': ''})
                    else:
                        output_data.append({'URL': url, 'Extracted Text': content, 'PDF Content': pdf_content})

                # Create DataFrame from the results
                output_df = pd.DataFrame(output_data)

                # Display results
                st.write("Extraction Results:")
                st.dataframe(output_df)

                # Provide a way to download the DataFrame as CSV
                csv = output_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='extracted_text.csv',
                    mime='text/csv',
                )
            else:
                st.write("Click the button to start extraction.")
        else:
            st.error("The uploaded Excel file must contain a column named 'URL'.")

if __name__ == "__main__":
    main()
