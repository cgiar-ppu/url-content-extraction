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

def remove_illegal_characters(text):
    if isinstance(text, str):
        # Define the illegal characters according to the XML 1.0 specification
        illegal_chars = [
            (0x00, 0x08),
            (0x0B, 0x0C),
            (0x0E, 0x1F),
            (0x7F, 0x84),
            (0x86, 0x9F),
            (0xFDD0, 0xFDEF),
            (0xFFFE, 0xFFFF),
        ]
        # Exclude tab (0x09), line feed (0x0A), and carriage return (0x0D)
        return ''.join(
            c for c in text
            if not any(low <= ord(c) <= high for (low, high) in illegal_chars)
        )
    else:
        return text


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
            # Optionally log the error
            # print(f"Error extracting PDF content from {url}: {str(e)}")
            return ""
    except Exception as e:
        # Optionally log the error
        # print(f"Error extracting PDF content from {url}: {str(e)}")
        return ""

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
async def fetch_content(session, A, url):
    try:
        async with session.get(url, timeout=30, ssl=False) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                # Directly fetch and extract content from PDF
                content = await response.read()
                text = extract_pdf_content(url, content)
                return A, url, text, '', 'pdf'  # Empty string for PDF links content
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
                        async with session.get(pdf_link, timeout=30, ssl=False) as pdf_response:
                            pdf_response.raise_for_status()
                            pdf_content = await pdf_response.read()
                            pdf_text = extract_pdf_content(pdf_link, pdf_content)
                            pdf_contents.append(pdf_text)
                    except Exception as e:
                        # Skip appending error message; optionally log the error
                        # print(f"Error fetching {pdf_link}: {str(e)}")
                        continue  # Move to the next PDF link
                # Combine all PDF contents
                combined_pdf_content = "\n".join(pdf_contents)
                return A, url, text, combined_pdf_content, 'html'
    except Exception as e:
        # Return empty content on error; optionally log the error
        # print(f"Error fetching {url}: {str(e)}")
        return A, url, '', '', 'error'

# Main function to process URLs asynchronously
async def process_urls(tasks_to_process):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_content(session, A, url) for A, url in tasks_to_process]
        progress_bar = st.progress(0)
        for idx, future in enumerate(asyncio.as_completed(tasks)):
            A, url, content, pdf_content, content_type = await future
            results.append((A, url, content, pdf_content, content_type))
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
            # Build a list of (A, URL) tuples
            tasks_to_process = []
            for cell in url_cells:
                urls_in_cell = [url.strip() for url in str(cell).split(',')]
                for url in urls_in_cell:
                    tasks_to_process.append((cell, url))

            if st.button("Start Extraction"):
                st.write("Processing URLs...")
                # Run the asynchronous processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(process_urls(tasks_to_process))

                # Prepare output data
                output_data = []
                for A, url, content, pdf_content, content_type in results:
                    # Skip entries with empty content if desired
                    if content or pdf_content:
                        output_data.append({'A': A, 'URL': url, 'Extracted Text': content, 'PDF Content': pdf_content})
                    else:
                        # Optionally include entries with no content
                        output_data.append({'A': A, 'URL': url, 'Extracted Text': '', 'PDF Content': ''})

                # Create DataFrame from the results
                output_df = pd.DataFrame(output_data)

                # Sanitize the DataFrame to remove illegal characters
                for col in output_df.select_dtypes(include=['object']).columns:
                    output_df[col] = output_df[col].apply(remove_illegal_characters)

                # Display results
                st.write("Extraction Results:")
                st.dataframe(output_df)

                # Provide a way to download the DataFrame as Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    output_df.to_excel(writer, index=False)
                excel_data = output.getvalue()

                st.download_button(
                    label="Download data as Excel",
                    data=excel_data,
                    file_name='extracted_text.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            else:
                st.write("Click the button to start extraction.")
        else:
            st.error("The uploaded Excel file must contain a column named 'URL'.")

if __name__ == "__main__":
    main()
