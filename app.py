import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from io import BytesIO
from pypdf import PdfReader

# Function to extract text from PDF content
def extract_pdf_content(url, content):
    try:
        pdf_file = BytesIO(content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting PDF content from {url}: {str(e)}"

# Function to fetch content from a single URL
async def fetch_content(session, url):
    try:
        async with session.get(url, timeout=30) as response:
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                content = await response.read()
                text = extract_pdf_content(url, content)
                return url, text, 'pdf'
            else:
                content = await response.text()
                return url, content, 'html'
    except Exception as e:
        return url, f"Error fetching {url}: {str(e)}", 'error'

# Main function to process URLs asynchronously
async def process_urls(urls):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_content(session, url) for url in urls]
        progress_bar = st.progress(0)
        for idx, future in enumerate(asyncio.as_completed(tasks)):
            url, content, content_type = await future
            results.append((url, content, content_type))
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
                for url, content, content_type in results:
                    if content_type == 'error':
                        output_data.append({'URL': url, 'Extracted Text': content})
                    else:
                        if content_type == 'pdf':
                            text = content
                        elif content_type == 'html':
                            # Parse and extract text using BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text(separator='\n')
                        else:
                            text = content
                        output_data.append({'URL': url, 'Extracted Text': text})

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
