import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import random  # Import random for delays
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
async def fetch_content(session, semaphore, other_data, url, min_delay, max_delay, base_delay, max_retries, status_queue):
    async with semaphore:
        retries = 0
        error_message = ''
        while retries <= max_retries:
            try:
                # Wait a random delay before making the request
                delay = random.uniform(min_delay, max_delay)
                await asyncio.sleep(delay)
                # Now make the request
                status_queue.append(f"Fetching URL: {url}")
                async with session.get(url, timeout=30, ssl=False) as response:
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                        # Directly fetch and extract content from PDF
                        content = await response.read()
                        text = extract_pdf_content(url, content)
                        if not text.strip():
                            raise Exception("Extracted PDF content is empty.")
                        status_queue.append(f"Successfully extracted PDF content from {url}")
                        return other_data, url, text, '', 'pdf', retries, ''
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
                            pdf_retries = 0
                            while pdf_retries <= max_retries:
                                try:
                                    # Wait before fetching each PDF link
                                    pdf_delay = random.uniform(min_delay, max_delay)
                                    await asyncio.sleep(pdf_delay)
                                    status_queue.append(f"Fetching PDF link: {pdf_link}")
                                    async with session.get(pdf_link, timeout=30, ssl=False) as pdf_response:
                                        pdf_response.raise_for_status()
                                        pdf_content = await pdf_response.read()
                                        pdf_text = extract_pdf_content(pdf_link, pdf_content)
                                        if not pdf_text.strip():
                                            raise Exception("Extracted PDF content is empty.")
                                        pdf_contents.append(pdf_text)
                                        status_queue.append(f"Successfully extracted PDF content from {pdf_link}")
                                    break  # Exit the retry loop on success
                                except Exception as e:
                                    pdf_retries += 1
                                    pdf_error_message = str(e)
                                    status_queue.append(f"Error fetching {pdf_link}: {e} (Retry {pdf_retries}/{max_retries})")
                                    if pdf_retries > max_retries:
                                        status_queue.append(f"Max retries reached for {pdf_link}. Skipping.")
                                        break
                                    backoff_delay = base_delay * (2 ** (pdf_retries - 1)) + random.uniform(0, 1)
                                    await asyncio.sleep(backoff_delay)
                                    continue
                        # Combine all PDF contents
                        combined_pdf_content = "\n".join(pdf_contents)
                        if not text.strip() and not combined_pdf_content.strip():
                            raise Exception("Extracted content is empty.")
                        status_queue.append(f"Successfully extracted content from {url}")
                        return other_data, url, text, combined_pdf_content, 'html', retries, ''
            except Exception as e:
                retries += 1
                error_message = str(e)
                status_queue.append(f"Error fetching {url}: {e} (Retry {retries}/{max_retries})")
                if retries > max_retries:
                    # Return empty content on error; optionally log the error
                    status_queue.append(f"Max retries reached for {url}. Skipping.")
                    return other_data, url, '', '', 'error', retries, error_message
                backoff_delay = base_delay * (2 ** (retries - 1)) + random.uniform(0, 1)
                await asyncio.sleep(backoff_delay)
                continue  # Retry the loop

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
            # Add options to set parameters
            st.sidebar.write("Settings")
            max_concurrent_tasks = st.sidebar.slider("Max Concurrent Tasks", 1, 50, 10)
            min_delay = st.sidebar.number_input("Min Delay between requests (seconds)", min_value=0.0, value=1.0)
            max_delay = st.sidebar.number_input("Max Delay between requests (seconds)", min_value=0.0, value=3.0)
            base_delay = st.sidebar.number_input("Base Delay for Exponential Backoff (seconds)", min_value=1.0, value=2.0)
            max_retries = st.sidebar.number_input("Max Retries on Error", min_value=0, value=3)

            tasks_to_process = []
            # Iterate over each row in the dataframe
            for idx, row in df.iterrows():
                cell = row['URL']
                urls_in_cell = [url.strip() for url in str(cell).split(',')]
                # For each URL, store the entire row data along with the URL
                other_data = row.to_dict()
                del other_data['URL']  # Remove the 'URL' key, as we will have the URL separately
                for url in urls_in_cell:
                    tasks_to_process.append((other_data, url))

            if st.button("Start Extraction"):
                st.write("Processing URLs...")
                # Placeholders for dynamic updates
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                result_table_placeholder = st.empty()
                log_placeholder = st.container()

                # Shared status queue for logging
                status_queue = []

                # Run the asynchronous processing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def run_tasks():
                    results = []
                    result_df = pd.DataFrame(columns=['URL', 'Status', 'Retries', 'Error Message'])
                    semaphore = asyncio.Semaphore(max_concurrent_tasks)
                    async with aiohttp.ClientSession() as session:
                        # Wrap coroutines in tasks
                        tasks = [asyncio.create_task(fetch_content(
                            session, semaphore, other_data, url, min_delay, max_delay, base_delay, max_retries, status_queue
                        )) for other_data, url in tasks_to_process]
                        total_tasks = len(tasks)
                        for idx, future in enumerate(asyncio.as_completed(tasks)):
                            # Update progress information
                            progress = (idx + 1) / total_tasks
                            progress_bar.progress(progress)
                            other_data, url, content, pdf_content, content_type, retries, error_message = await future
                            # Prepare the result row by combining other data with extraction results
                            result_row = other_data.copy()
                            result_row['URL'] = url
                            result_row['Extracted Text'] = content
                            result_row['PDF Content'] = pdf_content
                            result_row['Retries'] = retries
                            result_row['Error Message'] = error_message
                            status = 'Success' if content or pdf_content else 'Failed'
                            result_row['Status'] = status
                            results.append(result_row)
                            # Update result DataFrame
                            result_df = pd.DataFrame(results)
                            # Display partial results
                            result_table_placeholder.dataframe(result_df)
                            # Update status placeholder
                            status_placeholder.write(f"Processed {idx + 1}/{total_tasks} URLs")
                            # Display status messages as bullet points
                            log_placeholder.markdown('\n'.join(f"- {line}" for line in status_queue))
                    return result_df

                output_df = loop.run_until_complete(run_tasks())

                # Sanitize the DataFrame to remove illegal characters
                for col in output_df.select_dtypes(include=['object']).columns:
                    output_df[col] = output_df[col].apply(remove_illegal_characters)

                # Display final results
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
