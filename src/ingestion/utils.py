import os
import fitz  
import tqdm

import requests
from bs4 import BeautifulSoup

from langchain_experimental.text_splitter import SemanticChunker


def get_all_files_with_types(types, directory):
    files = []

    # Walk through the directory and its subdirectories
    for root, _, filenames in os.walk(directory):
        # Loop through each file in the current directory
        for filename in filenames:
            # Check if the file ends with one of the specified types, append if it does
            if filename.lower().endswith(tuple(types)):
                files.append(os.path.join(root, filename))

    return files


def fetch_website_content(url):
    try:
        # Send an HTTP request to get the website content
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None

    # Parse the webpage content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text content from the parsed HTML
    content = soup.get_text(separator=' ')  
    return content
