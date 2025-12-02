"""
Web Search Integration for Telegram Bot
Adds DuckDuckGo search and web page reading capabilities
"""

from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, link, and snippet
    """
    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('href', ''),
                    'snippet': result.get('body', '')
                })
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
    except Exception as e:
        logger.error(f"Error searching web: {e}")
        return []

def fetch_webpage(url: str, max_length: int = 2000) -> str:
    """
    Fetch and extract text content from a webpage
    
    Args:
        url: URL to fetch
        max_length: Maximum length of extracted text
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Truncate if needed
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        logger.info(f"Fetched {len(text)} characters from {url}")
        return text
        
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {e}")
        return f"Error fetching page: {str(e)}"

def web_search_command(query: str) -> str:
    """
    Perform a web search and format results
    
    Args:
        query: Search query
        
    Returns:
        Formatted search results
    """
    if not query:
        return "Please provide a search query. Example: /search python tutorials"
    
    results = search_web(query, max_results=5)
    
    if not results:
        return "No results found or search failed."
    
    output = f"🔍 **Search results for:** {query}\n\n"
    
    for i, result in enumerate(results, 1):
        output += f"{i}. **{result['title']}**\n"
        output += f"   {result['snippet']}\n"
        output += f"   🔗 {result['link']}\n\n"
    
    return output

def web_read_command(url: str) -> str:
    """
    Read and summarize a webpage
    
    Args:
        url: URL to read
        
    Returns:
        Page content or error message
    """
    if not url:
        return "Please provide a URL. Example: /read https://example.com"
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    content = fetch_webpage(url, max_length=3000)
    
    if content.startswith("Error"):
        return content
    
    return f"📄 **Content from:** {url}\n\n{content}"
