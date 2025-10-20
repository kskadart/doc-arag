from typing import Dict, Any
import httpx
from bs4 import BeautifulSoup


async def scrape_url(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Scrape a web page and extract its content.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with html, text, title, and metadata
        
    Raises:
        Exception: If scraping fails
    """
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            html_content = response.text
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else "Untitled"
            
            # Extract main content
            # Try to find main content area
            main_content = (
                soup.find("main") or 
                soup.find("article") or 
                soup.find("div", class_="content") or
                soup.find("body")
            )
            
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            cleaned_text = "\n\n".join(lines)
            
            return {
                "html": html_content,
                "text": cleaned_text,
                "title": title.strip() if title else "Untitled",
                "url": str(url),
                "status_code": response.status_code,
            }
    
    except httpx.HTTPError as e:
        raise Exception(f"HTTP error while scraping {url}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to scrape {url}: {str(e)}")


def clean_html_text(html: str) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html: HTML content
        
    Returns:
        Cleaned text
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    return "\n\n".join(lines)

