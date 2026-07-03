from bs4 import BeautifulSoup
import re
import os

html_path = '/home/xbill/.gemini/antigravity-cli/brain/f8a9fd24-48af-401a-be12-650524abfb57/.system_generated/steps/2971/content.md'
output_path = '/home/xbill/gemma4-tips-aws/gpu-2B-qat-inf-devops-agent/clean_gemma4_docs.md'

if not os.path.exists(html_path):
    print("HTML path not found!")
    exit(1)

with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Parse using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the main documentation body
# Hugging Face docs usually reside in class docstring, class DocBuilderPage, or a container like <main> or articles
main_content = soup.find('main') or soup.find('article') or soup

# Extract text and headers, code blocks
lines = []
for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol', 'table']):
    if element.name in ['h1', 'h2', 'h3', 'h4']:
        level = int(element.name[1])
        lines.append(f"\n{'#' * level} {element.get_text().strip()}")
    elif element.name == 'p':
        text = element.get_text().strip()
        if text:
            lines.append(f"\n{text}")
    elif element.name == 'pre':
        code = element.get_text().strip()
        lines.append(f"\n```python\n{code}\n```")
    elif element.name in ['ul', 'ol']:
        for li in element.find_all('li'):
            lines.append(f"- {li.get_text().strip()}")
    elif element.name == 'table':
        # Simple table extraction
        headers = [th.get_text().strip() for th in element.find_all('th')]
        if headers:
            lines.append(f"\n| {' | '.join(headers)} |")
            lines.append(f"| {' | '.join(['---'] * len(headers))} |")
        for row in element.find_all('tr'):
            cols = [td.get_text().strip() for td in row.find_all('td')]
            if cols:
                lines.append(f"| {' | '.join(cols)} |")

clean_md = '\n'.join(lines)
# Remove consecutive blank lines
clean_md = re.sub(r'\n{3,}', '\n\n', clean_md)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(clean_md)

print(f"Successfully wrote clean markdown docs to {output_path} (Size: {len(clean_md)} characters)")
