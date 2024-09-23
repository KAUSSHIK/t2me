import re
import PyPDF2

def load_and_clean_text(file_path):
    if file_path.lower().endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

if __name__ == "__main__":
    resume_text = load_and_clean_text('resume.pdf')
    linkedin_text = load_and_clean_text('linkedin_resume.pdf')
    
    # Combine texts
    combined_text = resume_text + " " + linkedin_text
    
    # Save the combined text
    with open('combined_text.txt', 'w', encoding='utf-8') as file:
        file.write(combined_text)