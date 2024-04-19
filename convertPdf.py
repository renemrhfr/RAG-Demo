import os
from pypdf import PdfReader

files_to_process = []


def collectfiles():
    """
    Collects files from documents/pdf and adds them to files_to_process
    """
    for file in os.listdir(os.path.join('documents', 'pdf')):
        full_path = os.path.join(os.path.join('documents', 'pdf'), file)
        if os.path.isfile(full_path):
            files_to_process.append(full_path)


def extract_text_from_pdf(pdf_path):
    """
    Extracts the text from a PDF File and writes it into a .txt file located in /documents/export
    """
    filename = os.path.basename(pdf_path)
    filename_without_extension, _ = os.path.splitext(filename)
    output_file_path = os.path.join('documents', 'export', filename_without_extension + '.txt')
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() or ''
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
        return full_text
    except FileNotFoundError:
        print(f"The file {pdf_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    collectfiles()
    for file in files_to_process:
        extract_text_from_pdf(file)
