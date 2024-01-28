import docx

def read_text_from_txt(file):
    """Read text from a .txt file."""
    with open(file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def read_text_from_docx(file):
    """Read text from a .docx file."""
    doc = docx.Document(file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def read_text(file):
    """Read text from a file (either .txt or .docx)."""
    filename = file.name
    if filename.endswith('.txt'):
        return read_text_from_txt(file)
    elif filename.endswith('.docx'):
        return read_text_from_docx(file)
    else:
        raise ValueError("Unsupported file format. Only .txt and .docx files are supported.")