"""Create short-lived PDF fixtures for ingestion tests."""

from pathlib import Path


def create_two_page_pdf(path: Path, headings: bool = True, text: bool = True) -> None:
    import pymupdf

    document = pymupdf.open()
    first = document.new_page()
    if text:
        if headings:
            first.insert_text((72, 72), "Overview", fontsize=20, fontname="hebo")
            first.insert_text((72, 112), "First page paragraph for the overview section.", fontsize=11)
        else:
            first.insert_text((72, 72), "First plain paragraph without a heading.", fontsize=11)

    second = document.new_page()
    if text:
        if headings:
            second.insert_text((72, 32), "Second page continuation of the overview section.", fontsize=11)
            second.insert_text((72, 72), "Eligibility", fontsize=20, fontname="hebo")
            second.insert_text((72, 112), "Second section paragraph with useful details.", fontsize=11)
        else:
            second.insert_text((72, 72), "Second plain paragraph without a heading.", fontsize=11)
    document.save(path)
    document.close()


def create_encrypted_pdf(path: Path) -> bool:
    import pymupdf

    encryption = getattr(pymupdf, "PDF_ENCRYPT_AES_256", None)
    if encryption is None:
        return False
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Encrypted test PDF")
    document.save(path, encryption=encryption, owner_pw="owner", user_pw="user")
    document.close()
    return True