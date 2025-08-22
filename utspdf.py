# PDFextractor.py
# -*- coding: utf-8 -*-
"""
PDFextractor: Extract and clean text from PDFs (local file or URL).

Core functions:
- extract_pdf_text(...)        -> extract text (pdfminer) with optional OCR fallback
- clean_pdf_text(...)          -> robust cleaner for typical PDF artefacts
- postclean_pdf_text(...)      -> extra pass for spaced-out headings, barfalls, symbols
- aggressive_pdf_dejunk(...)   -> pragmatic "just make it readable" cleaner

Dependencies:
    pip install pdfminer.six requests
Optional (for OCR fallback on scanned PDFs):
    pip install pytesseract pdf2image pillow
System packages:
    Poppler (for pdf2image) and Tesseract OCR if using ocr=True
"""

from __future__ import annotations
import io
import os
import re
import tempfile
import unicodedata
from typing import Iterable, List, Optional, Union, Literal, Set
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# -------------------------------
# Extraction
# -------------------------------

def extract_pdf_text(
    source: Union[str, bytes, io.BytesIO],
    *,
    password: Optional[str] = None,
    ocr: bool = False,
    ocr_lang: str = "eng",
    per_page: bool = False,
    max_pages: Optional[int] = None,
) -> Union[str, List[str]]:
    """
    Extract text from a PDF given a local file path, a URL, or in-memory bytes.

    Parameters
    ----------
    source : str | bytes | io.BytesIO
        - Local path ('report.pdf'), URL ('https://.../paper.pdf'),
          raw PDF bytes, or a BytesIO object.
    password : str | None
        Password for encrypted PDFs (if needed).
    ocr : bool
        If True, run OCR on pages that appear image-only or empty.
        Requires: pytesseract, pdf2image, pillow, and Poppler/Tesseract on system.
    ocr_lang : str
        Tesseract language(s), e.g., 'eng', 'eng+fra'.
    per_page : bool
        If True, return a list of page strings; else a single concatenated string.
    max_pages : int | None
        If set, limit extraction to the first N pages.

    Returns
    -------
    str | List[str]
    """
    import requests
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LAParams

    def _is_url(s: str) -> bool:
        return bool(re.match(r"^https?://", s, flags=re.I))

    def _read_source_to_bytes(src: Union[str, bytes, io.BytesIO]) -> bytes:
        if isinstance(src, bytes):
            return src
        if isinstance(src, io.BytesIO):
            return src.getvalue()
        if isinstance(src, str) and _is_url(src):
            r = requests.get(src, timeout=60)
            r.raise_for_status()
            return r.content
        if isinstance(src, str) and os.path.exists(src):
            with open(src, "rb") as f:
                return f.read()
        raise FileNotFoundError("Could not read 'source' as path, URL, bytes, or BytesIO.")

    def _pdfminer_extract_per_page(pdf_bytes: bytes) -> List[str]:
        laparams = LAParams()
        pages_text: List[str] = []
        count = 0
        for page_layout in extract_pages(io.BytesIO(pdf_bytes), password=password or "", laparams=laparams):
            if (max_pages is not None) and (count >= max_pages):
                break
            buf = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    buf.append(element.get_text())
            pages_text.append("".join(buf).strip())
            count += 1
        return pages_text

    def _needs_ocr(page_text: str) -> bool:
        return len(page_text.strip()) < 10

    def _ocr_extract_per_page(pdf_bytes: bytes) -> List[str]:
        from pdf2image import convert_from_bytes  # requires Poppler
        import pytesseract
        from PIL import Image
        images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=max_pages or None)
        out = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang=ocr_lang)
            out.append(text.strip())
        return out

    pdf_bytes = _read_source_to_bytes(source)
    miner_pages = _pdfminer_extract_per_page(pdf_bytes)

    if ocr:
        if all(_needs_ocr(t) for t in miner_pages):
            pages = _ocr_extract_per_page(pdf_bytes)
        else:
            # OCR only the empty-ish pages, reuse rendered images once
            from pdf2image import convert_from_bytes
            import pytesseract
            images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=max_pages or None)
            pages: List[str] = []
            for i, text_already in enumerate(miner_pages):
                if _needs_ocr(text_already):
                    pages.append(pytesseract.image_to_string(images[i], lang=ocr_lang).strip())
                else:
                    pages.append(text_already)
    else:
        pages = miner_pages

    return pages if per_page else "\n\n".join(pages).strip()


# -------------------------------
# Cleaning – robust pass
# -------------------------------

def clean_pdf_text_simple(raw: str,
                   *,
                   normalise_unicode: bool = True,
                   collapse_whitespace: bool = True,
                   fix_hyphenation: bool = True,
                   fix_spaced_letters: bool = True,
                   fix_newline_broken_words: bool = True,
                   remove_page_numbers: bool = True,
                   remove_repeating_headers_footers: bool = True,
                   min_repeating_header_len: int = 6,
                   header_repeat_threshold: int = 3,
                   paragraph_join: bool = True) -> str:
    """
    Clean messy text extracted from PDFs.
    Targets: ligatures/odd spaces, page numbers, repeating headers/footers,
    hyphenation at line breaks, spaced-letter artefacts, and whitespace tidy.
    """
    txt = raw

    # 1) Unicode normalisation
    if normalise_unicode:
        txt = unicodedata.normalize('NFKC', txt)
        txt = txt.replace('\u00A0', ' ').replace('\u2009', ' ').replace('\u202F', ' ')

    # 2) Remove standalone page numbers
    if remove_page_numbers:
        lines = txt.splitlines()
        lines = [ln for ln in lines if not re.fullmatch(r'\s*\d{1,4}\s*', ln)]
        txt = "\n".join(lines)

    # 3) Remove repeating headers/footers
    if remove_repeating_headers_footers:
        lines = txt.splitlines()
        counter = Counter([ln.strip() for ln in lines if len(ln.strip()) >= min_repeating_header_len])
        repeated = {k for k, v in counter.items() if v >= header_repeat_threshold and len(k) <= 80}
        if repeated:
            lines = [ln for ln in lines if ln.strip() not in repeated]
            txt = "\n".join(lines)

    # 4) Fix hyphenation at line breaks: 'sustain-\nability' -> 'sustainability'
    if fix_hyphenation:
        txt = re.sub(r'(\w)-\n([a-z])', r'\1\2', txt)

    # 5) Fix newline-broken words without hyphen (conservative, insert space)
    if fix_newline_broken_words:
        txt = re.sub(r'([A-Za-z])\n([a-z])', r'\1 \2', txt)

    # 6) Collapse 'W o o l w o r t h s' -> 'Woolworths' and vertical stacks
    if fix_spaced_letters:
        def _collapse_spaced_letters(m: re.Match) -> str:
            return m.group(0).replace(' ', '')
        txt = re.sub(r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b', _collapse_spaced_letters, txt)
        # vertical single-letter lines (4+)
        txt = re.sub(
            r'(?m)(?:^(?:[A-Za-z]|[’\']|\&)\s*$\n){4,}',
            lambda m: ''.join([ln.strip() for ln in m.group(0).splitlines()]),
            txt
        )

    # 7) Normalise bullets/dashes
    txt = txt.replace('•', '- ').replace('●', '- ').replace('–', '-').replace('—', '-')

    # 8) Preserve paragraphs but join intra-paragraph single line breaks
    if paragraph_join:
        txt = re.sub(r'\n{2,}', '\n\n', txt)

        def _join_intra_para(block: str) -> str:
            lines = block.splitlines()
            out: List[str] = []
            for i, ln in enumerate(lines):
                if i > 0 and re.match(r'^\s*([-*]|\d+[\).])\s+', ln):
                    out.append('\n' + ln.strip())
                else:
                    out.append((' ' if i > 0 else '') + ln.strip())
            return ''.join(out)

        parts = txt.split('\n\n')
        parts = [_join_intra_para(p) for p in parts]
        txt = '\n\n'.join(parts)

    if collapse_whitespace:
        txt = re.sub(r'[ \t]+', ' ', txt)
        txt = re.sub(r'[ \t]+\n', '\n', txt)
        txt = txt.strip()

    return txt


# -------------------------------
# Cleaning – targeted post-pass
# -------------------------------

def postclean_pdf_text(
    txt: str,
    *,
    fix_spaced_out_headings: bool = True,
    remove_symbols: bool = True,
    collapse_hyphen_bars: bool = True,
    normalise_blank_lines: bool = True,
) -> str:
    """
    Aggressive post-clean for stubborn artefacts:
    - Spaced-out headings/words (e.g., 'Wo ow or th s ...')
    - Stray symbols (▲, ◆, etc.)
    - Hyphen/dash barfalls across many lines
    - Excessive blank lines
    """
    if remove_symbols:
        txt = re.sub(r"[▲■◆●□▪▫▶◀◼◻◽◾◆◇★☆❖✦✧✱✳︎✴︎✷✸✺✻✼✽●○•♦︎♣︎♠︎♥︎♢]", "", txt)

    if collapse_hyphen_bars:
        pattern = r"(?:\n[ \t]*-+[ \t]*){3,}\n?"
        txt = re.sub(pattern, "\n—\n", txt)

    if fix_spaced_out_headings:
        def _fix_spaced_heading_line(line: str) -> str:
            toks = line.split()
            if not toks:
                return line
            short_ratio = sum(len(t) <= 2 for t in toks) / max(1, len(toks))
            if short_ratio < 0.6:
                return line
            collapsed = re.sub(r"\s+(?=[A-Za-z])", "", line)
            collapsed = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", collapsed)
            collapsed = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", collapsed)
            return collapsed.strip()

        lines = txt.splitlines()
        fixed_lines = []
        for ln in lines:
            if re.search(r"[A-Za-z]", ln) and ln.count(" ") >= 6:
                fixed_lines.append(_fix_spaced_heading_line(ln))
            else:
                fixed_lines.append(ln)
        txt = "\n".join(fixed_lines)

        # Residual per-word spacing within a line
        txt = re.sub(r"\b(?:[A-Za-z]\s){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), txt)

    if normalise_blank_lines:
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = txt.strip()

    return txt


# -------------------------------
# Cleaning – pragmatic “just fix it”
# -------------------------------

def aggressive_pdf_dejunk(
    text: str,
    *,
    normalise_unicode: bool = True,
    drop_spaced_runs: bool = True,
    drop_all_caps_lines: bool = False,
    flatten_newlines: Literal["space","single"] = "space",
) -> str:
    """
    Strong cleaner for stubborn PDF artefacts.

    Default behaviour:
      - Remove long runs of 1–2 letter tokens (spaced-out headings/nav).
      - Remove isolated symbol lines and dash/hyphen 'barfalls'.
      - Flatten all newlines (so no \\n\\n remnants).
      - Normalise repeated '- -' sequences within lines.

    Set drop_spaced_runs=False to reconstruct those runs instead of dropping.
    """
    s = text

    if normalise_unicode:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")

    # remove barfalls & decorative lines
    s = re.sub(r'(?m)^(?:[ \t]*[–—\-•◦·]\s*){2,}$\n?', '', s)
    s = re.sub(r'(?:\n[ \t]*[–—\-•◦·][ \t]*){2,}', '\n', s)
    s = re.sub(r'[▲■◆▶◀◼◻◽◾◇★☆❖✦✧✱✳●○•♦]', '', s)

    # collapse multi hyphens inside lines
    s = re.sub(r'(?:\s*-\s*){2,}', ' - ', s)

    # spaced-letter runs
    spaced_run_pat = re.compile(r'(?:\b[A-Za-z]{1,2}\b(?:\s+)){8,}')
    if drop_spaced_runs:
        s = spaced_run_pat.sub(' ', s)
    else:
        def _rebuild(m: re.Match) -> str:
            chunk = m.group(0)
            joined = re.sub(r'\s+', '', chunk)
            joined = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', joined)
            joined = re.sub(r'(?<=[A-Za-z])(?=[A-Z][a-z])', ' ', joined)
            # a few optional soft fixes
            joined = re.sub(r'\bPeopel+l?\b', 'People', joined)
            joined = re.sub(r'\bSustanab', 'Sustainab', joined)
            joined = joined.replace('Overvewi', 'Overview')
            return joined
        s = spaced_run_pat.sub(_rebuild, s)

    if drop_all_caps_lines:
        s = re.sub(r'(?m)^[^a-z\n]{8,}$\n?', '', s)

    if flatten_newlines == "space":
        s = re.sub(r'\s*\n+\s*', ' ', s)
    else:
        s = re.sub(r'\n{2,}', '\n', s)

    s = re.sub(r'[ \t]{2,}', ' ', s).strip()
    return s



def clean_pdf_text(
    text: str,
    *,
    # --- from clean_pdf_text_simple ---
    normalise_unicode: bool = True,
    collapse_whitespace: bool = True,
    fix_hyphenation: bool = True,
    fix_spaced_letters: bool = True,
    fix_newline_broken_words: bool = True,
    remove_page_numbers: bool = True,
    remove_repeating_headers_footers: bool = True,
    min_repeating_header_len: int = 6,
    header_repeat_threshold: int = 3,
    paragraph_join: bool = True,

    # --- from postclean_pdf_text ---
    fix_spaced_out_headings: bool = True,
    remove_symbols: bool = True,
    collapse_hyphen_bars: bool = True,
    normalise_blank_lines: bool = True,

    # --- from aggressive_pdf_dejunk ---
    drop_spaced_runs: bool = True,                 # if True, drop long spaced-letter runs outright
    drop_all_caps_lines: bool = False,
    flatten_newlines: Literal["none","space","single"] = "none",  # “space” trumps paragraph_join
) -> str:
    """
    Unified PDF text cleaner that merges:
      - clean_pdf_text
      - postclean_pdf_text
      - aggressive_pdf_dejunk

    ORDER OF OPERATIONS (deduplicated):
      1) Unicode normalisation and space fixes
      2) Remove page numbers & repeating headers/footers
      3) Remove decorative symbol/dash 'barfalls' and collapse multi-hyphens in-line
      4) Fix hyphenation at line breaks and newline-broken words
      5) Repair 'W o o l w o r t h s' per-word spacing
      6) Handle 'spaced-out heading lines' (choose: fix vs drop vs keep)
      7) Optional: drop ALL-CAPS lines
      8) Normalise bullets/dashes (en/em → '-')
      9) Line/paragraph handling:
         - if flatten_newlines == "space": ALL newlines → space
         - elif "single": 2+ newlines → 1 blank line
         - else if paragraph_join: keep blank lines, but join single linewraps inside paragraphs
      10) Blank-line and whitespace tidying

    PARAMETER INTERACTIONS:
      - If drop_spaced_runs=True, long spaced-out runs are removed and
        fix_spaced_out_headings is ignored for those lines.
      - If flatten_newlines != "none", it takes precedence over paragraph_join.
    """
    import re, unicodedata
    from collections import Counter

    s = text

    # 1) Unicode normalisation + odd spaces
    if normalise_unicode:
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")

    # 2) Remove page numbers & repeating headers/footers
    if remove_page_numbers:
        lines = s.splitlines()
        lines = [ln for ln in lines if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
        s = "\n".join(lines)

    if remove_repeating_headers_footers:
        lines = s.splitlines()
        counts = Counter(ln.strip() for ln in lines if len(ln.strip()) >= min_repeating_header_len)
        repeated = {k for k, v in counts.items() if v >= header_repeat_threshold and len(k) <= 80}
        if repeated:
            lines = [ln for ln in lines if ln.strip() not in repeated]
            s = "\n".join(lines)

    # 3) Remove decorative symbol lines / barfalls; collapse multi-hyphens in-line
    if remove_symbols:
        # Kill isolated decorative symbol lines quickly
        s = re.sub(r'[▲■◆▶◀◼◻◽◾◇★☆❖✦✧✱✳︎✴︎●○•♦]', '', s)

    if collapse_hyphen_bars:
        # Multi-line cascades like \n-\n-\n- or variants with en/em dashes / bullets
        s = re.sub(r'(?m)^(?:[ \t]*[–—\-•◦·]\s*){2,}$\n?', '', s)              # lines that are only dashes/bullets
        s = re.sub(r'(?:\n[ \t]*[–—\-•◦·][ \t]*){2,}', '\n', s)                 # repeated dash-only lines merged

    # Inside-line: collapse repeated "- - -"
    s = re.sub(r'(?:\s*-\s*){2,}', ' - ', s)

    # 4) Fix hyphenation/newline-broken words
    if fix_hyphenation:
        s = re.sub(r'(\w)-\n([a-z])', r'\1\2', s)                               # sustain-\nability → sustainability
    if fix_newline_broken_words:
        s = re.sub(r'([A-Za-z])\n([a-z])', r'\1 \2', s)                         # sustain\nability → sustain ability

    # 5) Collapse per-word spaced letters: "W o o l w o r t h s" → "Woolworths"
    if fix_spaced_letters:
        s = re.sub(r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b', lambda m: m.group(0).replace(' ', ''), s)
        # vertical single-letter stacks (4+ lines)
        s = re.sub(
            r'(?m)(?:^(?:[A-Za-z]|[’\']|\&)\s*$\n){4,}',
            lambda m: ''.join(ln.strip() for ln in m.group(0).splitlines()),
            s
        )

    # 6) Handle "spaced-out heading lines" (many short tokens). Choose: drop vs fix vs keep.
    spaced_run_pat = re.compile(r'(?:\b[A-Za-z]{1,2}\b(?:\s+)){8,}')
    def _looks_spaced_heading_line(line: str) -> bool:
        toks = line.split()
        if not toks:
            return False
        short_ratio = sum(len(t) <= 2 for t in toks) / max(1, len(toks))
        return (line.count(" ") >= 6) and (short_ratio >= 0.6)

    lines = s.splitlines()
    new_lines = []
    for ln in lines:
        if _looks_spaced_heading_line(ln):
            if drop_spaced_runs:
                # Drop the line entirely
                continue
            elif fix_spaced_out_headings:
                # Reconstruct by removing intra-letter spaces then inserting camel boundaries
                collapsed = re.sub(r"\s+(?=[A-Za-z])", "", ln)
                collapsed = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", collapsed)
                collapsed = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", collapsed)
                new_lines.append(collapsed.strip())
            else:
                new_lines.append(ln)  # keep as-is
        else:
            new_lines.append(ln)
    s = "\n".join(new_lines)

    # 7) Optionally drop ALL-CAPS lines
    if drop_all_caps_lines:
        s = re.sub(r'(?m)^[^a-z\n]{8,}$\n?', '', s)

    # 8) Normalise bullets/dashes (unify en/em to '-')
    s = s.replace('•', '- ').replace('●', '- ').replace('–', '-').replace('—', '-')

    # 9) Newline/paragraph handling
    if flatten_newlines == "space":
        # Replace ANY run of newlines with a single space
        s = re.sub(r'\s*\n+\s*', ' ', s)
    elif flatten_newlines == "single":
        s = re.sub(r'\n{2,}', '\n', s)  # reduce big gaps to single blank line
    else:
        # paragraph_join behaviour: keep blank lines, but join linewraps inside paragraphs
        if paragraph_join:
            s = re.sub(r'\n{2,}', '\n\n', s)  # normalise huge gaps first

            def _join_intra_para(block: str) -> str:
                lines = block.splitlines()
                out: list[str] = []
                for i, ln in enumerate(lines):
                    # preserve true lists starting with bullets/numbers on new line
                    if i > 0 and re.match(r'^\s*([-*]|\d+[\).])\s+', ln):
                        out.append('\n' + ln.strip())
                    else:
                        out.append((' ' if i > 0 else '') + ln.strip())
                return ''.join(out)

            parts = s.split('\n\n')
            parts = [_join_intra_para(p) for p in parts]
            s = '\n\n'.join(parts)

    # 10) Blank-line and whitespace tidying
    if normalise_blank_lines:
        s = re.sub(r'\n{3,}', '\n\n', s)
        s = re.sub(r'[ \t]+\n', '\n', s)
    if collapse_whitespace:
        s = re.sub(r'[ \t]{2,}', ' ', s).strip()

    return s



__all__ = [
    "extract_pdf_text",
    # old cleaners may remain for backwards compatibility if you like:
    "clean_pdf_text_simple",
    "postclean_pdf_text",
    "aggressive_pdf_dejunk",
    # new unified cleaner:
    "clean_pdf_text",
]

























#############################################
# Plotting wordclouds
#############################################
def show_wordcloud(
    text: str,
    *,
    max_words: int = 200,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    colormap: str = "viridis",
    stopwords: Optional[set[str]] = None,
    title: Optional[str] = None,
    include_stopwords: bool = False,          # if True, do NOT remove stopwords
    use_nltk_stopwords: bool = False          # optionally add NLTK stopwords when removing
) -> None:
    """
    Display a simple wordcloud from text.

    Parameters
    ----------
    text : str
        Input text (already extracted & cleaned).
    max_words : int
        Maximum words to display.
    width, height : int
        Dimensions of the figure.
    background_color : str
        Background colour ('white', 'black', etc.).
    colormap : str
        Matplotlib colormap for colouring words.
    stopwords : set[str] | None
        Optional *additional* stopwords to exclude (used only when include_stopwords=False).
    title : str | None
        Optional title for the plot.
    include_stopwords : bool
        If True, do not remove stopwords (i.e., include them in the cloud). Default False.
    use_nltk_stopwords : bool
        If True and include_stopwords=False, union NLTK English stopwords as well.
    """
    # Build the stopword set
    stopset: Set[str] = set()
    if not include_stopwords:
        stopset |= set(STOPWORDS)  # WordCloud's default list
        if use_nltk_stopwords:
            try:
                from nltk.corpus import stopwords as nltk_sw
                stopset |= set(nltk_sw.words("english"))
            except Exception:
                # NLTK stopwords not available; silently continue
                pass
        if stopwords:
            stopset |= set(stopwords)

    wc = WordCloud(
        max_words=max_words,
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        stopwords=stopset  # empty set => include everything; non-empty => remove
    ).generate(text)

    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=12)
    plt.show()