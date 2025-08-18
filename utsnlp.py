#############################################
# Keep this
#############################################
import sys
import subprocess

def install_packages(element):
    try:
        print(f"Installing package \"{element}\"...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", element]) 
        print(f"DONE: Package {element} is up to date.")
    except:
        print(f"ERROR: Unable to download \"{element}\" package, please check your internet connection or package name spelling and try again.")
    return None

try:
    import pylibcheck
    print(f"Package pylibcheck is installed and loaded.")
except:
    print(f"Installing pylibcheck... Package pylibcheck is installed.")
    install_packages("pylibcheck")
    import pylibcheck



#############################################
# Check if libraries are installed
#############################################
# Add additional libraries to check in the list below:
packages_list = ["numpy",
                "Wikipedia-API",
                "pandas",
                "matplotlib",
                "seaborn",
                "contractions",
                "nltk",
                "wordcloud",
                "plotly",
                "d3blocks",
                "bs4"]    

def check_packages(packages_list):
    for element in packages_list:
        if pylibcheck.checkPackage(element):
            print(f"OK: Package {element} is installed.")
        else:
            install_packages(element)
    return None

check_packages(packages_list)

#############################################
# Load libraries
#############################################
import wikipediaapi

import numpy as np
from PIL import Image

import re, unicodedata
import string
import contractions
import itertools

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot

import math
from collections import Counter, defaultdict
from typing import Iterable, Optional, Tuple, List, Dict, Literal
import pandas as pd
from d3blocks import D3Blocks

import requests
from bs4 import BeautifulSoup

# When editing a module, and not wanting to restatrt kernel every time use:
# import importlib
# importlib.reload(bc)
# import utsbootcamp as bc


#############################################
# Functions
#############################################
def get_text_from_url(u):
    r = requests.get(u, timeout=12)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Simple readability: join paragraph text
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    txt = "\n\n".join(paras)
    # Light cleanup
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def download_wikipedia_text(topic, 
                            language='en', 
                            user_agent='MyWikiApp/1.0', 
                            summary=False, 
                            include_url=False, 
                            save_to_file=False, 
                            file_name=None, 
                            include_sections=False, 
                            include_links=False, 
                            include_categories=False, 
                            include_languages=False):
    """
    Downloads text from a Wikipedia page on a specified topic with additional options.

    Parameters:
    - topic (str): The title of the Wikipedia page to download.
    - language (str, optional): The language of the Wikipedia page (default is 'en' for English).
    - user_agent (str, optional): The user agent to use for the Wikipedia API request (default is 'MyWikiApp/1.0').
    - summary (bool, optional): If True, return a summary of the page instead of the full text (default is False).
    - include_url (bool, optional): If True, include the URL of the Wikipedia page in the output (default is False).
    - save_to_file (bool, optional): If True, save the text or summary to a file (default is False).
    - file_name (str, optional): The name of the file to save the content to. If not specified, a file name is generated based on the topic.
    - include_sections (bool, optional): If True, include the sections of the Wikipedia page in the output (default is False).
    - include_links (bool, optional): If True, include the links from the Wikipedia page in the output (default is False).
    - include_categories (bool, optional): If True, include the categories of the Wikipedia page in the output (default is False).
    - include_languages (bool, optional): If True, include the available languages for the Wikipedia page in the output (default is False).

    Returns:
    - str: The text or summary of the Wikipedia page with additional information as specified, or an error message if the page does not exist. If 'save_to_file' is True, returns a message indicating the file where the content is saved.

    Example usage:
    >>> topic = "Machine learning"
    >>> options = {
    ...     'summary': True,
    ...     'include_url': True,
    ...     'include_sections': True,
    ...     'include_links': True,
    ...     'include_categories': True,
    ...     'include_languages': True,
    ...     'save_to_file': True,
    ...     'file_name': 'machine_learning_info.txt'
    ... }
    >>> text = download_wikipedia_text(topic, **options)
    >>> print(text)
    
    The ** operator is used to unpack a dictionary into keyword arguments when calling a function. 
    When you see **options in the function call, it means that the options dictionary is being unpacked and 
    its key-value pairs are passed as keyword arguments to the function.
    
    """
    # Create a Wikipedia API instance with the specified user agent
    wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)

    # Get the page for the specified topic
    page = wiki_wiki.page(topic)

    # Check if the page exists
    if page.exists():
        content = ""

        # Add summary or full text
        content += page.summary if summary else page.text

        # Add sections
        if include_sections:
            content += "\n\nSections:\n" + "\n".join([section.title for section in page.sections])

        # Add links
        if include_links:
            content += "\n\nLinks:\n" + "\n".join([link for link in page.links])

        # Add categories
        if include_categories:
            content += "\n\nCategories:\n" + "\n".join([category for category in page.categories])

        # Add languages
        if include_languages:
            content += "\n\nLanguages:\n" + "\n".join([lang for lang in page.langlinks])

        # Add URL
        if include_url:
            content += '\n\nURL: ' + page.fullurl

        # Save to file
        if save_to_file:
            file_name = file_name or f"{topic.replace(' ', '_')}.txt"
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(content)
            return f"Content saved to file: {file_name}"

        # Return content
        return content
    else:
        # Return an error message if the page doesn't exist
        return f"The page for the topic '{topic}' does not exist on Wikipedia."
		
def find_text_elements(text,
                       pattern=r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|\$[^$]*\$|<[^>]*>',
                       remove=False):
    """
    Finds or removes specific text elements from the input text based on a regular expression pattern.

    Parameters:
    - text (str):              The input text from which to find or remove elements.
    - pattern (str, optional): The regular expression pattern used to identify the text elements. 
                               The default pattern matches text between round brackets (parentheses), 
                               square brackets, curly brackets, $ signs for LaTeX expressions, 
                               and < and > signs for HTML tags, without nesting.
                               Default pattern: r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|\$[^$]*\$|<[^>]*>'
                               where:
                               \( [^()]* \)    : Matches text between round brackets (parentheses) without nesting.
                               \[ [^\[\]]* \]  : Matches text between square brackets without nesting.
                               \{ [^{}]* \}    : Matches text between curly brackets without nesting.
                               \$ [^$]* \$     : Matches text between $ signs for LaTeX expressions.
                               < [^>]* >       : Matches text between < and > signs for HTML tags.
    - remove (bool, optional): If True, the identified text elements are removed from the input text. 
                               If False, the identified text elements are returned as a list. 
                               Default value is False.

    Returns:
    - If remove is False, returns a list of matches found in the input text based on the pattern.
    - If remove is True, returns the input text with the identified elements removed.

    Examples:
    >>> text = "This is a sample text with (parentheses), [brackets], {curly brackets}, $LaTeX$ expression, and <html> tags."
    >>> find_text_elements(text)
    ['(parentheses)', '[brackets]', '{curly brackets}', '$LaTeX$', '<html>']
    
    >>> find_text_elements(text, remove=True)
    'This is a sample text with , , , , and .'

    Note:
    - The default pattern is designed to match specific text elements without nesting. 
      If you need to handle nested structures, you will need to modify the pattern accordingly.
    - The regular expression patterns can be customized to match different text elements 
      as per the requirements of your application.
    """    
   
    if remove:
        text = re.sub(pattern, '', text)
        return text
    else:
        matches = re.findall(pattern, text)
        return matches

def remove_text_inside_brackets(text, brackets='''{}()[]'''):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def plot_top_words(text, n=20):
    tokens = [w.lower() for w in text.split() if w.isalpha()]
    counts = Counter(tokens).most_common(n)
    words, freqs = zip(*counts)

    plt.figure(figsize=(10,5))
    plt.bar(words, freqs)
    plt.xticks(rotation=45)
    plt.title(f"Top {n} Words")
    plt.show()

    
def preprocess_text(text, 
                    remove_text_in_brackets=True,
                    remove_url=True,
                    remove_html_tags=True,
                    to_lower=True, 
                    expand_contractions=True,
                    remove_non_english=True,                
                    non_english_strategy: str = "drop",       #  "drop" | "transliterate"
                    remove_punctuation=True, 
                    remove_digits=True, 
                    remove_stopwords=True, 
                    remove_short_words_leq: int | None = 2,
                    lemmatize=True, 
                    stem=False, 
                    custom_stopwords=None,
                    custom_brackets=None):
    """
    Preprocesses and cleans text with various options.
    
    This function provides options for converting text to lowercase, removing punctuation, removing digits, 
    removing stopwords, applying lemmatization, applying stemming, and specifying custom stopwords. 
    You can adjust these options according to your needs by setting the corresponding parameters to True or False. 
    For example, if you want to keep the punctuation, you can call the function with 'remove_punctuation=False'.

    Parameters:
    - text (str): The text to be preprocessed.
    - to_lower (bool, optional): Convert text to lowercase (default is True).
    - remove_punctuation (bool, optional): Remove punctuation from text (default is True).
    - remove_digits (bool, optional): Remove digits from text (default is True).
    - remove_stopwords (bool, optional): Remove stopwords from text (default is True).
    - lemmatize (bool, optional): Apply lemmatization to words (default is True).
    - stem (bool, optional): Apply stemming to words (default is False).
    - custom_stopwords (list, optional): A list of custom stopwords to remove (default is None).

    Returns:
    - str: The preprocessed and cleaned text.

    Example usage:
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> cleaned_text = preprocess_text(text)
    >>> print(cleaned_text)
    """
    
    # Remove brackets 
    if remove_text_in_brackets:
        if custom_brackets:
            text=remove_text_inside_brackets(text, brackets=custom_brackets)
        else:
            text=remove_text_inside_brackets(text)
            
    
    # Remove URLs
    if remove_url:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
    # Remove HTML tags
    if remove_html_tags:
        text = re.sub(r'<.*?>', '', text)
        
    # Expanding Contractions
    if expand_contractions:
        text = contractions.fix(text)
        
    # Convert text to lowercase
    if to_lower:
        text = text.lower()

    # Remove/transliterate non-English characters BEFORE punctuation removal
    if remove_non_english:
        if non_english_strategy.lower() == "transliterate":
            try:
                from unidecode import unidecode
                text = unidecode(text)
            except Exception:
                # Fallback: strip diacritics via NFKD then drop non-ASCII
                text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        else:
            # "drop": keep only ASCII (letters, digits, ASCII punctuation/space)
            text = ''.join(ch for ch in text if ch.isascii())
            
    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        tokens = [word for word in tokens if word not in stop_words]
    
    # Remove short tokens (length <= threshold)
    if remove_short_words_leq and remove_short_words_leq > 0:
        tokens = [w for w in tokens if len(w) > remove_short_words_leq]
        
    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Apply lemmatization
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Apply stemming
    if stem:
        tokens = [stemmer.stem(word) for word in tokens]

    # ... after you finish building `tokens` (and any lemmatising/stemming) ...
    def _detok(tokens):
        return TreebankWordDetokenizer().detokenize(tokens)

    def smart_detokenize(tokens: list[str]) -> str:
        s = TreebankWordDetokenizer().detokenize(tokens)
        # Normalise unicode (turn NBSP etc. into regular form)
        s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    
        # --- Glue punctuation to the left (fix "word ." -> "word.")
        s = re.sub(r"\s+([,.;:?!%])", r"\1", s)
    
        # --- Glue closing brackets/quotes to the left
        s = re.sub(r"\s+([)\]\}])", r"\1", s)
    
        # --- Remove space after opening brackets/quotes
        s = re.sub(r"([(\[\{])\s+", r"\1", s)
    
        # --- Currency: "$ 3" -> "$3"
        s = re.sub(r"([£$€])\s+(\d)", r"\1\2", s)
    
        # --- Possessives: "Sydney 's" -> "Sydney's"
        s = re.sub(r"\s+'s\b", r"'s", s)
    
        # --- Collapse leftover multiple spaces
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s
    
    # Rejoin tokens into a single string
    #cleaned_text = ' '.join(tokens) 
    
    # this OLD step results in seeing the extra spaces because word_tokenize splits punctuation into separate tokens, and then ' '.join(tokens) naively inserts a space between every token. 
    # Use NLTK’s detokeniser to re-assemble tokens with correct punctuation spacing and quotes:
    #cleaned_text = _detok(tokens)

    # Even the 2nd attem pt didn't work --> making "smart" detokenizer
    cleaned_text = smart_detokenize(tokens)

    return cleaned_text
	
def plot_wordcloud(text,
                   width=3000, height=2000, 
                   background_color='salmon', 
                   colormap='Pastel1', 
                   collocations=False, 
                   stopwords=None,
                   figsize=(18, 14),
                   mask=None,
                   min_word_length=0,
                   include_numbers=False):
    """
    Generates and plots a word cloud from the input text.

    Parameters:
    - text (str): The input text for generating the word cloud.
    - width (int, optional): Width of the word cloud image. Default is 3000.
    - height (int, optional): Height of the word cloud image. Default is 2000.
    - background_color (str, optional): Background color of the word cloud image. Default is 'salmon'.
    - colormap (str, optional): Colormap for coloring the words. Default is 'Pastel1'.
    - collocations (bool, optional): Whether to include collocations (bigrams) in the word cloud. Default is False.
    - stopwords (set, optional): Set of stopwords to exclude from the word cloud. Default is STOPWORDS from wordcloud.
    - figsize (tuple, optional): Size of the figure for plotting the word cloud. Default is (40, 30).
    - mask (string, optional): Path and file name to masking image file
    
    Returns:
    - None
    
    Example usage:
    text = "Python is a great programming language for data analysis and visualization. Python is popular for data science."
    plot_wordcloud(text,stopwords=['is','a'])

    """
    if mask:
        # Import image to np.array
        mask = np.array(Image.open(mask))

    # Generate word cloud
    wordcloud = WordCloud(width=width, height=height, 
                          background_color=background_color, 
                          colormap=colormap, 
                          collocations=collocations, 
                          stopwords=stopwords,
                          mask=mask,
                          min_word_length=min_word_length,
                          include_numbers=include_numbers).generate(text)

    # Plot word cloud
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def print_colored_text(text_data):
    """
    Prints elements of a list in different colors.

    This function takes a single argument, 'text_data', which can be either a list of strings or a list of lists of strings.
    It prints each element in 'text_data' in a different color, cycling through a predefined set of colors. If 'text_data' is
    a list of lists, each sublist is printed in a single color.

    Parameters:
    - text_data (list): A list of strings or a list of lists of strings to be printed in different colors.

    Usage:
    - print_colored_text(["string1", "string2", "string3"])
    - print_colored_text([["word1", "word2"], ["word3", "word4"], ["word5", "word6"]])
    
    Or:
    data = ["string1", "string2", "string3", "string4"]
    data_words = [["word1", "word2"], ["word3", "word4"], ["word5", "word6"], ["word7", "word8"]]
    print_colored_text(data)
    print_colored_text(data_words)
    """
   
    # Define colors
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'reset': '\033[0m'
    }

    # Create a cycle of colors
    color_cycle = itertools.cycle(colors.values())

    # Check if 'text_data' is a list of lists of strings
    if all(isinstance(item, list) for item in text_data):
        # 'text_data' is a list of lists of strings
        print("Data Type: a list of lists of strings:")
        for sublist in text_data[:4]:
            color = next(color_cycle)
            print(f"{color}[", end="")
            for word in sublist:
                print(f"{word} ", end="")
            print(f"]{colors['reset']}")
    elif all(isinstance(item, str) for item in text_data):
        # 'text_data' is a list of strings
        print("Data Type: a list of strings")
        for item in text_data[:4]:
            color = next(color_cycle)
            print(f"{color}{item}{colors['reset']}")
    else:
        print("Invalid input: 'text_data' must be a list of strings or a list of lists of strings.")
    print('\n')











#############################################
# D3graphs Functions
#############################################
def plot_semantic_network_d3(
    text: str,
    *,
    # --- construction mode ---
    mode: Literal["adjacent", "window", "dependency"] = "window",
    n: int = 2,                          # used only if mode="adjacent" (2=bigram, 3=trigram)
    window_size: int = 5,                # used only if mode="window": symmetric skip-gram window
    # --- linguistic normalisation ---
    use_spacy: bool = True,              # enable lemmatisation / POS / NER if available
    spacy_model: str = "en_core_web_sm",
    lowercase: bool = True,
    pos_keep: Tuple[str, ...] = ("NOUN", "PROPN", "ADJ", "VERB"),  # kept POS when use_spacy=True
    keep_entities: bool = True,          # merge named entities to single tokens (United_Nations)
    remove_short_leq: int = 2,           # drop tokens with length <= this
    stopwords: Optional[Iterable[str]] = None,
    token_pattern: str = r"[A-Za-z']+",  # regex fallback tokenizer when use_spacy=False
    # --- weighting / pruning ---
    weight: Literal["freq", "pmi", "ppmi"] = "ppmi",
    top_k_edges: int = 200,
    min_count: int = 2,
    # --- visual output ---
    filepath: str = "semantic_network.html",
    title: Optional[str] = None,
    showfig: bool = True,
    notebook: bool = False,
    scaler: str = "minmax",
    dark_mode: bool = False,
) -> str:
    """
    Build an interactive *semantic* network with D3Blocks.

    Modes
    -----
    - 'adjacent':   edges from adjacent n-grams (n=2 bigrams, n=3 trigrams as (w1 w2)->w3)
    - 'window':     symmetric skip-gram within `window_size` (captures topical co-occurrence)
    - 'dependency': dependency edges from spaCy (amod, compound, nsubj, dobj, pobj, attr)

    Weighting
    ---------
    - 'freq': counts
    - 'pmi':  pointwise mutual information
    - 'ppmi': positive PMI = max(PMI, 0)  (recommended default)

    Notes
    -----
    - For 'dependency' mode, spaCy is required.
    - For 'adjacent' n=3, left node is "w1 w2" -> right node "w3".
    """
    # ---------- 0) stopwords / helpers ----------
    sw = set(w.lower() for w in (stopwords or []))
    def _is_ok_token(tok_text: str) -> bool:
        return (tok_text and
                (not lowercase or tok_text == tok_text.lower()) and
                (remove_short_leq <= 0 or len(tok_text) > remove_short_leq) and
                (tok_text not in sw))

    # ---------- 1) linguistic preprocessing ----------
    doc_tokens: List[str] = []
    ents_to_merge: List[Tuple[int, int, str]] = []  # (start, end, merged_text) for spaCy

    if use_spacy or mode == "dependency":
        try:
            import spacy
            nlp = spacy.load(spacy_model, disable=["tagger","lemmatizer","ner","attribute_ruler","textcat","senter","morphologizer","tok2vec"])
            # enable only what we need for speed
            nlp.enable_pipe("tok2vec")
            nlp.add_pipe("senter")
        except Exception:
            # fallback: load full model if minimal fails
            import spacy
            nlp = spacy.load(spacy_model)
        doc = nlp(text)

        # Optionally run NER (merge entities)
        if keep_entities and not doc.has_annotation("ENT_IOB"):
            # If we disabled NER above, re-run on a lightweight pipeline
            try:
                import spacy
                nlp_ner = spacy.load(spacy_model)  # full
                doc = nlp_ner(text)
            except Exception:
                pass

        # Merge entities into single tokens
        if keep_entities and doc.has_annotation("ENT_IOB"):
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if ent_text and len(ent_text.split()) > 1:
                    ents_to_merge.append((ent.start, ent.end, "_".join(ent_text.split())))

        # Build tokens with lemma/POS filtering
        keep_span = [True] * len(doc)
        for (s, e, merged) in ents_to_merge:
            # mark entity internal tokens as dropped, keep head with merged label
            for i in range(s+1, e):
                keep_span[i] = False

        for i, tok in enumerate(doc):
            if not keep_span[i]:
                continue
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            # use merged entity label for entity heads
            if keep_entities and any(s == i for (s, e, merged) in ents_to_merge):
                t = next(m for (s, e, m) in ents_to_merge if s == i)
                lemma = t
                pos = "PROPN"
            else:
                lemma = tok.lemma_.lower() if lowercase else tok.lemma_
                pos = tok.pos_
            if pos_keep and pos not in pos_keep:
                continue
            if remove_short_leq > 0 and len(lemma) <= remove_short_leq:
                continue
            if lemma in sw:
                continue
            doc_tokens.append(lemma)
    else:
        # regex fallback
        toks = re.findall(token_pattern, text)
        toks = [t.lower() for t in toks] if lowercase else toks
        for t in toks:
            if remove_short_leq > 0 and len(t) <= remove_short_leq:
                continue
            if t in sw:
                continue
            doc_tokens.append(t)

    if not doc_tokens:
        raise ValueError("No tokens to build network. Relax filters or disable POS/NER.")

    # ---------- 2) edge construction ----------
    edges: List[Tuple[str, str]] = []

    if mode == "adjacent":
        assert n >= 2, "For 'adjacent' mode, use n >= 2."
        if n == 2:
            for i in range(len(doc_tokens) - 1):
                edges.append((doc_tokens[i], doc_tokens[i+1]))
        else:
            for i in range(len(doc_tokens) - (n - 1)):
                left = " ".join(doc_tokens[i:i+n-1])
                right = doc_tokens[i+n-1]
                edges.append((left, right))

    elif mode == "window":
        W = max(2, window_size)
        # symmetric, unordered edges within a window; use sorted pair to make undirected
        for i in range(len(doc_tokens)):
            wi = doc_tokens[i]
            for j in range(i+1, min(len(doc_tokens), i+W)):
                wj = doc_tokens[j]
                if wi == wj:
                    continue
                a, b = (wi, wj) if wi < wj else (wj, wi)
                edges.append((a, b))

    elif mode == "dependency":
        # requires spaCy doc
        if not (use_spacy or mode == "dependency"):
            raise ValueError("Dependency mode requires spaCy.")
        try:
            import spacy
        except Exception:
            raise ValueError("spaCy not installed. Install and load a model to use dependency mode.")
        # We need the parsed doc; rebuild quickly to ensure deps exist
        import spacy
        nlp_dep = spacy.load(spacy_model)
        doc_dep = nlp_dep(text)

        RELS = {"amod", "compound", "nsubj", "dobj", "pobj", "attr"}
        def ok_token(tok):
            if tok.is_space or tok.is_punct or tok.like_num: return False
            lemma = tok.lemma_.lower() if lowercase else tok.lemma_
            if pos_keep and tok.pos_ not in pos_keep: return False
            if remove_short_leq > 0 and len(lemma) <= remove_short_leq: return False
            if lemma in sw: return False
            return True

        def lemma_of(tok):
            return tok.lemma_.lower() if lowercase else tok.lemma_

        for tok in doc_dep:
            if not ok_token(tok): 
                continue
            h = tok.head
            if tok.dep_ in RELS and ok_token(h):
                edges.append((lemma_of(h), lemma_of(tok)))
            # optionally collapse prep->pobj (head is PREP, child is pobj)
            if tok.dep_ == "prep":
                for ch in tok.children:
                    if ch.dep_ == "pobj" and ok_token(ch):
                        # link head of PREP to the pobj directly
                        if ok_token(tok.head):
                            edges.append((lemma_of(tok.head), lemma_of(ch)))
    else:
        raise ValueError("Unknown mode. Use 'adjacent', 'window', or 'dependency'.")

    if not edges:
        raise ValueError("No edges formed. Try a different mode or relax filters.")

    # ---------- 3) counts & weighting ----------
    edge_counts = Counter(edges)

    # unigrams for PMI
    unigram = Counter()
    for a, b in edges:
        unigram[a] += 1
        unigram[b] += 1

    items: List[Tuple[str, str, float, int]] = []  # (src, dst, weight_value, raw_count)
    if weight == "freq":
        for (s, t), c in edge_counts.items():
            items.append((s, t, float(c), c))

    else:
        # PMI / PPMI
        N_pairs = sum(edge_counts.values()) or 1
        N_tok = sum(unigram.values()) or 1
        for (s, t), c in edge_counts.items():
            p_uv = c / N_pairs
            p_u = unigram[s] / N_tok
            p_v = unigram[t] / N_tok
            if p_u <= 0 or p_v <= 0:
                continue
            pmi = math.log2(max(p_uv / (p_u * p_v), 1e-12))
            val = pmi if weight == "pmi" else max(pmi, 0.0)
            items.append((s, t, val, c))

    # prune by min_count first, then by weight
    items = [it for it in items if it[3] >= min_count]
    if not items:
        raise ValueError("No edges survived 'min_count'. Lower it or change mode/filters.")
    items.sort(key=lambda x: x[2], reverse=True)
    items = items[:top_k_edges]

    # ---------- 4) build D3Blocks graph ----------
    df_edges = pd.DataFrame([(s, t, w) for (s, t, w, c) in items], columns=["source", "target", "weight"])
    d3 = D3Blocks()
    d3.d3graph(
        df_edges,
        scaler=scaler,
        dark_mode=dark_mode,
        title=title or f"{mode.capitalize()} network (weight={weight})",
        filepath=filepath,
        showfig=showfig,
        notebook=notebook
    )
    return filepath, df_edges


def plot_ngram_network_d3(
    text: str,
    *,
    n: int = 2,                       # 2=bigram, 3=trigram (uses adjacent tokens)
    top_k_edges: int = 50,            # keep N most frequent edges
    min_count: int = 2,               # drop rare edges
    stopwords: Optional[Iterable[str]] = None,
    token_pattern: str = r"[A-Za-z']+",  # simple regex tokenizer
    lowercase: bool = True,
    filepath: str = "ngram_network.html",
    title: Optional[str] = None,
    showfig: bool = True,
    notebook: bool = False,
    scaler: str = "minmax",           # edge-width scaling in D3Blocks
    dark_mode: bool = False,
) -> str:
    """
    Build an interactive n-gram co-occurrence network with D3Blocks (d3graph).

    Parameters
    ----------
    text : str
        Cleaned text.
    n : int
        n-gram size (2=bigram, 3=trigram).
    top_k_edges : int
        Keep the top-N edges by frequency (after min_count filter).
    min_count : int
        Minimum frequency to retain an edge.
    stopwords : Iterable[str] | None
        Words to exclude before forming n-grams (applied to all tokens).
    token_pattern : str
        Regex for tokenisation.
    lowercase : bool
        If True, lowercase tokens prior to processing.
    filepath : str
        Output HTML file path for the interactive chart.
    title : str | None
        Chart title (shown in HTML).
    showfig : bool
        Open the visual in the browser.
    notebook : bool
        Show inline in notebooks.
    scaler : str
        Edge scaling: 'zscore', 'minmax', or None.
    dark_mode : bool
        Toggle dark mode background in D3Blocks.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    assert n >= 2, "Use n >= 2 for n-gram networks."
    # 1) Tokenise
    toks = re.findall(token_pattern, text)
    if lowercase:
        toks = [t.lower() for t in toks]
    sw = set(w.lower() for w in (stopwords or []))
    toks = [t for t in toks if t and t not in sw]

    # 2) Build adjacent n-grams and turn into pairwise edges (source->target)
    #    For bigrams (n=2): edges are (w_i, w_{i+1})
    #    For trigrams (n=3): edges are ((w_i w_{i+1}) -> w_{i+2})
    edges: List[Tuple[str, str]] = []
    if n == 2:
        for i in range(len(toks) - 1):
            edges.append((toks[i], toks[i+1]))
    else:
        for i in range(len(toks) - (n - 1)):
            left = " ".join(toks[i:i+n-1])
            right = toks[i+n-1]
            edges.append((left, right))

    # 3) Count and prune
    counts = Counter(edges)
    # filter by min_count then take top_k
    items = [(src, dst, w) for (src, dst), w in counts.items() if w >= min_count]
    items.sort(key=lambda x: x[2], reverse=True)
    items = items[:top_k_edges]
    if not items:
        raise ValueError("No edges survived filtering. Lower 'min_count' or increase 'top_k_edges'.")

    # 4) Build DataFrame for D3Blocks (expects columns: source, target, weight)
    df_edges = pd.DataFrame(items, columns=["source", "target", "weight"])

    # 5) Create interactive network via D3Blocks (d3graph)
    d3 = D3Blocks()
    d3.d3graph(
        df_edges,
        scaler=scaler,
        dark_mode=dark_mode,
        title=title or (f"{'Bi' if n==2 else f'{n}-gram'} co-occurrence network"),
        filepath=filepath,
        showfig=showfig,
        notebook=notebook,
    )
    return filepath, df_edges