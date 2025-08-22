
# sentiment_toolkit.py — reusable helpers to run and compare multiple sentiment analysers
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import importlib, re

def safe_import(module_name: str, pip_hint: Optional[str] = None):
    try:
        return importlib.import_module(module_name)
    except Exception:
        print(f"Optional dependency '{module_name}' is not available." + (f" Install with: {pip_hint}" if pip_hint else ""))
        return None

ORDER_5 = ["very negative","negative","neutral","positive","very positive"]
MAP_5 = {"very negative":-2,"negative":-1,"neutral":0,"positive":1,"very positive":2}

def map_vader_compound(c: float) -> str:
    if c <= -0.6: return "very negative"
    if c < -0.05: return "negative"
    if c <= 0.05: return "neutral"
    if c < 0.6: return "positive"
    return "very positive"

def map_polarity(p: float) -> str:
    if p <= -0.6: return "very negative"
    if p < -0.05: return "negative"
    if p <= 0.05: return "neutral"
    if p < 0.6: return "positive"
    return "very positive"

def label_to_triple(raw_label: str) -> str:
    lbl = (raw_label or "").strip().upper()
    mapping = {"LABEL_0":"NEGATIVE","LABEL_1":"NEUTRAL","LABEL_2":"POSITIVE",
               "NEG":"NEGATIVE","NEU":"NEUTRAL","POS":"POSITIVE",
               "NEGATIVE":"NEGATIVE","NEUTRAL":"NEUTRAL","POSITIVE":"POSITIVE"}
    return mapping.get(lbl, lbl)

def triple_to_five(triple_label: str, score: float, strong: float = 0.85) -> str:
    t = (triple_label or "").upper()
    if t == "NEGATIVE": return "Very Negative" if score >= strong else "Negative"
    if t == "POSITIVE": return "Very Positive" if score >= strong else "Positive"
    return "Neutral"

# Corrected stars_to_five function that returns lowercase (matching harmonise_5 expectations)
def stars_to_five_corrected(stars: int) -> str:
    """Convert star rating (1-5) to five-point sentiment scale (lowercase)"""
    star_map = {
        1: "very negative",
        2: "negative", 
        3: "neutral",
        4: "positive",
        5: "very positive"
    }
    return star_map.get(int(stars), "neutral")

@dataclass
class SentimentAnalyser:
    def analyse_vader(self, texts: Iterable[str]):
        nltk = safe_import("nltk","pip install nltk")
        if nltk is None: return None
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            try: nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError: nltk.download('vader_lexicon')
            sia = SentimentIntensityAnalyzer()
            rows = []
            for t in texts:
                s = sia.polarity_scores(t)
                rows.append({"analyser":"NLTK VADER","text":t,"compound":s["compound"],
                             "neg":s["neg"],"neu":s["neu"],"pos":s["pos"],
                             "interpretation":map_vader_compound(s["compound"])})
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception as e:
            print("VADER failed:", e); return None

    def analyse_textblob(self, texts: Iterable[str]):
        tb = safe_import("textblob","pip install textblob")
        if tb is None: return None
        try:
            from textblob import TextBlob
            rows = []
            for t in texts:
                b = TextBlob(t)
                pol, sub = float(b.sentiment.polarity), float(b.sentiment.subjectivity)
                rows.append({"analyser":"TextBlob","text":t,"polarity":pol,"subjectivity":sub,
                             "interpretation":map_polarity(pol)})
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception as e:
            print("TextBlob failed:", e); return None

    def analyse_spacy_textblob(self, texts: Iterable[str]):
        """Simplified spaCy + spacytextblob analyzer following the working pattern"""
        spacy_mod = safe_import("spacy", "pip install spacy")
        if spacy_mod is None:
            return None
            
        try:
            import spacy
            
            # Load model
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                print("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                return None
            
            # Try to add spacytextblob (following your working approach)
            stbl = safe_import("spacytextblob", "pip install spacytextblob")
            if stbl is None:
                print("spacytextblob not available")
                return None
                
            try:
                # Direct approach like your working code
                if "spacytextblob" not in nlp.pipe_names:
                    nlp.add_pipe("spacytextblob")
                    
                # Process texts
                docs = list(nlp.pipe(list(texts), batch_size=32))
                rows = []
                
                for text, doc in zip(texts, docs):
                    pol = float(doc._.blob.polarity)
                    sub = float(doc._.blob.subjectivity)
                    rows.append({
                        "analyser": "spaCy+spacytextblob",
                        "text": text,
                        "polarity": pol,
                        "subjectivity": sub,
                        "interpretation": map_polarity(pol)
                    })
                
                import pandas as pd
                return pd.DataFrame(rows)
                
            except Exception as e:
                print(f"spacytextblob pipeline error: {e}")
                return None
                
        except Exception as e:
            print("spaCy+spacytextblob failed:", e)
            return None

    def analyse_flair(self, texts: Iterable[str], neutral_band: float = 0.55):
        flair_mod = safe_import("flair","pip install flair")
        if flair_mod is None: return None
        try:
            from flair.models import TextClassifier
            from flair.data import Sentence
            clf = TextClassifier.load("en-sentiment")
            sents = [Sentence(t) for t in texts]; clf.predict(sents, mini_batch_size=32)
            rows = []
            for t, s in zip(texts, sents):
                lab = s.labels[0] if s.labels else None
                raw = lab.value if lab else ""; score = float(lab.score) if lab else 0.0
                if score < neutral_band: interp = "Neutral"
                elif "POS" in raw.upper(): interp = "Positive"
                elif "NEG" in raw.upper(): interp = "Negative"
                else: interp = "Neutral"
                rows.append({"analyser":"Flair (en-sentiment)","text":t,"raw_label":raw,"score":score,
                             "interpretation":interp})
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception as e:
            print("Flair failed:", e); return None

    def analyse_hf_transformer(self, texts: Iterable[str], model_name: str):
        tr = safe_import("transformers","pip install transformers torch --upgrade")
        if tr is None: return None
        try:
            from transformers import pipeline
            clf = pipeline(task="sentiment-analysis", model=model_name)
            preds = clf(list(texts))
            rows = []
            for t, p in zip(texts, preds):
                raw = str(p.get("label","")); score = float(p.get("score",0.0))
                # Handle star ratings (fixed logic)
                if "star" in raw.lower():
                    # Extract number of stars using correct regex
                    import re
                    m = re.search(r"(\d+)", raw)  # Single backslash, not double
                    if m:
                        stars = int(m.group(1))
                        # Use corrected stars_to_five function that returns lowercase
                        interp = stars_to_five_corrected(stars)
                    else:
                        # Fallback if no number found
                        interp = "neutral"
                else:
                    # Handle standard labels (LABEL_0, POSITIVE, etc.)
                    triple = label_to_triple(raw); interp = triple_to_five(triple, score)
                rows.append({"analyser":f"HF:{model_name}","text":t,"raw_label":raw,"score":score,
                             "interpretation":interp})
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"Transformers model '{model_name}' failed:", e); return None

def consolidate(frames):
    import pandas as pd
    dfs = [f for f in frames if f is not None]
    if not dfs: return pd.DataFrame(columns=["analyser","text","interpretation"])
    return pd.concat(dfs, ignore_index=True, sort=False)

def harmonise_5(df):
    import pandas as pd
    d = df.copy()
    def _row(r):
        interp = str(r.get("interpretation","")).strip().lower()
        if interp in MAP_5: return interp
        raw = str(r.get("raw_label","") or r.get("Raw Label","")).lower()
        if raw:
            sc = float(r.get("score", 0.0)) if r.get("score") is not None else 0.0
            from_label = triple_to_five(label_to_triple(raw), sc).lower()
            return from_label
        if r.get("compound") is not None: return map_vader_compound(float(r["compound"]))
        if r.get("polarity") is not None: return map_polarity(float(r["polarity"]))
        return "neutral"
    d["interpretation_5"] = d.apply(_row, axis=1)
    d["sentiment_numeric"] = d["interpretation_5"].map(MAP_5).astype(int)
    return d

class plots:
    @staticmethod
    def _shorten_analyser_name(name, max_len=15):
        """Shorten analyser names for better display"""
        name = str(name)
        if len(name) <= max_len:
            return name
        
        # Common abbreviations
        replacements = {
            'spacytextblob': 'spaCy+TB',
            'spacytextblob (fallback)': 'spaCy+TB*',
            'TextBlob': 'TB',
            'NLTK VADER': 'VADER',
            'Flair (en-sentiment)': 'Flair',
            'cardiffnlp/twitter-roberta-base-sentiment-latest': 'RoBERTa-Twitter',
            'nlptown/bert-base-multilingual-uncased-sentiment': 'BERT-Multi',
            'distilbert-base-uncased-finetuned-sst-2-english': 'DistilBERT-SST2',
        }
        
        for full, short in replacements.items():
            if full in name:
                return short
        
        # Generic shortening for HF models
        if name.startswith('HF:'):
            model = name[3:]
            if '/' in model:
                parts = model.split('/')
                if len(parts[-1]) > max_len:
                    return f"HF:{parts[-1][:max_len-3]}..."
                return f"HF:{parts[-1]}"
            elif len(model) > max_len-3:
                return f"HF:{model[:max_len-6]}..."
        
        # Fallback: truncate with ellipsis
        return f"{name[:max_len-3]}..." if len(name) > max_len else name

    @staticmethod
    def plot_distribution(df):
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Create data
        counts = (df.groupby(["analyser","interpretation_5"], as_index=False)
                    .size().pivot(index="analyser", columns="interpretation_5", values="size")
                    .reindex(columns=ORDER_5).fillna(0))
        
        # Shorten analyser names
        counts.index = [plots._shorten_analyser_name(name) for name in counts.index]
        
        # Create plot with improved styling
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Color palette for sentiment
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # red, orange, green, blue, purple
        
        counts.plot(kind="bar", stacked=True, ax=ax, color=colors, 
                   alpha=0.8, width=0.7, edgecolor='white', linewidth=0.5)
        
        # Styling
        ax.set_xlabel("Analyser", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=11, fontweight='bold')
        ax.set_title("Distribution of Sentiment Interpretations by Analyser", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Improve tick labels
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Legend styling
        legend = ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), 
                          loc='upper left', fontsize=9, title_fontsize=10)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap(df, text_col="text", max_text_len=50, max_analyser_len=15):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        def shorten_text(s, n=max_text_len):
            s = str(s).strip().replace("\n", " ")
            return (s[:n-1] + "…") if len(s) > n else s
        
        # Prepare data
        pivot = (df.assign(text_short=lambda d: d[text_col].map(lambda x: shorten_text(x, max_text_len)))
                   .pivot_table(index="text_short", columns="analyser",
                                values="sentiment_numeric", aggfunc="mean"))
        
        # Shorten analyser names
        pivot.columns = [plots._shorten_analyser_name(name, max_analyser_len) for name in pivot.columns]
        
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(min(14, len(pivot.columns) * 1.2), 
                                       max(6, len(pivot.index) * 0.35)))
        
        # Create heatmap with better colormap
        im = ax.imshow(pivot.values, aspect="auto", cmap='RdYlBu_r', 
                      vmin=-2, vmax=2, interpolation='nearest')
        
        # Set ticks and labels
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
        
        # Title and labels
        ax.set_title("Sentiment Heatmap: Text × Analyser", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Analyser", fontsize=11, fontweight='bold')
        ax.set_ylabel("Text Samples", fontsize=11, fontweight='bold')
        
        # Add colorbar with custom labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Sentiment Score', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        cbar.set_ticks([-2, -1, 0, 1, 2])
        cbar.set_ticklabels(['Very Neg', 'Negative', 'Neutral', 'Positive', 'Very Pos'])
        
        # Add text annotations for values
        if len(pivot.index) <= 20 and len(pivot.columns) <= 8:  # Only for smaller heatmaps
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    if not np.isnan(pivot.iloc[i, j]):
                        text = ax.text(j, i, f'{pivot.iloc[i, j]:.1f}',
                                     ha="center", va="center", fontsize=7,
                                     color="white" if abs(pivot.iloc[i, j]) > 1 else "black")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_agreement(df, max_analyser_len=15):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Prepare data with better error handling
        try:
            wide = (df.assign(text_id=lambda d: d["text"].factorize()[0])
                      .pivot_table(index="text_id", columns="analyser",
                                   values="interpretation_5", aggfunc="first"))
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            return
        
        analysers = list(wide.columns)
        if len(analysers) < 2:
            print("Need at least 2 analysers for agreement matrix")
            return
            
        # Initialize agreement matrix with explicit dtype
        agree = np.full((len(analysers), len(analysers)), np.nan, dtype=float)
        
        for i, a in enumerate(analysers):
            for j, b in enumerate(analysers):
                try:
                    if i == j:
                        agree[i, j] = 100.0  # Perfect self-agreement
                    else:
                        # Get valid comparisons
                        valid = wide[[a, b]].dropna()
                        if valid.empty:
                            agree[i, j] = np.nan
                        else:
                            # Ensure we're comparing strings/categorical data properly
                            col_a = valid[a].astype(str)
                            col_b = valid[b].astype(str)
                            pct = (col_a == col_b).mean() * 100.0
                            agree[i, j] = pct
                except Exception as e:
                    print(f"Error comparing {a} vs {b}: {e}")
                    agree[i, j] = np.nan
        
        # Shorten analyser names
        short_names = [plots._shorten_analyser_name(name, max_analyser_len) for name in analysers]
        agree_df = pd.DataFrame(agree, index=short_names, columns=short_names)
        
        # Handle NaN values for visualization
        agree_clean = np.nan_to_num(agree_df.values, nan=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(agree_clean, aspect="auto", cmap='viridis', 
                      vmin=0, vmax=100, interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        
        # Title and labels
        ax.set_title("Pairwise Agreement Matrix\n(% Exact Match on Sentiment Classification)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Analyser", fontsize=11, fontweight='bold')
        ax.set_ylabel("Analyser", fontsize=11, fontweight='bold')
        
        # Add percentage annotations
        for i in range(len(short_names)):
            for j in range(len(short_names)):
                val = agree_df.iloc[i, j]
                if not pd.isna(val):
                    color = "white" if val < 50 else "black"
                    text = ax.text(j, i, f'{val:.0f}%',
                                 ha="center", va="center", fontsize=8,
                                 color=color, fontweight='bold')
                else:
                    # Show "N/A" for missing comparisons
                    ax.text(j, i, 'N/A', ha="center", va="center", 
                           fontsize=8, color="gray", style='italic')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Agreement (%)', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_average(df, max_analyser_len=15):
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Prepare data
        stats = df.groupby("analyser")["sentiment_numeric"].agg(["mean", "std"]).sort_values("mean")
        
        # Shorten analyser names
        stats.index = [plots._shorten_analyser_name(name, max_analyser_len) for name in stats.index]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Color gradient based on sentiment
        colors = ['#d62728' if x < -0.5 else '#ff7f0e' if x < 0 else 
                 '#2ca02c' if x > 0.5 else '#1f77b4' for x in stats["mean"]]
        
        # Create bar plot
        bars = ax.bar(range(len(stats)), stats["mean"], 
                     yerr=stats["std"], capsize=4, alpha=0.8,
                     color=colors, edgecolor='black', linewidth=0.5)
        
        # Styling
        ax.set_xlabel("Analyser", fontsize=11, fontweight='bold')
        ax.set_ylabel("Mean Sentiment Score", fontsize=11, fontweight='bold')
        ax.set_title("Average Sentiment by Analyser\n(with Standard Deviation)", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats.index, rotation=45, ha='right', fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats["mean"], stats["std"])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.05,
                   f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Grid and styling
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=-2.2, top=2.2)
        
        # Add sentiment scale reference
        ax.text(0.02, 0.98, 'Scale: -2=Very Negative, 0=Neutral, +2=Very Positive', 
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.show()