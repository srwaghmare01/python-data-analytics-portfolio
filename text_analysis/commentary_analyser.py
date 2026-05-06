"""
text_analysis/commentary_analyser.py

Analyses tab-separated football commentary files.
Supports word counting, minute-finding, and alphabetical order detection.

Usage:
    python commentary_analyser.py
"""

import re
import unicodedata
import numpy as np


def load_commentary(fpath: str):
    """Load a tab-separated commentary file into minute and comment arrays."""
    data = np.loadtxt(fpath, dtype=str, delimiter="\t", skiprows=1)
    minutes = data[:, 0].astype(float)
    comments = data[:, 1]
    return minutes, comments


def txtanalyser(fname: str, t: str, f, sel: str):
    """
    Search a commentary file for a specific term.

    Parameters
    ----------
    fname : str
        Path to commentary file.
    t : str
        Exact word to search for.
    f : callable
        NumPy aggregation function e.g. np.mean, np.sum.
    sel : str
        'count' returns total occurrences.
        'find' returns aggregated match minutes.

    Returns
    -------
    float or np.nan
    """
    minutes, comments = load_commentary(fname)

    if sel == "count":
        counts = np.array([c.split().count(t) for c in comments])
        return float(np.sum(counts))

    elif sel == "find":
        matched = [minutes[i] for i, c in enumerate(comments) if t in c.split()]
        if not matched:
            return np.nan
        return float(f(matched))

    else:
        raise ValueError(f"Invalid sel='{sel}'. Use 'count' or 'find'.")


def normalize(text: str) -> str:
    """
    Prepare text for alphabetical comparison.
    Removes punctuation, converts accented characters to ASCII,
    and handles Ñ/ñ explicitly.
    """
    text = re.sub(r"[^\w\s]", "", text)
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = text.replace("Ñ", "N").replace("ñ", "n")
    return text


def words_in_order(sentence: str, check_ties: bool = True) -> bool:
    """Return True if words in a sentence are in alphabetical order."""
    words = sentence.lower().split()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if not w1 or not w2:
            continue
        if check_ties:
            if w1 > w2:
                return False
        else:
            if w1[0] > w2[0]:
                return False
    return True


def find_alphabetical_order(fpath: str, check_ties: bool = True) -> list:
    """
    Return all commentary lines whose words are in alphabetical order.

    Parameters
    ----------
    fpath : str
        Path to commentary file.
    check_ties : bool
        True = full letter-by-letter comparison.
        False = first letter only.

    Returns
    -------
    list of str
    """
    results = []
    with open(fpath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            commentary = parts[1]
            if words_in_order(normalize(commentary), check_ties):
                results.append(commentary.strip())
    return results


if __name__ == "__main__":
    import numpy as np
    # Example usage — replace with your commentary file path
    print(txtanalyser("commentary.txt", "GOAL", np.mean, "find"))
    print(find_alphabetical_order("commentary.txt", check_ties=True)[0])
