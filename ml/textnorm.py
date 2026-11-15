"""Shared text normaliser and analyzers. web/coursemate_core.js ports this file line for line.

Rules (keep identical in JS):
- normalise: Unicode NFKC, then lowercase.
- tokens: ASCII letter/digit runs, or single Han characters (so "proposal幾時due" -> proposal, 幾, 時, due).
  Single-character tokens are kept ("quiz 2", "part a", "get 0").
- word1: unigrams. word12: unigrams + adjacent bigrams joined by one space.
- char35: sklearn-style char_wb 3-5-grams over the tokens joined by spaces, counted in code points.
"""
import re
import unicodedata

# Han ranges: CJK Ext A, CJK Unified, CJK Compatibility
HAN = "".join(chr(a) + "-" + chr(b) for a, b in [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)])
TOKEN_RE = re.compile("[a-z0-9]+|[" + HAN + "]")


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def tokens(s: str) -> list:
    return TOKEN_RE.findall(normalize(s))


def word1(s: str) -> list:
    return tokens(s)


def word12(s: str) -> list:
    t = tokens(s)
    return t + [t[i] + " " + t[i + 1] for i in range(len(t) - 1)]


def char_wb(s: str, lo: int = 3, hi: int = 5) -> list:
    out = []
    for w in tokens(s):
        w = " " + w + " "
        L = len(w)
        for n in range(lo, hi + 1):
            off = 0
            out.append(w[off:off + n])
            while off + n < L:
                off += 1
                out.append(w[off:off + n])
            if off == 0:  # word shorter than n: counted once
                break
    return out


def char35(s: str) -> list:
    return char_wb(s, 3, 5)


ANALYZERS = {"word1": word1, "word12": word12, "char35": char35}
