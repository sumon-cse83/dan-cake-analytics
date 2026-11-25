"""Convert the LaTeX report (report/main.tex) into a Microsoft Word document (report/main.docx).

How it works
------------
1. The LaTeX source is read exactly as Tectonic compiles it: chapters, generated tables and generated numbers.
   Cross-reference numbers and citation labels come from main.aux / main.bbl of the last LaTeX run, so the Word
   file always shows the same numbers as the PDF.
2. Display equations and TikZ diagrams are compiled once with LaTeX (preview package) and inserted as
   300-dpi pictures. Inline maths, tables, lists, listings, figures and captions become native Word content.
3. The result is written as Word-flavoured HTML; tools/word_finish.ps1 then lets Microsoft Word convert it to
   .docx and applies the page set-up, styles, table of contents, lists of figures/tables and page numbers.

Usage (from the project folder; Microsoft Word must be installed):
    cd report; ..\\.tex\\Library\\bin\\tectonic.exe -X compile -k main.tex; cd ..
    python tools\\latex_to_word.py
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report'
BUILD = REPORT / '_word_build'
TECTONIC = ROOT / '.tex' / 'Library' / 'bin' / 'tectonic.exe'
OUT_DOCX = REPORT / 'main.docx'

TEXTWIDTH_PX = 566          # 15 cm text width at 96 px per inch
TEXTHEIGHT_PX = 933         # 24.7 cm text height
SNIPPET_DPI = 300

warnings: set[str] = set()


# ============================================================================ helpers for braces and arguments
def match_brace(s: str, i: int, open_='{', close='}') -> int:
    """s[i] is an opening brace; return the index of the matching closing brace."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == '\\':
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j
        j += 1
    raise ValueError(f'unbalanced {open_} at {i}: {s[i:i + 60]!r}')


def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in ' \t\n':
        i += 1
    return i


def read_arg(s: str, i: int) -> tuple[str, int]:
    """Read one mandatory argument starting at i (after optional spaces)."""
    i = skip_ws(s, i)
    if i >= len(s):
        return '', i
    if s[i] == '{':
        j = match_brace(s, i)
        return s[i + 1:j], j + 1
    if s[i] == '\\':
        m = re.match(r'\\([a-zA-Z]+|.)', s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def read_opt(s: str, i: int) -> tuple[str | None, int]:
    j = skip_ws(s, i)
    if j < len(s) and s[j] == '[':
        depth, k = 0, j
        while k < len(s):
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
            elif s[k] == ']' and depth == 0:
                return s[j + 1:k], k + 1
            k += 1
    return None, i


def find_env_end(s: str, name: str, start: int) -> tuple[int, int]:
    """start is just after \\begin{name}; return (content_end, after_end)."""
    pat = re.compile(r'\\(begin|end)\{' + re.escape(name) + r'\}')
    depth = 1
    for m in pat.finditer(s, start):
        depth += 1 if m.group(1) == 'begin' else -1
        if depth == 0:
            return m.start(), m.end()
    raise ValueError(f'no \\end{{{name}}}')


def split_top(s: str, sep: str) -> list[str]:
    """Split on sep at brace depth 0 (sep is '&' or '\\\\')."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if sep == '\\\\' and s.startswith('\\\\', i) and depth == 0:
            parts.append(''.join(cur))
            cur = []
            i += 2
            opt, i2 = read_opt(s, i)   # \\[2pt]
            if opt is not None and re.fullmatch(r'[\d.]+\w*', opt.strip()):
                i = i2
            continue
        if c == '\\':
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        if c == sep and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append(''.join(cur))
    return parts


def strip_comments(s: str) -> str:
    out = []
    for line in s.split('\n'):
        m = re.search(r'(?<!\\)%', line)
        if m:
            line = line[:m.start()]
            out.append(line)
            out.append('\x01')      # a comment also removes the line end
        else:
            out.append(line + '\n')
    return ''.join(out).replace('\n\x01', '').replace('\x01', '')


# ============================================================================ accents and escaping
ACCENTS = {"'": '\u0301', '`': '\u0300', '^': '\u0302', '"': '\u0308', '~': '\u0303', '=': '\u0304',
           '.': '\u0307', 'c': '\u0327', 'v': '\u030C', 'u': '\u0306', 'H': '\u030B'}


def esc(t: str) -> str:
    return html.escape(t, quote=False)


# ============================================================================ the converter
class Converter:
    def __init__(self):
        self.macros: dict[str, str] = {}
        self.labels: dict[str, tuple[str, str]] = {}     # key -> (number, type)
        self.cites: dict[str, tuple[str, str]] = {}      # key -> (short author, year)
        self.snippets: list[tuple[str, str]] = []        # (kind, latex)
        self.listings: list[tuple[str | None, str]] = []
        self.chap = 0
        self.chap_label = '0'          # '1', '2', ... for chapters, 'A', 'B', ... after \appendix
        self.appendix = False
        self.counters = {'section': 0, 'subsection': 0, 'subsubsection': 0}
        self.floatn = {'Figure': 0, 'Table': 0, 'Listing': 0}
        self.float_kind: str | None = None
        self.last_caption_no: str | None = None
        self.snip_img: dict[int, tuple[Path, int, int]] = {}
        self.img_scale = 1.0

    # ------------------------------------------------------------------ inputs
    def load_aux(self):
        aux = (REPORT / 'main.aux').read_text(encoding='utf8')
        for m in re.finditer(r'\\newlabel\{([^}]+)@cref\}\{\{\[(\w+)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}', aux):
            self.labels[m.group(1)] = (m.group(3), m.group(2))
        for m in re.finditer(r'\\bibcite\{([^}]+)\}\{\{\d+\}\{(.*?)\}\{\{(.*?)\}\}', aux):
            year = re.sub(r'\{\\natexlab\{(\w)\}\}', r'\1', m.group(2))
            self.cites[m.group(1)] = (m.group(3), year)

    def load_macros(self, text: str):
        for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{', text):
            j = match_brace(text, m.end() - 1)
            self.macros[m.group(1)] = text[m.end():j]

    def read_tex(self, rel: str) -> str:
        p = REPORT / rel
        if not p.suffix:
            p = p.with_suffix('.tex')
        return self.expand_inputs(p.read_text(encoding='utf8'))

    def expand_inputs(self, s: str) -> str:
        s = self.protect_verbatim(s)
        s = strip_comments(s)
        return re.sub(r'\\input\{([^}]+)\}', lambda m: self.read_tex(m.group(1)), s)

    def protect_verbatim(self, s: str) -> str:
        def lst(m):
            self.listings.append((m.group(1), m.group(2).strip('\n')))
            return f'\x00LST{len(self.listings) - 1}\x00'
        s = re.sub(r'\\begin\{lstlisting\}(?:\[([^\]]*)\])?\n(.*?)\\end\{lstlisting\}', lst, s, flags=re.S)

        def lstin(m):
            code = (REPORT / m.group(2)).read_text(encoding='utf8')
            first = re.search(r'firstline=(\d+)', m.group(1))
            if first:
                code = '\n'.join(code.split('\n')[int(first.group(1)) - 1:])
            self.listings.append((m.group(1), code.strip('\n')))
            return f'\x00LST{len(self.listings) - 1}\x00'
        s = re.sub(r'\\lstinputlisting\[(.*?)\]\{([^}]+)\}', lstin, s, flags=re.S)
        return s

    def protect_snippets(self, s: str) -> str:
        def add(kind, body):
            self.snippets.append((kind, body))
            return f'\x00SNIP{len(self.snippets) - 1}\x00'
        s = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', lambda m: add('tikz', m.group(0)), s,
                   flags=re.S)
        s = re.sub(r'\\\[(.*?)\\\]', lambda m: add('math', m.group(1)), s, flags=re.S)
        s = re.sub(r'\\begin\{align\*\}(.*?)\\end\{align\*\}',
                   lambda m: add('math', r'\begin{aligned}' + m.group(1) + r'\end{aligned}'), s, flags=re.S)
        return s

    def expand_macros(self, s: str) -> str:
        names = sorted(self.macros, key=len, reverse=True)
        pat = re.compile(r'\\(' + '|'.join(map(re.escape, names)) + r')(?![a-zA-Z])[ \t]*(\n(?![ \t]*\n))?[ \t]*')
        nospace = set(",.'/?;:!~-)]}")

        def rep(m):
            val = self.macros[m.group(1)]
            rest = s[m.end():m.end() + 2]
            if val.endswith('\\xspace'):
                val = val[:-len('\\xspace')]
                if rest and rest[0] not in nospace and not rest.startswith('\\%') and not rest.startswith('{}'):
                    val += ' '
            return val
        for _ in range(3):              # macros can contain macros
            s = pat.sub(rep, s)
        return s

    # ------------------------------------------------------------------ references
    def ref_names(self, keys: list[str], cap=True) -> str:
        names = {'chapter': 'Chapter', 'section': 'Section', 'subsection': 'Section', 'subsubsection': 'Section',
                 'paragraph': 'Section', 'figure': 'Figure', 'table': 'Table', 'lstlisting': 'Listing',
                 'equation': 'Equation', 'appendix': 'Appendix'}
        groups: list[tuple[str, list[str]]] = []
        for k in keys:
            k = k.strip()
            if k not in self.labels:
                warnings.add(f'undefined reference {k}')
                num, typ = '??', 'section'
            else:
                num, typ = self.labels[k]
            name = names.get(typ, typ.capitalize())
            if groups and groups[-1][0] == name:
                groups[-1][1].append(num)
            else:
                groups.append((name, [num]))
        parts = []
        for name, nums in groups:
            if len(nums) == 1:
                parts.append(f'{name}\u00a0{nums[0]}')
            else:
                plural = name + 's'
                parts.append(f'{plural}\u00a0' + ', '.join(nums[:-1]) + ' and ' + nums[-1])
        return ' and '.join(parts)

    def cite(self, kind: str, keys: str, pre: str | None, post: str | None) -> str:
        items = []
        for k in [k.strip() for k in keys.split(',')]:
            if k not in self.cites:
                warnings.add(f'undefined citation {k}')
                items.append((k, '?'))
            else:
                a, y = self.cites[k]
                items.append((self.inline(a), y))
        if kind == 'citet':
            txt = ', '.join(f'{a} ({y})' for a, y in items)
            if post:
                txt = txt[:-1] + f', {self.inline(post)})'
            return txt
        txt = '; '.join(f'{a}, {y}' for a, y in items)
        if post:
            txt += ', ' + self.inline(post)
        if pre:
            txt = self.inline(pre) + ' ' + txt
        return f'({txt})'

    # ------------------------------------------------------------------ images
    def img_tag(self, path: Path, width_px: float, height_limit: float | None = None) -> str:
        with Image.open(path) as im:
            w, h = im.size
        width_px = min(width_px, TEXTWIDTH_PX)
        height = width_px * h / w
        if height_limit and height > height_limit:
            width_px *= height_limit / height
            height = height_limit
        return (f'<img width={round(width_px)} height={round(height)} '
                f'src="{path.resolve().as_uri()}" alt="{esc(path.stem)}">')

    def include_graphics(self, opts: str | None, name: str, scale=1.0) -> str:
        p = None
        for base in (REPORT / 'figures', REPORT):
            for ext in ('', '.png', '.jpg', '.jpeg'):
                q = base / (name + ext)
                if q.is_file():
                    p = q
                    break
            if p:
                break
        if p is None:
            warnings.add(f'missing image {name}')
            return f'[missing image {esc(name)}]'
        frac, hlim = 1.0, None
        if opts:
            m = re.search(r'width=([\d.]*)\\(?:linewidth|textwidth)', opts)
            if m:
                frac = float(m.group(1) or 1)
            m = re.search(r'height=([\d.]*)\\textheight', opts)
            if m:
                hlim = float(m.group(1) or 1) * TEXTHEIGHT_PX
        return self.img_tag(p, frac * TEXTWIDTH_PX * scale, hlim)

    # ------------------------------------------------------------------ inline text
    def inline(self, s: str) -> str:
        out: list[str] = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == '\\':
                i = self.command(s, i, out)
                if i < 0:           # a font switch consumed the rest of the group
                    break
                continue
            if c == '$':
                j = s.index('$', i + 1)
                while s[j - 1] == '\\':
                    j = s.index('$', j + 1)
                out.append(self.math(s[i + 1:j]))
                i = j + 1
                continue
            if c == '{':
                j = match_brace(s, i)
                out.append(self.inline(s[i + 1:j]))
                i = j + 1
                continue
            if c == '}':
                i += 1
                continue
            if c == '\x00':
                j = s.index('\x00', i + 1)
                out.append(self.placeholder(s[i + 1:j], inline=True))
                i = j + 1
                continue
            if c == '~':
                out.append('\u00a0')
            elif s.startswith('---', i):
                out.append('\u2014')
                i += 2
            elif s.startswith('--', i):
                out.append('\u2013')
                i += 1
            elif s.startswith('``', i):
                out.append('\u201c')
                i += 1
            elif s.startswith("''", i):
                out.append('\u201d')
                i += 1
            elif c == '`':
                out.append('\u2018')
            elif c == "'":
                out.append('\u2019')
            elif c in ' \t\n':
                if out and not out[-1].endswith((' ', '\u00a0', '<br>')):
                    out.append(' ')
            else:
                out.append(esc(c))
            i += 1
        return ''.join(out)

    def command(self, s: str, i: int, out: list[str]) -> int:
        """Handle the control sequence at s[i]; return the next position (or -1 when the rest was consumed)."""
        nxt = s[i + 1] if i + 1 < len(s) else ''
        if nxt == '(':                       # \( inline maths \)
            j = s.index('\\)', i + 2)
            out.append(self.math(s[i + 2:j]))
            return j + 2
        if nxt in 'cvuH' and s[i + 2:i + 3] == '{':     # \c{c} and similar accents
            arg, j = read_arg(s, i + 2)
            out.append(unicodedata.normalize('NFC', arg + ACCENTS[nxt]))
            return j
        if not nxt.isalpha():
            i += 2
            if nxt == '\\':
                _, i = read_opt(s, i)
                out.append('<br>')
            elif nxt in '%&_#${}':
                out.append(esc(nxt))
            elif nxt in ',;:':
                out.append('\u2009')
            elif nxt in ' \n':
                out.append(' ')
            elif nxt in ACCENTS:
                arg, i = read_arg(s, i)
                if arg.startswith('\\'):         # \'\i : spaces after the control word are ignored
                    while i < len(s) and s[i] == ' ':
                        i += 1
                base = {'\\i': 'i', '\\j': 'j'}.get(arg, arg)
                out.append(unicodedata.normalize('NFC', base + ACCENTS[nxt]))
            # \- \/ \@ and similar produce nothing
            return i
        m = re.match(r'[a-zA-Z]+\*?', s[i + 1:])
        name = m.group(0)
        i += 1 + len(name)
        star = name.endswith('*')
        name = name.rstrip('*')

        simple = {'LaTeX': 'LaTeX', 'TeX': 'TeX', 'dots': '\u2026', 'ldots': '\u2026', 'textendash': '\u2013',
                  'textemdash': '\u2014', 'i': '\u0131', 'ss': '\u00df', 'quad': '\u2003', 'qquad': '\u2003\u2003',
                  'newline': '<br>', 'linebreak': '<br>', 'textbackslash': '\\', 'textasciitilde': '~',
                  'copyright': '\u00a9', 'pounds': '\u00a3', 'S': '\u00a7', 'P': '\u00b6', 'times': '\u00d7',
                  'newblock': ' '}
        ignore0 = {'centering', 'raggedright', 'raggedleft', 'noindent', 'indent', 'par', 'smallskip', 'medskip',
                   'bigskip', 'hfill', 'vfill', 'allowbreak', 'nobreak', 'relax', 'xspace', 'arraybackslash',
                   'protect', 'normalsize', 'small', 'footnotesize', 'scriptsize', 'tiny', 'large', 'Large',
                   'LARGE', 'huge', 'Huge', 'clearpage', 'newpage', 'cleardoublepage', 'item', 'toprule', 'midrule',
                   'bottomrule', 'hline', 'endgroup', 'begingroup', 'sloppy', 'strut', 'unskip', 'ignorespaces',
                   'onehalfspacing', 'singlespacing', 'selectfont', 'displaystyle', 'textstyle'}
        switches = {'em': ('<i>', '</i>'), 'itshape': ('<i>', '</i>'), 'bfseries': ('<b>', '</b>'),
                    'ttfamily': ('<span class=CodeInline>', '</span>'), 'sffamily': ('', ''),
                    'rmfamily': ('', ''), 'upshape': ('', ''), 'normalfont': ('', ''), 'scshape': ('', '')}
        wrap1 = {'textbf': ('<b>', '</b>'), 'emph': ('<i>', '</i>'), 'textit': ('<i>', '</i>'),
                 'textsl': ('<i>', '</i>'), 'underline': ('<u>', '</u>'), 'textsuperscript': ('<sup>', '</sup>'),
                 'textsubscript': ('<sub>', '</sub>'), 'texttt': ('<span class=CodeInline>', '</span>'),
                 'textsc': ('', ''), 'textrm': ('', ''), 'textnormal': ('', ''), 'textup': ('', ''),
                 'mbox': ('', ''), 'hbox': ('', ''), 'text': ('', ''), 'textsf': ('', ''), 'natexlab': ('', '')}

        def gobble(j):          # LaTeX ignores spaces (and one line end) after a control word
            k = j
            while k < len(s) and s[k] in ' \t':
                k += 1
            if k < len(s) and s[k] == '\n':
                k += 1
                while k < len(s) and s[k] in ' \t':
                    k += 1
            return k

        if name in simple:
            out.append(simple[name])
            return gobble(i) if name not in ('newblock',) else i
        if name in ignore0:
            return gobble(i)
        if name == 'appendix':           # chapters that follow are Appendix A, B, ...
            self.appendix, self.chap = True, 0
            return gobble(i)
        if name == 'penalty':
            return re.match(r'-?\d*', s[i:]).end() + i
        if name in switches:
            a, b = switches[name]
            rest = s[skip_ws(s, i):]
            out.append(a + self.inline(rest) + b)
            return -1
        if name in wrap1:
            arg, i = read_arg(s, i)
            a, b = wrap1[name]
            out.append(a + self.inline(arg) + b)
            return i
        if name == 'doi':
            arg, i = read_arg(s, i)
            out.append(f'doi: <a href="https://doi.org/{esc(arg)}">{esc(arg)}</a>')
            return i
        if name in ('path', 'url'):
            j = skip_ws(s, i)
            if s[j] == '{':
                arg, i = read_arg(s, i)
            else:                   # \path|...|
                k = s.index(s[j], j + 1)
                arg, i = s[j + 1:k], k + 1
            arg = arg.replace('\\_', '_').replace('\\%', '%').replace('\\#', '#')
            if name == 'url':
                out.append(f'<a href="{esc(arg)}">{esc(arg)}</a>')
            else:
                out.append(f'<span class=CodeInline>{esc(arg)}</span>')
            return i
        if name == 'href':
            url, i = read_arg(s, i)
            txt, i = read_arg(s, i)
            out.append(f'<a href="{esc(url)}">{self.inline(txt)}</a>')
            return i
        if name in ('citet', 'citep', 'cite', 'citealt', 'citealp'):
            o1, i = read_opt(s, i)
            o2, i = read_opt(s, i)
            keys, i = read_arg(s, i)
            pre, post = (o1, o2) if o2 is not None else (None, o1)
            kind = 'citet' if name in ('citet', 'cite') else 'citep'
            if name in ('citealt', 'citealp'):
                out.append(self.cite('citep', keys, pre, post)[1:-1])
            else:
                out.append(self.cite(kind, keys, pre, post))
            return i
        if name in ('citeauthor', 'citeyear'):
            keys, i = read_arg(s, i)
            a, y = self.cites.get(keys.strip(), ('?', '?'))
            out.append(self.inline(a) if name == 'citeauthor' else y)
            return i
        if name in ('cref', 'Cref', 'autoref'):
            keys, i = read_arg(s, i)
            out.append(self.ref_names(keys.split(',')))
            return i
        if name in ('ref', 'eqref'):
            key, i = read_arg(s, i)
            num = self.labels.get(key.strip(), ('??', ''))[0]
            out.append(f'({num})' if name == 'eqref' else num)
            return i
        if name in ('label', 'pageref', 'hspace', 'vspace', 'color', 'pagestyle', 'thispagestyle',
                    'addcontentsline', 'markboth', 'setlength', 'phantomsection', 'setstretch'):
            n_args = {'addcontentsline': 3, 'markboth': 2, 'setlength': 2}.get(name, 1)
            for _ in range(n_args):
                _, i = read_arg(s, i)
            return gobble(i)
        if name == 'rule':
            _, i = read_opt(s, i)
            w, i = read_arg(s, i)
            _, i = read_arg(s, i)
            cm = float(re.match(r'[\d.]+', w).group(0)) if 'cm' in w else 4
            out.append('_' * int(cm * 4))
            return i
        if name == 'parbox':
            _, i = read_opt(s, i)
            _, i = read_arg(s, i)
            body, i = read_arg(s, i)
            out.append(self.inline(body))
            return i
        if name in ('texorpdfstring', 'textcolor'):
            a, i = read_arg(s, i)
            b, i = read_arg(s, i)
            out.append(self.inline(a if name == 'texorpdfstring' else b))
            return i
        if name == 'includegraphics':
            opts, i = read_opt(s, i)
            f, i = read_arg(s, i)
            out.append(self.include_graphics(opts, f, getattr(self, 'img_scale', 1.0)))
            return i
        if name == 'footnote':
            arg, i = read_arg(s, i)
            out.append(f' ({self.inline(arg)})')
            return i
        if name == 'caption':        # only reaches here outside floats
            _, i = read_opt(s, i)
            arg, i = read_arg(s, i)
            out.append(self.inline(arg))
            return i
        warnings.add(f'unknown command \\{name}')
        return i

    # ------------------------------------------------------------------ inline maths
    GREEK = {n: chr(c) for n, c in [('alpha', 0x3B1), ('beta', 0x3B2), ('gamma', 0x3B3), ('delta', 0x3B4),
             ('epsilon', 0x3F5), ('varepsilon', 0x3B5), ('zeta', 0x3B6), ('eta', 0x3B7), ('theta', 0x3B8),
             ('kappa', 0x3BA), ('lambda', 0x3BB), ('mu', 0x3BC), ('nu', 0x3BD), ('xi', 0x3BE), ('pi', 0x3C0),
             ('rho', 0x3C1), ('sigma', 0x3C3), ('tau', 0x3C4), ('phi', 0x3D5), ('varphi', 0x3C6), ('chi', 0x3C7),
             ('psi', 0x3C8), ('omega', 0x3C9), ('Gamma', 0x393), ('Delta', 0x394), ('Theta', 0x398),
             ('Lambda', 0x39B), ('Sigma', 0x3A3), ('Phi', 0x3A6), ('Omega', 0x3A9), ('Pi', 0x3A0)]}
    MSYM = {'times': '\u00d7', 'cdot': '\u00b7', 'rightarrow': '\u2192', 'to': '\u2192', 'leftarrow': '\u2190',
            'leftrightarrow': '\u2194', 'Rightarrow': '\u21d2', 'geq': '\u2265', 'ge': '\u2265', 'leq': '\u2264',
            'le': '\u2264', 'neq': '\u2260', 'approx': '\u2248', 'in': '\u2208', 'notin': '\u2209', 'mid': '|',
            'dots': '\u2026', 'ldots': '\u2026', 'cdots': '\u22ef', 'sum': '\u2211', 'prod': '\u220f',
            'infty': '\u221e', 'pm': '\u00b1', 'top': '\u22a4', 'lVert': '\u2016', 'rVert': '\u2016',
            'ell': '\u2113', 'sim': '\u223c', 'propto': '\u221d', 'partial': '\u2202', 'nabla': '\u2207',
            'quad': '\u2003', 'qquad': '\u2003\u2003', '%': '%', 'lbrace': '{', 'rbrace': '}'}
    MFUNC = {'log', 'ln', 'exp', 'sin', 'cos', 'tan', 'max', 'min', 'arg', 'det', 'lim', 'sup', 'inf'}
    REL = set('=<>') | {'\u2265', '\u2264', '\u2260', '\u2248', '\u2208', '\u2209', '\u2192', '\u2194', '\u221d'}

    def math(self, s: str) -> str:
        out: list[str] = []
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c == '\\':
                nxt = s[i + 1] if i + 1 < n else ''
                if not nxt.isalpha():
                    i += 2
                    out.append({'{': '{', '}': '}', ',': '\u2009', ';': ' ', ':': ' ', '!': '', ' ': ' ',
                                '%': '%', '|': '\u2016', '\\': ' '}.get(nxt, nxt))
                    continue
                name = re.match(r'[a-zA-Z]+', s[i + 1:]).group(0)
                i += 1 + len(name)
                if name in self.GREEK:
                    out.append(f'<i>{self.GREEK[name]}</i>' if name[0].islower() else self.GREEK[name])
                elif name in self.MSYM:
                    sym = self.MSYM[name]
                    out.append(f' {sym} ' if sym in self.REL or name == 'times' else sym)
                elif name in self.MFUNC:
                    out.append(name)
                elif name in ('text', 'mathrm', 'textrm', 'operatorname', 'mbox'):
                    arg, i = read_arg(s, i)
                    out.append(self.inline(arg))
                elif name in ('mathbf', 'boldsymbol', 'bm'):
                    arg, i = read_arg(s, i)
                    out.append(f'<b>{self.math(arg)}</b>')
                elif name in ('mathit', 'mathcal'):
                    arg, i = read_arg(s, i)
                    out.append(self.math(arg))
                elif name in ('hat', 'widehat', 'bar', 'overline', 'tilde', 'widetilde', 'dot'):
                    arg, i = read_arg(s, i)
                    mark = {'hat': '\u0302', 'widehat': '\u0302', 'bar': '\u0304', 'overline': '\u0304',
                            'tilde': '\u0303', 'widetilde': '\u0303', 'dot': '\u0307'}[name]
                    inner = self.math(arg)
                    out.append(re.sub(r'([^<>\s])((?:</[a-z]+>)*)$', r'\1' + mark + r'\2', inner, count=1))
                elif name in ('frac', 'tfrac', 'dfrac'):
                    a, i = read_arg(s, i)
                    b, i = read_arg(s, i)
                    a, b = self.math(a), self.math(b)
                    if re.search(r'[ +\-\u2212]', re.sub('<[^>]+>', '', a).strip()):
                        a = f'({a})'
                    if re.search(r'[ +\-\u2212/]', re.sub('<[^>]+>', '', b).strip()):
                        b = f'({b})'
                    out.append(f'{a}/{b}')
                elif name == 'sqrt':
                    arg, i = read_arg(s, i)
                    out.append(f'\u221a({self.math(arg)})')
                elif name in ('left', 'right', 'bigl', 'bigr', 'Bigl', 'Bigr', 'big', 'Big', 'displaystyle',
                              'textstyle', 'limits', 'nolimits'):
                    pass
                else:
                    m = self.macros.get(name)
                    if m is not None:
                        out.append(esc(m.replace('\\xspace', '')))
                    else:
                        warnings.add(f'unknown maths command \\{name}')
                continue
            if c in '^_':
                arg, i = read_arg(s, i + 1)
                tag = 'sup' if c == '^' else 'sub'
                out.append(f'<{tag}>{self.math(arg)}</{tag}>')
                continue
            if c == '{':
                j = match_brace(s, i)
                out.append(self.math(s[i + 1:j]))
                i = j + 1
                continue
            if c.isalpha():
                j = i
                while j < n and s[j].isalpha():
                    j += 1
                out.append(f'<i>{s[i:j]}</i>')
                i = j
                continue
            if c in ' \t\n~':
                i += 1
                continue
            if c == '-':
                out.append('\u2212')
            elif c in '=<>':
                out.append(f' {esc(c)} ')
            elif c == '+':
                out.append(' + ')
            elif c == "'":
                out.append('\u2032')
            else:
                out.append(esc(c))
            i += 1
        res = ''.join(out)
        res = re.sub(r'  +', ' ', res).strip()
        res = re.sub(r'^\+ ', '+', res)
        return res.replace('<i></i>', '')

    # ------------------------------------------------------------------ placeholders
    def placeholder(self, tag: str, inline=False) -> str:
        if tag.startswith('SNIP'):
            p, w, h = self.snip_img[int(tag[4:])]
            if w > TEXTWIDTH_PX:
                h, w = h * TEXTWIDTH_PX / w, TEXTWIDTH_PX
            return f'<img width={round(w)} height={round(h)} src="{p.resolve().as_uri()}" alt="diagram">'
        if tag.startswith('LST'):
            opts, code = self.listings[int(tag[3:])]
            parts = []
            cap = re.search(r'caption=\{(.*?)\}', opts or '')
            if cap:
                self.floatn['Listing'] += 1
                no = f'{self.chap_label}.{self.floatn["Listing"]}'
                lab = re.search(r'label=\{(.*?)\}', opts or '')
                self.check_label(lab.group(1) if lab else None, no)
                parts.append(f'<p class=ListingCaption><b>Listing {no}:</b> {self.inline(cap.group(1))}</p>')
            lines = ''.join(f'<p class=CodeBlock>{esc(line) or "&nbsp;"}</p>'
                            for line in code.replace('\t', '    ').split('\n'))
            parts.append(lines.replace('  ', '&nbsp; '))
            return ''.join(parts)
        raise ValueError(tag)

    def check_label(self, key: str | None, mine: str):
        if key and key in self.labels and self.labels[key][0] != mine:
            warnings.add(f'numbering differs for {key}: LaTeX {self.labels[key][0]}, Word {mine}')

    # ------------------------------------------------------------------ blocks
    BLOCK_RE = re.compile(
        r'\\begin\{(figure|table|itemize|enumerate|description|tabular|tabularx|longtable|adjustbox|center|'
        r'minipage|quote|quotation|titlepage)\*?\}'
        r'|\\(chapter|section|subsection|subsubsection|paragraph)(\*?)(?![a-zA-Z])'
        r'|\n[ \t]*\n|\\par(?![a-zA-Z])|\x00(SNIP|LST)(\d+)\x00|\\caption(?![a-zA-Z])|\\dashpair(?![a-zA-Z])'
        r'|\\bibliography\{')

    def blocks(self, s: str, ctx: str = 'body') -> list[str]:
        out: list[str] = []
        para: list[str] = []
        prefix = ['']

        def flush():
            text = ''.join(para).strip()
            para.clear()
            if not text and not prefix[0]:
                return
            body = self.inline(text).strip()
            if not body and not prefix[0]:
                return
            if re.fullmatch(r'(<br>|\s|&nbsp;)*', body) and not prefix[0]:
                return
            cls = {'figure': 'FigureImage', 'table': 'TableNote', 'center': 'Centered'}.get(ctx, 'MsoNormal')
            if prefix[0] and ctx == 'body':
                cls = 'MsoNormal'
            out.append(f'<p class={cls}>{prefix[0]}{body}</p>')
            prefix[0] = ''

        i = 0
        while True:
            m = self.BLOCK_RE.search(s, i)
            if not m:
                para.append(s[i:])
                break
            para.append(s[i:m.start()])
            tok = m.group(0)
            i = m.end()
            if m.group(1):                                    # environments
                env = m.group(1)
                star = s[m.end() - 2] == '*'
                name = env + ('*' if star else '')
                inner_start = i
                if env in ('figure', 'table'):
                    _, inner_start = read_opt(s, inner_start)
                cend, after = find_env_end(s, name, i)
                content = s[inner_start:cend]
                i = after
                if env == 'titlepage':
                    continue
                flush()
                out.extend(self.environment(env, content, ctx))
            elif m.group(2):                                  # headings
                flush()
                _, i = read_opt(s, i)
                title, i = read_arg(s, i)
                lab = re.match(r'\s*\\label\{([^}]+)\}', s[i:])
                key = None
                if lab:
                    key = lab.group(1)
                    i += lab.end()
                out.append(self.heading(m.group(2), bool(m.group(3)), title, key))
                if m.group(2) == 'paragraph':
                    prefix[0] = out.pop()
            elif tok.strip() == '' or tok.startswith('\\par'):
                flush()
            elif m.group(4):                                  # placeholders
                flush()
                kind, idx = m.group(4), int(m.group(5))
                if kind == 'LST':
                    out.append(self.placeholder(f'LST{idx}'))
                else:
                    cls = 'Equation' if self.snippets[idx][0] == 'math' else 'FigureImage'
                    out.append(f'<p class={cls}>{self.placeholder(f"SNIP{idx}")}</p>')
            elif tok.startswith('\\caption'):
                flush()
                _, i = read_opt(s, i)
                cap, i = read_arg(s, i)
                lab = re.match(r'\s*\\label\{([^}]+)\}', s[i:])
                key = None
                if lab:
                    key = lab.group(1)
                    i += lab.end()
                out.append(self.caption(cap, key))
            elif tok.startswith('\\dashpair'):
                flush()
                args = []
                for _ in range(6):
                    a, i = read_arg(s, i)
                    args.append(a)
                out.append(self.dashpair(args))
            elif tok.startswith('\\bibliography'):
                flush()
                _, i = read_arg(s, i - 1)
                out.extend(self.bibliography())
        flush()
        return out

    def heading(self, level: str, star: bool, title: str, key: str | None) -> str:
        t = self.inline(title)
        if level == 'paragraph':
            return f'<b>{t}</b> '
        if star:
            tag = 'h1' if level == 'chapter' else 'h2'
            return f'<{tag}>{t}</{tag}>'
        if level == 'chapter':
            self.chap += 1
            self.counters = dict.fromkeys(self.counters, 0)
            self.floatn = dict.fromkeys(self.floatn, 0)
            no = chr(64 + self.chap) if self.appendix else str(self.chap)
            self.chap_label = no
            self.check_label(key, no)
            return f'<h1>{"Appendix" if self.appendix else "Chapter"} {no}: {t}</h1>'
        if level == 'section':
            self.counters.update(section=self.counters['section'] + 1, subsection=0, subsubsection=0)
            no = f'{self.chap_label}.{self.counters["section"]}'
            tag = 'h2'
        elif level == 'subsection':
            self.counters.update(subsection=self.counters['subsection'] + 1, subsubsection=0)
            no = f'{self.chap_label}.{self.counters["section"]}.{self.counters["subsection"]}'
            tag = 'h3'
        else:
            self.counters['subsubsection'] += 1
            no = f'{self.chap_label}.{self.counters["section"]}.{self.counters["subsection"]}.' \
                 f'{self.counters["subsubsection"]}'
            tag = 'h4'
        self.check_label(key, no)
        return f'<{tag}>{no}\u00a0\u00a0{t}</{tag}>'

    def caption(self, text: str, key: str | None) -> str:
        kind = self.float_kind or 'Figure'
        self.floatn[kind] += 1
        no = f'{self.chap_label}.{self.floatn[kind]}'
        self.check_label(key, no)
        return f'<p class={kind}Caption><b>{kind} {no}:</b> {self.inline(text)}</p>'

    def environment(self, env: str, content: str, ctx: str) -> list[str]:
        if env in ('figure', 'table'):
            saved = self.float_kind
            self.float_kind = 'Figure' if env == 'figure' else 'Table'
            # several pictures side by side: shrink a little so that they stay on one line in Word
            self.img_scale = 0.96 if content.count('\\includegraphics') > 1 else 1.0
            res = self.blocks(content, env)
            self.float_kind = saved
            self.img_scale = 1.0
            return res
        if env in ('adjustbox', 'minipage'):
            _, j = read_opt(content, 0)
            _, j = read_arg(content, j)
            return self.blocks(content[j:], ctx)
        if env in ('center', 'quote', 'quotation'):
            return self.blocks(content, 'center' if env == 'center' else ctx)
        if env in ('itemize', 'enumerate', 'description'):
            return [self.listing(env, content)]
        if env in ('tabular', 'tabularx', 'longtable'):
            return self.table(env, content)
        raise ValueError(env)

    def listing(self, env: str, content: str) -> str:
        opt, j = read_opt(content, 0)
        label = None
        if opt:
            m = re.search(r'label=([^,\]]*)', opt)
            if m:
                label = m.group(1).strip()
        content = content[j:]
        # split into items at nesting depth 0
        items, depth, last = [], 0, None
        for m in re.finditer(r'\\(begin|end)\{(itemize|enumerate|description)\}|\\item(?![a-zA-Z])', content):
            if m.group(1) == 'begin':
                depth += 1
            elif m.group(1) == 'end':
                depth -= 1
            elif depth == 0:
                if last is not None:
                    items.append(content[last:m.start()])
                last = m.end()
        if last is not None:
            items.append(content[last:])
        html_items = []
        for k, it in enumerate(items, 1):
            lab, j = read_opt(it, 0)
            body = self.blocks(it[j:], 'body')
            parts = []
            for b in body:
                mm = re.fullmatch(r'<p class=MsoNormal>(.*)</p>', b, flags=re.S)
                parts.append(mm.group(1) if mm else b)
            inner = '<br>'.join(p for p in parts if not p.startswith('<ul') and not p.startswith('<ol')
                                and not p.startswith('<p'))
            nested = ''.join(p for p in parts if p.startswith(('<ul', '<ol', '<p')))
            if label and env == 'enumerate':
                tag = label.replace('\\arabic*', str(k)).replace('\\alph*', chr(96 + k)).replace(
                    '\\roman*', ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'][k - 1])
                tag = self.inline(tag)
                html_items.append(f'<p class=LabelItem>{tag}<span style="mso-tab-count:1">\u00a0 </span>'
                                  f'{inner}</p>{nested}')
            elif lab is not None:
                html_items.append(f'<li><b>{self.inline(lab)}</b> {inner}{nested}</li>')
            else:
                html_items.append(f'<li>{inner}{nested}</li>')
        if label and env == 'enumerate':
            return ''.join(html_items)
        tag = 'ol' if env == 'enumerate' else 'ul'
        return f'<{tag}>' + ''.join(html_items) + f'</{tag}>'

    # ------------------------------------------------------------------ tables
    @staticmethod
    def parse_colspec(spec: str) -> list[dict]:
        cols, i, pre = [], 0, ''
        while i < len(spec):
            c = spec[i]
            if c in ' |':
                i += 1
            elif c in '@!<':
                _, i = read_arg(spec, i + 1)
            elif c == '>':
                pre, i = read_arg(spec, i + 1)
            elif c == '*':
                k, i = read_arg(spec, i + 1)
                sub, i = read_arg(spec, i)
                cols += Converter.parse_colspec(sub * int(k))
            elif c in 'lcr':
                cols.append({'align': {'l': 'left', 'c': 'center', 'r': 'right'}[c], 'width': None, 'pre': pre})
                pre = ''
                i += 1
            elif c in 'pmb':
                w, i = read_arg(spec, i + 1)
                cols.append({'align': 'left', 'width': w, 'pre': pre})
                pre = ''
            elif c == 'X':
                cols.append({'align': 'left', 'width': 'X', 'pre': pre})
                pre = ''
                i += 1
            else:
                i += 1
        for col in cols:
            if '\\centering' in col['pre']:
                col['align'] = 'center'
            elif '\\raggedleft' in col['pre']:
                col['align'] = 'right'
        return cols

    def table(self, env: str, content: str) -> list[str]:
        out = []
        j = 0
        if env == 'tabularx':
            _, j = read_arg(content, j)
        _, j = read_opt(content, j)
        spec, j = read_arg(content, j)
        body = content[j:]
        cols = self.parse_colspec(spec)
        if env == 'longtable':
            cap = re.match(r'\s*\\caption\{', body)
            if cap:
                k = match_brace(body, cap.end() - 1)
                text = body[cap.end():k]
                rest = body[k + 1:]
                lab = re.match(r'\s*\\label\{([^}]+)\}', rest)
                key = lab.group(1) if lab else None
                rest = rest[lab.end():] if lab else rest
                rest = re.sub(r'^\s*\\\\', '', rest)
                saved = self.float_kind
                self.float_kind = 'Table'
                out.append(self.caption(text, key))
                self.float_kind = saved
                body = rest
            first, _, rest = body.partition('\\endfirsthead')
            if '\\endhead' in rest:
                rest = rest.split('\\endhead', 1)[1]
            if '\\endfoot' in rest:
                foot, rest = rest.split('\\endfoot', 1)
            body = first + rest + '\\bottomrule'
        rows = split_top(body, '\\\\')
        rule_re = re.compile(r'\s*\\(toprule|midrule|bottomrule|hline|cmidrule|addlinespace|cline)(\([^)]*\))?'
                             r'(\[[^\]]*\])?(\{[^}]*\})?')
        parsed = []          # (cells, rule_above)
        pending = None
        header_rows = 0
        seen_mid = False
        for r in rows:
            while True:
                m = rule_re.match(r)
                if not m:
                    break
                kind = m.group(1)
                if kind in ('toprule', 'bottomrule'):
                    pending = 'heavy'
                elif kind in ('midrule', 'hline'):
                    pending = 'light'
                    if not seen_mid:
                        header_rows = len(parsed)
                        seen_mid = True
                elif kind == 'cmidrule' and pending is None:
                    pending = 'thin'
                r = r[m.end():]
            if r.strip() == '':
                continue
            parsed.append((split_top(r, '&'), pending))
            pending = None
        final_rule = pending
        full = any(c['width'] for c in cols)
        style = 'border-collapse:collapse;' + ('width:100%;' if full else '')
        cls = 'DataTable'
        rows_html = []
        widths = {'heavy': '1.5pt', 'light': '0.75pt', 'thin': '0.5pt'}
        for ri, (cells, rule) in enumerate(parsed):
            tds, ci = [], 0
            last_row = ri == len(parsed) - 1
            for cell in cells:
                span, align, txt = 1, cols[ci]['align'] if ci < len(cols) else 'left', cell.strip()
                if txt.startswith('\\multicolumn'):
                    nspan, k = read_arg(txt, len('\\multicolumn'))
                    cspec, k = read_arg(txt, k)
                    txt, _ = read_arg(txt, k)
                    span = int(nspan)
                    sub = self.parse_colspec(cspec)
                    align = sub[0]['align'] if sub else align
                mr = re.match(r'\\multirow\{[^}]*\}\{[^}]*\}\{', txt)
                if mr:
                    k = match_brace(txt, mr.end() - 1)
                    txt = txt[mr.end():k]
                pre = cols[ci]['pre'] if ci < len(cols) else ''
                inner = self.inline(txt)
                if '\\ttfamily' in pre:
                    inner = f'<span class=CodeInline>{inner}</span>'
                css = ['padding:0.5pt 4pt', 'vertical-align:top']
                if rule:
                    css.append(f'border-top:solid windowtext {widths[rule]}')
                if ri == header_rows - 1 and header_rows:
                    pass
                if last_row and final_rule:
                    css.append(f'border-bottom:solid windowtext {widths[final_rule]}')
                w = cols[ci]['width'] if ci < len(cols) else None
                if w and w != 'X' and span == 1:
                    m = re.match(r'([\d.]+)cm', w)
                    if m:
                        css.append(f'width:{float(m.group(1)) * 37.8:.0f}px')
                colspan = f' colspan={span}' if span > 1 else ''
                tds.append(f'<td{colspan} style="{";".join(css)}"><p class=TableText style="text-align:{align}">'
                           f'{inner or "&nbsp;"}</p></td>')
                ci += span
            rows_html.append('<tr>' + ''.join(tds) + '</tr>')
        out.append(f'<table class={cls} data-header="{header_rows}" style="{style}">' + ''.join(rows_html)
                   + '</table>')
        return out

    def dashpair(self, a: list[str]) -> str:
        saved = self.float_kind
        self.float_kind = 'Figure'
        cells = []
        for img, cap, key in (a[0:3], a[3:6]):
            pic = self.include_graphics(r'width=\linewidth,height=.4\textheight', 'screens/' + img.strip(), 0.485)
            cells.append(f'<td style="width:50%;vertical-align:bottom;padding:0 4pt">'
                         f'<p class=FigureImage>{pic}</p>{self.caption(cap, key.strip())}</td>')
        self.float_kind = saved
        return '<table class=LayoutTable style="width:100%;border:none">' + '<tr>' + ''.join(cells) + '</tr></table>'

    # ------------------------------------------------------------------ references
    def bibliography(self) -> list[str]:
        bbl = strip_comments((REPORT / 'main.bbl').read_text(encoding='utf8'))
        out = ['<h1>References</h1>']
        for m in re.finditer(r'\\bibitem\[.*?\]\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', bbl,
                             flags=re.S):
            body = m.group(2).strip()
            out.append(f'<p class=Reference>{self.inline(body)}</p>')
        return out


# ============================================================================ LaTeX snippets -> pictures
def render_snippets(conv: Converter):
    BUILD.mkdir(exist_ok=True)
    main = (REPORT / 'main.tex').read_text(encoding='utf8')
    pre = main[main.index('\\usepackage[T1]'):main.index('\\begin{document}')]
    gen = (REPORT / 'generated').resolve().as_posix()
    pre = pre.replace('\\input{generated/', f'\\input{{{gen}/')
    pre = re.sub(r'\\graphicspath\{.*\}', '', pre)
    pre = pre.replace('\\usepackage{setspace}\\onehalfspacing', '\\usepackage{setspace}')
    pre = re.sub(r'\\AtBeginEnvironment.*', '', pre)
    body = []
    for kind, tex in conv.snippets:
        if kind == 'math':
            body.append('\\begin{preview}$\\displaystyle ' + tex.strip() + '$\\end{preview}\n\\newpage\n')
        else:
            body.append('\\begin{preview}' + tex + '\\end{preview}\n\\newpage\n')
    doc = ('\\documentclass[12pt]{report}\n' + pre + '\n\\usepackage[active,tightpage]{preview}\n'
           '\\setlength\\PreviewBorder{2pt}\n\\pagestyle{empty}\n\\begin{document}\n' + ''.join(body)
           + '\\end{document}\n')
    tex = BUILD / 'snippets.tex'
    tex.write_text(doc, encoding='utf8')
    res = subprocess.run([str(TECTONIC), '-X', 'compile', 'snippets.tex'], cwd=BUILD, capture_output=True,
                         text=True)
    if res.returncode:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise SystemExit('LaTeX snippets failed to compile')
    pdf = fitz.open(BUILD / 'snippets.pdf')
    if len(pdf) != len(conv.snippets):
        raise SystemExit(f'expected {len(conv.snippets)} snippet pages, got {len(pdf)}')
    for k, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=SNIPPET_DPI)
        p = BUILD / f'snippet_{k:02d}.png'
        pix.save(p)
        conv.snip_img[k] = (p, pix.width * 96 / SNIPPET_DPI, pix.height * 96 / SNIPPET_DPI)


# ============================================================================ document
CSS = """
@page { size: 595.3pt 841.9pt; margin: 70.9pt 70.9pt 70.9pt 99.2pt; }
p.MsoNormal { margin: 0 0 6pt 0; text-align: justify; }
p.FigureImage { text-align: center; margin: 6pt 0 2pt 0; }
p.Equation { text-align: center; margin: 4pt 0 6pt 0; }
p.FigureCaption { text-align: center; margin: 2pt 0 10pt 0; }
p.TableCaption { text-align: center; margin: 8pt 0 4pt 0; }
p.ListingCaption { text-align: center; margin: 8pt 0 4pt 0; }
p.TableNote { margin: 3pt 0 8pt 0; text-align: left; }
p.TableText { margin: 0; }
p.LabelItem { margin: 0 0 3pt 36pt; text-indent: -36pt; text-align: justify; }
p.CodeBlock { margin: 0; }
p.Reference { margin: 0 0 4pt 18pt; text-indent: -18pt; text-align: left; }
p.Centered { text-align: center; }
p.TitleLine { text-align: center; margin: 0; }
p.FrontHeading { text-align: center; }
p.Placeholder { }
span.CodeInline { font-family: Consolas; }
"""


def title_page(c: Converter) -> str:
    M = c.macros
    t = lambda k: c.inline(M[k])            # noqa: E731

    def line(text, size, bold=False, space=0, color=None):
        style = f'font-size:{size}pt;margin-top:{space}pt;' + (f'color:#{color};' if color else '')
        text = f'<b>{text}</b>' if bold else text
        return f'<p class=TitleLine style="{style}">{text}</p>'
    return ''.join([
        line(t('University'), 14), line(t('Department'), 12, space=4),
        line(t('ProjectTitle'), 20, True, 90, '1F4E79'),
        line(t('ProjectSubtitle'), 15, False, 14),
        line('Final Project Report', 14, True, 70, 'C55A11'),
        line('A project report submitted in partial fulfilment of the requirements for the degree of', 13, False, 16),
        line(t('Programme'), 15, True, 6),
        line(f'<b>Submitted by:</b> {t("StudentName")}', 13, False, 70),
        line(t('StudentID'), 13),
        line(f'<b>Supervisor:</b> {t("Supervisor")}', 13, False, 14),
        line(t('SubmissionDate'), 13, False, 90),
    ])


def main():
    if not (REPORT / 'main.aux').is_file() or not (REPORT / 'main.bbl').is_file():
        raise SystemExit('Run tectonic with -k first so that main.aux and main.bbl exist.')
    c = Converter()
    c.load_aux()
    main_tex = (REPORT / 'main.tex').read_text(encoding='utf8')
    for g in ('generated/numbers.tex', 'generated/algo_numbers.tex'):
        c.load_macros((REPORT / g).read_text(encoding='utf8'))
    c.load_macros(strip_comments(main_tex))
    c.macros.pop('dashpair', None)

    body = main_tex[main_tex.index('\\begin{document}') + 16:main_tex.index('\\end{document}')]
    body = c.expand_inputs(body)
    body = re.sub(r'\\newcommand\{\\dashpair\}\[6\]\{.*?\\end\{figure\}\}\n', '', body, flags=re.S)
    body = c.protect_snippets(body)
    body = c.expand_macros(body)
    # parts of main.tex that Word builds itself
    body = re.sub(r'\\(tableofcontents|listoffigures|listoftables)', lambda m: f'\n\n[[{m.group(1)}]]\n\n', body)
    body = re.sub(r'\\renewcommand\{\\bibsection\}.*\n|\\bibliographystyle\{[^}]*\}', '', body)
    body = re.sub(r'\\begingroup\\setstretch\{1\}.*?(?=\\bibliography\{)', '', body, flags=re.S)
    body = re.sub(r'\\pagenumbering\{\w+\}', '', body)
    render_snippets(c)
    parts = c.blocks(body)
    html_body = '\n'.join(parts)
    for key, title in (('tableofcontents', 'Table of Contents'), ('listoffigures', 'List of Figures'),
                       ('listoftables', 'List of Tables')):
        html_body = html_body.replace(f'<p class=MsoNormal>[[{key}]]</p>',
                                      f'<p class=FrontHeading>{title}</p><p class=Placeholder>[[{key}]]</p>')
    doc = ('<html><head><meta charset="utf-8"><title>' + esc(c.macros['ProjectTitle']) + '</title><style>' + CSS
           + '</style></head><body>' + title_page(c) + html_body + '</body></html>')
    out_html = BUILD / 'report.html'
    out_html.write_text(doc, encoding='utf8')
    print(f'HTML written: {out_html}  ({len(c.snippets)} rendered diagrams/equations, {len(c.listings)} listings)')
    for w in sorted(warnings):
        print('WARNING:', w)

    if '--html-only' in sys.argv:
        return
    ps = ROOT / 'tools' / 'word_finish.ps1'
    res = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps),
                          '-Html', str(out_html), '-Docx', str(OUT_DOCX)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode:
        print(res.stderr)
        raise SystemExit('Word conversion failed')
    if '--keep-build' not in sys.argv:
        pass
    shutil.copy2(OUT_DOCX, ROOT / 'docs' / 'report.docx')
    print('Copied to docs/report.docx')


if __name__ == '__main__':
    main()
