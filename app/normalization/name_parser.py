"""Split an as-recorded name cell into one fact per column.

A single source cell routinely carries several distinct facts at once::

    Willie Pashegezhik (Cloud Running in a Line)
    Campau (also spelt Compo)
    Fletcher (Souliere)
    Christine/Christina

Nia cannot match a person by name because those facts are fused. This module pulls them
apart into the schema below, leaving the source string untouched -- the original remains
the only thing displayed or cited.

Three relations are kept separate because they behave differently:

* ``extra``    - a DIFFERENT name for the same person (alias, maiden, married, adoptive).
  Matters for identity matching: "Souliere" must find "Alice Fletcher".
* ``spelling`` - the SAME name written differently. Matters for fuzzy matching:
  "Compo" must find "Wesley Campau".
* ``meaning``  - an English translation of an Indigenous name. Searchable as text, but
  never matched as a name.

Anything the rules cannot place confidently goes to ``alias``, which always feeds
matching. A misclassified variant still locates the person; a *dropped* one does not, so
the parser never guesses between extra and spelling when the evidence is weak.

No dataset vocabulary is encoded here. The cues are punctuation conventions and general
English/French given-name morphology, measured against the curated workbook rather than
assumed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- schema


@dataclass(frozen=True)
class NameParts:
    """One parsed name. Every tuple field is multi-valued and order-preserving."""

    first: str = ""
    middle: tuple[str, ...] = ()
    last: str = ""
    extra_first: tuple[str, ...] = ()
    extra_last: tuple[str, ...] = ()
    first_spelling: tuple[str, ...] = ()
    last_spelling: tuple[str, ...] = ()
    first_meaning: tuple[str, ...] = ()
    last_meaning: tuple[str, ...] = ()
    indigenous: str = ""
    indigenous_spelling: tuple[str, ...] = ()
    indigenous_meaning: tuple[str, ...] = ()
    alias: tuple[str, ...] = ()
    # Text found in the name cell that is not a name at all: a provenance note, a
    # community, a date, a status marker, a sentence about the family. Keeping it here
    # means nothing a person recorded is thrown away, while the name fields stay clean and
    # the search index never matches somebody by their community or by the word "removed".
    residue: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def searchable(self) -> tuple[str, ...]:
        """Every token a name query should be able to match, deduplicated."""
        parts: list[str] = [self.first, self.last, self.indigenous]
        for group in (
            self.middle,
            self.extra_first,
            self.extra_last,
            self.first_spelling,
            self.last_spelling,
            self.indigenous_spelling,
            self.alias,
        ):
            parts.extend(group)
        seen: dict[str, None] = {}
        for item in parts:
            token = item.strip()
            if token:
                seen.setdefault(token, None)
        return tuple(seen)


# ----------------------------------------------------------------------- vocabulary

# Generational and honorific tokens are not names and must not become a middle name.
SUFFIXES = frozenset({"sr", "sr.", "jr", "jr.", "ii", "iii", "iv", "snr", "jnr"})
TITLES = frozenset({"rev", "rev.", "mr", "mr.", "mrs", "mrs.", "miss", "fr", "fr.", "dr", "dr."})
# Professional designations recorded beside a name. Matched as a phrase, not a token.
_CREDENTIAL = re.compile(r"\bP\.?\s*Eng\.?\b", re.IGNORECASE)

# A general list of English/French given names, used only to tell an ordinary middle name
# ("Robert Nolan" -> middle) from an Indigenous name carried alongside the English one
# ("Willie Pashegezhik" -> a second identity). This is language morphology, not dataset
# vocabulary: no entry is specific to these records.
COMMON_GIVEN_NAMES = frozenset(
    """
    abel abigail ada adam adeline agnes alan albert alberta alex alexander alexandra alfred
    alice allan allen alma alvin amanda amelia andrew angela ann anna anne annie anthony
    archie arnold arthur audrey barbara basil beatrice benjamin bernard bernice bertha bessie
    betsey betsy betty beulah beverly bill billy blanche brenda brian bruce bryan calvin
    carl carol caroline catherine cecil cecilia charles charlie charlotte chris christina
    christine clara clarence clark claude clayton clifford clinton colin constance cora
    cynthia daniel danny darlene david dawn dean deborah delbert dennis diane don donald
    donna dora doreen doris dorothy douglas duncan earl edgar edith edmund edna edward
    edwin eileen elaine eleanor elgin elias elijah elizabeth ella ellen elmer elsie emily
    emma eric ernest ernie esther ethel etta eugene eva evelyn everett fanny fay flora
    florence floyd frances francis frank fred freda frederick garnet gary gene george gerald
    geraldine gerry gertrude gilbert gladys glen glenn gloria godfrey gordon grace greg
    gregory harold harriet harry harvey hazel helen henry herbert herman hilda howard
    hubert hugh ida ira irene iris irvin isaac isabel isabella isaiah israel jack jacob james
    jane janet janice jason jean jeanette jeffrey jennie jeremiah jerome jerry jesse jessie
    joan joanne joe john johnny jonathan joseph josephine joshua joyce juanita judith
    judy julia julie juliette june justin karen katherine kathleen keith kenneth kevin
    laura lawrence leo leon leonard leroy leslie lester lewis lila lillian lily linda
    lionel lloyd lois lorraine louis louisa louise lucy luke lydia lyle mabel madeline
    maggie marcel margaret maria marian marie marilyn marion marjorie mark marlene martha
    martin marvin mary mathew matilda matthew maud maurice may melvin mercy michael mildred
    millie milton minnie mitchell moses murray myrtle nancy naomi nathan nellie nelson
    nicholas nora norma norman olive oliver ophelia orville oscar owen pamela patricia
    patrick paul paula pauline pearl peggy percy peter philip phillip phoebe phyllis rachel
    ralph raymond rebecca regina reginald rhea richard rita robert roberta robin roderick
    roger roland ronald rose rosemary roy ruby russell ruth sam samuel sandra sarah shirley
    sidney silas simon solomon sophia stanley stella stephen steve stewart susan susanna
    susie sammy sylvia ted terrance theresa thomas timothy tina tom tommy tony vera vernon veronica
    victor victoria vincent viola violet virginia vivian wallace walter wanda warren wayne
    wesley wilbert wilfred wilfrid william willie wilma wilson winnie yvonne zachariah
    """.split()
)

# --------------------------------------------------------------------------- cues

_ALSO_SPELT = re.compile(r"\b(?:also\s+|sometimes\s+|occasionally\s+)?spel[lt]\w*\b", re.IGNORECASE)
# Prose a transcriber wrote around a name: "We called her Bessy", "sometimes spelt X".
# Whatever survives a strip must still be a name, never a fragment of the sentence, or it
# becomes a searchable token and the person turns up under words like "called".
_PROSE_LEAD = re.compile(
    r"^\s*(?:we\s+|they\s+|she\s+|he\s+|it\s+)?(?:was\s+|were\s+|is\s+|are\s+)?"
    r"(?:also\s+|sometimes\s+|occasionally\s+|often\s+)?"
    r"(?:call(?:ed|s)?|know(?:n|s)?|name[ds]?|refer(?:red)?(?:\s+to)?)\s+"
    r"(?:her|him|them|as|by)?\s*",
    re.IGNORECASE,
)
_PROSE_WORD = re.compile(
    r"\b(?:we|they|she|he|it|was|were|is|are|her|him|them|called|calls|known|knows|"
    r"sometimes|occasionally|often|also|the|and|but|because|who|which|that)\b",
    re.IGNORECASE,
)


def _strip_prose(text: str) -> str:
    """Reduce a fragment to the name inside it, or to nothing if it is all prose."""
    cleaned = _PROSE_LEAD.sub("", text or "").strip(" :,-?.")
    if not cleaned:
        return ""
    words = cleaned.split()
    # A leading run of ordinary sentence words is narration, not part of the name.
    while words and _PROSE_WORD.fullmatch(words[0]):
        words.pop(0)
    while words and _PROSE_WORD.fullmatch(words[-1]):
        words.pop()
    return " ".join(words).strip(" :,-?.")
_AKA = re.compile(r"\b(?:a\.?k\.?a\.?|also\s+known\s+as|nee|née)\b", re.IGNORECASE)
_OR = re.compile(r"^\s*or\s+", re.IGNORECASE)
_BRACKET = re.compile(r"\[([^\]]*)\]")
_PAREN = re.compile(r"\(([^)]*)\)")
_QUOTED = re.compile("[\"“]([^\"”]+)[\"”]")
_SPLITTERS = re.compile(r"\s*(?:/|&|\band\b|\bor\b)\s*", re.IGNORECASE)
_ENGLISH_WORD = re.compile(r"^[a-z][a-z'-]*$")

# Parenthetical notes describing the record's provenance or status rather than the person.
# Recorded as flags so the name itself stays clean.
_ANNOTATION = re.compile(
    r"\b(?:source|vital\s+stats?|nctr|cirnac|removed|census|archive|died)\b",
    re.IGNORECASE,
)
_CHILDREN_NOTE = re.compile(r"^children$", re.IGNORECASE)
_SPELLING_UNCERTAIN = re.compile(r"^sp\.?\s*\??$", re.IGNORECASE)
_INDIAN_NAME_LABEL = re.compile(r"^indian\s+name$", re.IGNORECASE)
_MEANING_LABEL = re.compile(
    r"^\s*(?:meaning|means|i\.?e\.?)\s*[:\-\s]*[\"“]?(.+?)[\"”]?\s*$",
    re.IGNORECASE,
)
_RELATIONSHIP = re.compile(
    r"\b(?:sister|brother|father|mother|son|daughter|wife|husband|widow|cousin|child of|sibling)\b",
    re.IGNORECASE,
)
_EDITORIAL = re.compile(
    r"\b(?:moved to|rsdit|see also|to be reviewed|placeholder)\b",
    re.IGNORECASE,
)
_OR_SPLIT = re.compile(r"\s+or\s+", re.IGNORECASE)

# Similarity above which two forms are the same name spelled differently. Taken from the
# curated corpus: spelling pairs average 0.86, alias pairs 0.31.
SPELLING_THRESHOLD = 0.78
SLASH_SPELLING_THRESHOLD = 0.60


# Transliterated Anishinaabemowin and Cree names carry digraphs that are vanishingly rare
# in English given names. Used only to tell a second Indigenous identity ("Willie
# Pashegezhik") from an ordinary middle name ("Robert Nolan"); it never affects matching,
# since both columns feed search.
_INDIGENOUS_SHAPE = re.compile(
    r"qua|ooq|aush|eesh|shk|kwa|ahw|ahb|ahs|oosh|wun|gezh|nooq|beek|dahs",
    re.IGNORECASE,
)
_INITIAL = re.compile(r"^[A-Za-z]\.?$")
# "B.G." and "W.E." are two initials written as one token.
_RUN_OF_INITIALS = re.compile(r"^(?:[A-Za-z]\.){2,}$")

# Conventional abbreviations of given names, expanded so the canonical form is the name
# itself. General English/French usage, not specific to these records.
ABBREVIATIONS = {
    "geo": "George",
    "wm": "William",
    "chas": "Charles",
    "jas": "James",
    "jno": "John",
    "thos": "Thomas",
    "robt": "Robert",
    "rich": "Richard",
    "edw": "Edward",
    "benj": "Benjamin",
    "saml": "Samuel",
    "jos": "Joseph",
    "margt": "Margaret",
    "eliz": "Elizabeth",
}


def _clean_token(token: str) -> str:
    """Trim punctuation a transcriber left on a token without altering the name."""
    return token.strip(" ,;:\"'").strip()


def _abbreviation_of(token: str) -> str:
    """The modern spelling of a marked abbreviation: ``Wm.`` -> ``William``, else ``""``.

    Only a token the source explicitly marks as abbreviated with a trailing period
    qualifies. A bare ``Chas`` is left alone: without the period it may simply be the
    recorded name.
    """
    cleaned = _clean_token(token)
    if not cleaned.endswith("."):
        return ""
    return ABBREVIATIONS.get(cleaned.rstrip(".").lower(), "")


def _is_initial(token: str) -> bool:
    """``R.`` -- one letter the transcriber marked as standing for a name."""
    cleaned = _clean_token(token)
    return len(cleaned) == 2 and cleaned[0].isalpha() and cleaned.endswith(".")


def _expand_token(token: str) -> str:
    """Normalise one name token, keeping the form the register actually uses.

    An initial and a conventional abbreviation are both kept as written -- ``R.``, ``Wm.``
    -- because that is what the page says and what a researcher reading it will search
    for. The expansion is not lost: ``_abbreviation_of`` returns it and the callers file it
    as a spelling, so ``William`` still finds ``Wm.`` and both are searchable.
    """
    cleaned = _clean_token(token)
    if not cleaned:
        return ""
    if _is_initial(cleaned) or _abbreviation_of(cleaned):
        return cleaned
    return cleaned.rstrip(".")


def _explode_tokens(tokens: list[str]) -> list[str]:
    """Split runs of joined initials so each letter is its own token."""
    out: list[str] = []
    for token in tokens:
        cleaned = _clean_token(token)
        if _RUN_OF_INITIALS.match(cleaned):
            # "B.G." is two initials, and each keeps the period that marks it as one --
            # splitting on "." would hand back a bare "B" and lose that.
            out.extend(f"{part}." for part in cleaned.split(".") if part)
        elif cleaned:
            out.append(cleaned)
    return out


def _bare(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def _is_middle_name(token: str) -> bool:
    """Whether a plain token beside the first name is a middle name rather than an alias.

    In the curated corpus ordinary middle names outnumber second identities 169 to 26, so
    the default is middle. An initial is always a middle. A token outside the general
    given-name vocabulary that carries Indigenous orthography is a separate identity.
    """
    cleaned = token.strip(".").lower()
    if not cleaned:
        return False
    if _INITIAL.match(token):
        return True
    if cleaned in COMMON_GIVEN_NAMES:
        return True
    return not (_INDIGENOUS_SHAPE.search(cleaned) or len(cleaned) >= 9)


def _ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, _bare(left), _bare(right)).ratio()


_RELATION_WORD = re.compile(r"^\s*(?:or|aka|a\.k\.a\.|nee|née|sp\.?\??)\s", re.IGNORECASE)


_QUOTE_CHARS = " ,;\"“”‘’«»ΓÇ£ΓÇ¥"


def _strip_meaning_label(text: str) -> str:
    """Pull the gloss out of 'meaning \"Stand in Middle\"' / 'meaning Speaks in Pairs'."""
    cleaned = text.strip()
    labeled = _MEANING_LABEL.match(cleaned)
    body = labeled.group(1) if labeled else cleaned
    return body.strip(_QUOTE_CHARS)


def looks_like_meaning(text: str) -> bool:
    """Whether a parenthetical is an English gloss rather than another name.

    Translations read as ordinary English ("Cloud Running in a Line", "Big Wind"); alternate
    names do not. Multi-word English is decisive. A single word is left alone: "Duck" is a
    gloss but "Souliere" is a surname, and nothing in the string separates them.
    """
    cleaned = _strip_meaning_label(text)
    if _MEANING_LABEL.match(text.strip()) and cleaned:
        return True
    words = cleaned.split()
    if len(words) < 2 or _RELATION_WORD.match(cleaned):
        return False
    if _EDITORIAL.search(cleaned) or _RELATIONSHIP.search(cleaned):
        return False
    lowered = [w.strip(".,'\"").lower() for w in words]
    if any(w in COMMON_GIVEN_NAMES for w in lowered):
        return False
    return all(_ENGLISH_WORD.match(w) for w in lowered)


_DIMINUTIVE_SUFFIX = re.compile(r"(?:ie|ey|y|o)$", re.IGNORECASE)


def is_alternate_name(base: str, variant: str) -> bool:
    """Whether two similar forms are different names rather than one name respelled.

    High string similarity normally means a respelling, but English diminutives break
    that: John/Johnny and Charles/Charlie look alike yet are two names for one person,
    which is an identity fact rather than an orthographic one. Two forms are alternates
    when both are established given names, or when one is the other's diminutive.
    """
    left, right = base.strip().lower(), variant.strip().lower()
    if not left or not right or left == right:
        return False
    short, long_ = sorted((left, right), key=len)
    # "John" -> "Johnny", "Jack" -> "Jackie": the longer form is the shorter one plus a
    # diminutive ending. Deliberately narrow -- a shared prefix alone is the signature of
    # a respelling ("Kezhikgobness" / "Kezhikgobnes"), which must stay a spelling.
    return (
        len(short) >= 3
        and long_.startswith(short)
        and bool(_DIMINUTIVE_SUFFIX.search(long_))
        and len(long_) - len(short) <= 3
    )


def classify_variant(base: str, variant: str, cue: str) -> str:
    """Return 'spelling', 'extra' or 'alias' for one variant of ``base``.

    Punctuation is the primary signal and string similarity breaks the ambiguous cues.
    When neither is decisive the variant becomes an alias rather than a guess.
    """
    if cue in {"spelt", "bracket", "uncertain"}:
        return "spelling"
    if cue in {"aka", "or", "quoted"}:
        return "extra"
    if is_alternate_name(base, variant):
        return "extra"
    similarity = _ratio(base, variant)
    if cue == "slash":
        if similarity >= SLASH_SPELLING_THRESHOLD:
            return "spelling"
        return "extra" if similarity < 0.4 else "alias"
    if cue == "paren":
        if similarity >= SPELLING_THRESHOLD:
            return "spelling"
        return "extra" if similarity < 0.55 else "alias"
    if cue == "plain":
        return "extra"
    return "alias"


# ------------------------------------------------------------------- decomposition


@dataclass
class Decomposed:
    """One cell pulled apart: a base form, its variants, glosses and quality flags."""

    base: str = ""
    variants: list[tuple[str, str]] = field(default_factory=list)
    meanings: list[str] = field(default_factory=list)
    # Provenance and status notes lifted out of the cell. Not names, but not noise either.
    residue: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def decompose(text: str) -> Decomposed:
    """Pull bracketed, parenthetical, quoted and slashed forms out of one cell."""
    out = Decomposed()
    working = (text or "").strip()
    if not working:
        return out

    # "Bott[omles]" is one name with an editorial insertion, not a name plus a variant:
    # the brackets mark letters the transcriber supplied. The canonical form is the text
    # with the brackets removed, and the source form is kept as a spelling so a search for
    # either finds the record. A bracket standing alone ("[Briceut]") is simply the name.
    inline = re.search(r"(?<=[A-Za-z])\[[^\]]*\]|\[[^\]]*\](?=[A-Za-z])", working)
    if inline:
        # Record only the affected token, not the whole cell, so the variant is a name.
        for token in working.split():
            if "[" in token:
                out.variants.append((token.strip(" ,;"), "spelt"))
        working = working.replace("[", "").replace("]", "")
    elif working.startswith("[") and working.endswith("]") and working.count("[") == 1:
        # The transcriber bracketed the whole name because they were unsure of it.
        # That is uncertainty metadata, not a second spelling of the name.
        out.flags.append("uncertain")
        working = working[1:-1].strip()

    for pattern, cue in ((_BRACKET, "bracket"), (_PAREN, "paren"), (_QUOTED, "quoted")):
        for match in list(pattern.finditer(working)):
            inner = match.group(1).strip()
            working = working.replace(match.group(0), " ", 1)
            if not inner or inner == "?":
                if inner == "?":
                    # "Florette (?)" -- the source marked the name doubtful. That is
                    # uncertainty, not another spelling of the name.
                    out.flags.append("uncertain")
                continue
            if _CHILDREN_NOTE.match(inner):
                out.flags.append("annotated")
                continue
            if _SPELLING_UNCERTAIN.match(inner):
                out.flags.append("uncertain")
                continue
            if _INDIAN_NAME_LABEL.match(inner):
                continue
            if _ALSO_SPELT.search(inner):
                cleaned = _strip_prose(_ALSO_SPELT.sub("", inner))
                if cleaned:
                    out.variants.append((cleaned, "spelt"))
                continue
            if _ANNOTATION.search(inner):
                out.flags.append("annotated")
                if inner not in out.residue:
                    out.residue.append(inner)
                continue
            if _EDITORIAL.search(inner):
                out.flags.append("editorial")
                continue
            if _RELATIONSHIP.search(inner):
                out.flags.append("relationship")
                continue
            if looks_like_meaning(inner):
                out.meanings.append(_strip_meaning_label(inner))
                continue
            quoted = _QUOTED.search(inner)
            if quoted and pattern is _PAREN:
                # "(Christinanna \"Chris\")" carries two separate alternate names.
                out.variants.append((quoted.group(1).strip(), "quoted"))
                inner = _QUOTED.sub(" ", inner).strip()
                if not inner:
                    continue
            if _AKA.search(inner):
                for piece in _AKA.split(inner):
                    for token in _SPLITTERS.split(piece):
                        token = token.strip(" ,;")
                        if token:
                            out.variants.append((token, "aka"))
                continue
            if _OR.match(inner):
                stripped = _OR.sub("", inner).strip()
                if stripped:
                    out.variants.append((stripped, "or"))
                continue
            for token in _SPLITTERS.split(inner):
                token = token.strip(" ,;(").strip(")")
                if token.endswith("?"):
                    out.flags.append("uncertain")
                    token = token.rstrip("? ").strip()
                if token:
                    out.variants.append((token, cue))

    working = re.sub(r"\s+", " ", working).strip(" ,;")
    if working.endswith("?"):
        # "Duboos?" -- the transcriber doubted the reading. The mark is uncertainty,
        # not a second spelling.
        out.flags.append("uncertain")
        working = working.rstrip("? ").strip()
    if "*" in working:
        out.flags.append("uncertain")
        working = working.replace("*", "").strip()
    if _CREDENTIAL.search(working):
        out.flags.append("credential")
        working = _CREDENTIAL.sub(" ", working)
        working = re.sub(r"\s+", " ", working).strip(" ,;")

    # "Joanne or Yvonne" / "Richard JOHN or rickard francis" -- the alternative is a
    # different recorded name, not a middle name called "or".
    if _OR_SPLIT.search(working):
        pieces = [p.strip(" ,;") for p in _OR_SPLIT.split(working) if p.strip(" ,;")]
        if pieces:
            working = pieces[0]
            for piece in pieces[1:]:
                out.variants.append((piece, "or"))

    if _AKA.search(working):
        pieces = _AKA.split(working)
        working = pieces[0].strip()
        for piece in pieces[1:]:
            for token in _SPLITTERS.split(piece):
                token = token.strip(" ,;")
                if not token:
                    continue
                # "aka Jimmy Shabwunookuhum" names two alternates: an English nickname
                # and an Indigenous name. Split them the same way a plain run is split.
                run: list[str] = []
                for word in token.split():
                    if _is_middle_name(word):
                        if run:
                            out.variants.append((" ".join(run), "aka"))
                            run = []
                        out.variants.append((word, "aka"))
                    else:
                        run.append(word)
                if run:
                    out.variants.append((" ".join(run), "aka"))

    # Competing spellings are offered inside a single word ("Pat/Patrick H."), so the
    # split applies per token -- splitting the whole cell would drag the middle name
    # into the variant as "Patrick H.".
    # A separator written with a space beside it -- "Boissoneau/ Belleau" -- still joins
    # one token to another. Closing the gap first keeps the per-token split working;
    # without this the slash survives into the surname itself.
    working = re.sub(r"\s*/\s*", "/", working)

    rebuilt: list[str] = []
    for token in working.split():
        pieces = [p.strip() for p in _SPLITTERS.split(token) if p.strip()]
        if len(pieces) <= 1:
            rebuilt.append(token)
            continue
        # "Sheilah/Sheila", "Katheleen/Kathleen" -- the form written first is canonical.
        # The register's own order is the transcriber's judgement about which spelling
        # belongs to the person, and preferring the more familiar-looking side instead
        # silently overrode it: final_list_master.csv curates seven of these and wants the
        # first form in all seven. Every other form is kept as a spelling, so this decides
        # only which one is the label, never which ones can be found.
        chosen = pieces[0]
        rebuilt.append(chosen)
        for piece in pieces:
            if piece != chosen:
                out.variants.append((piece, "slash"))

    out.base = " ".join(rebuilt).strip(" ,;")
    return out


# ------------------------------------------------------------------ component parts


@dataclass
class _Component:
    """One parsed name component: its canonical form plus everything found beside it."""

    canonical: str = ""
    middle: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    spelling: list[str] = field(default_factory=list)
    meaning: list[str] = field(default_factory=list)
    alias: list[str] = field(default_factory=list)
    residue: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def place(self, base: str, variant: str, cue: str) -> None:
        bucket = classify_variant(base, variant, cue)
        target = {"spelling": self.spelling, "extra": self.extra}.get(bucket, self.alias)
        if variant not in target:
            target.append(variant)


def _component(text: str, *, split_middle: bool) -> _Component:
    """Parse one cell into a canonical form and its variants.

    ``split_middle`` distinguishes a given-name cell, where trailing tokens are usually
    middle names, from a surname cell, where the whole base is the surname.
    """
    out = _Component()
    parsed = decompose(text)
    out.flags.extend(parsed.flags)
    out.meaning.extend(parsed.meanings)
    out.residue.extend(parsed.residue)
    if not parsed.base:
        # Nothing survived as a canonical form, so every variant is simply an alias.
        for variant, _cue in parsed.variants:
            if variant not in out.alias:
                out.alias.append(variant)
        return out

    kept: list[str] = []
    for token in _explode_tokens(parsed.base.split()):
        lowered = _clean_token(token).lower()
        if lowered in TITLES:
            out.flags.append("titled")
            continue
        if lowered in SUFFIXES:
            # A generational suffix is not an alternate given name. Keep the exact
            # recorded form in a flag so it is not discarded.
            out.flags.append(f"suffix:{_clean_token(token)}")
            continue
        if _PROSE_WORD.fullmatch(lowered):
            # "Eliza Miksahbekuhnooqua (Wampum Strap) We called her Bessy" -- the trailing
            # sentence is narration about the person, not part of their name. Left in, its
            # words become searchable name tokens and the record turns up under "called".
            out.flags.append("narrative")
            continue
        normalized = _expand_token(token)
        if normalized:
            kept.append(normalized)
            # "Wm." stays canonical because that is what the register says; "William" is
            # filed beside it so a search for either form finds the person.
            expansion = _abbreviation_of(token)
            if expansion and expansion not in out.spelling:
                out.spelling.append(expansion)

    if not kept:
        return out

    if split_middle:
        out.canonical = kept[0]
        # A gloss in the cell is decisive: "Joseph Ogauns (Little Pickerel)" translates
        # Ogauns, so whatever precedes the gloss is the Indigenous name being explained,
        # not a middle name. Orthography alone misses short names like Naunge or Kewans.
        glossed = bool(parsed.meanings)
        # Consecutive tokens outside the given-name vocabulary form ONE Indigenous name
        # ("Weesug Pedequahum"), not two. A recognised given name always stands alone.
        run: list[str] = []
        for token in kept[1:]:
            # An initial is always a middle name, gloss or not. Absorbing it into the
            # run glued "Mary J. Owenshan" into a single alternate name of "J Owenshan".
            if _INITIAL.match(token):
                if run:
                    out.extra.append(" ".join(run))
                    run = []
                out.middle.append(token)
                continue
            if _is_middle_name(token) and not (glossed and token.strip(".").lower() not in COMMON_GIVEN_NAMES):
                if run:
                    out.extra.append(" ".join(run))
                    run = []
                out.middle.append(token)
            else:
                run.append(token)
        if run:
            out.extra.append(" ".join(run))
    else:
        out.canonical = " ".join(kept)

    for variant, cue in parsed.variants:
        out.place(out.canonical, variant, cue)
    return out


def _is_gloss_phrase(text: str) -> bool:
    """Whether text is an English translation rather than another proper name."""
    if _MEANING_LABEL.match(text.strip()):
        return True
    if not looks_like_meaning(text):
        return False
    words = _strip_meaning_label(text).split()
    return any(word.lower().strip(".,'\"") in _GLOSS_CUES for word in words)


def _split_trailing_meaning(text: str) -> tuple[str, str]:
    """Split 'Cheguans little man close by' into the name and a trailing English gloss."""
    words = text.split()
    if len(words) < 3:
        return text, ""
    for index in range(1, len(words)):
        head = " ".join(words[:index])
        tail = " ".join(words[index:])
        if head.lower().rstrip(".") in TITLES:
            continue
        if _is_gloss_phrase(tail):
            return head, _strip_meaning_label(tail)
    return text, ""


def _indigenous(text: str) -> _Component:
    """Parse an Indigenous-name cell.

    These use their own separators: a hyphen, comma, or newline introduces a gloss or a
    respelling where a Master-list cell would use parentheses. Newline-separated forms
    are distinct recorded variants, not one concatenated name.
    """
    out = _Component()
    working = (text or "").strip()
    if not working:
        return out

    lines = [ln.strip(" ,;") for ln in re.split(r"[\r\n]+", working) if ln.strip(" ,;")]
    primary = lines[0] if lines else working
    rest_lines = lines[1:]

    # "Shahwanenooqua-lady of the south wind" -- hyphen before an English phrase is a gloss.
    hyphen = re.split(r"\s*[-–—]\s*", primary, maxsplit=1)
    if len(hyphen) == 2 and looks_like_meaning(hyphen[1]):
        out.meaning.append(_strip_meaning_label(hyphen[1].strip(" ,")))
        primary = hyphen[0]

    # Ampersands join two complete recorded forms of the same person.
    ampersand_bits = [p.strip(" ,;") for p in re.split(r"\s*&\s*", primary) if p.strip(" ,;")]
    if len(ampersand_bits) > 1:
        primary = ampersand_bits[0]
        rest_lines = ampersand_bits[1:] + rest_lines

    pieces = [p.strip(" ,;") for p in primary.split(",") if p.strip(" ,;")]
    parsed = decompose(pieces[0] if pieces else primary)
    out.flags.extend(parsed.flags)
    out.meaning.extend(parsed.meanings)
    base, trailing = _split_trailing_meaning(parsed.base)
    if trailing:
        out.meaning.append(trailing)
    out.canonical = base
    for variant, cue in parsed.variants:
        out.place(out.canonical, variant, cue)
    # "Micowatch, Mecowatch, Mucowatch" -- comma-separated respellings of one name.
    further: list[str] = []
    for piece in pieces[1:] + rest_lines:
        further.extend(part.strip(" ,;") for part in piece.split(",") if part.strip(" ,;"))
    for piece in further:
        hyphenated = re.split(r"\s*[-–—]\s*", piece, maxsplit=1)
        if len(hyphenated) == 2 and looks_like_meaning(hyphenated[1]):
            out.meaning.append(_strip_meaning_label(hyphenated[1].strip(" ,")))
            piece = hyphenated[0]
        inner = decompose(piece)
        out.flags.extend(inner.flags)
        out.meaning.extend(inner.meanings)
        inner_base, inner_meaning = _split_trailing_meaning(inner.base)
        if inner_meaning:
            out.meaning.append(inner_meaning)
        if inner_base:
            out.place(out.canonical or inner_base, inner_base, "slash")
            if not out.canonical:
                out.canonical = inner_base
        for variant, cue in inner.variants:
            out.place(out.canonical, variant, cue)
    if out.canonical:
        kept_tokens: list[str] = []
        for token in out.canonical.split():
            if token.lower() in TITLES:
                out.flags.append("titled")
            else:
                kept_tokens.append(token)
        out.canonical = " ".join(kept_tokens)
    return out


# ------------------------------------------------------------------ whole records


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for value in values:
        cleaned = value.strip(" ,;")
        if cleaned:
            seen.setdefault(cleaned, None)
    return tuple(seen)


def _parse_master_base(
    first: str = "",
    middle: str = "",
    last: str = "",
    indigenous: str = "",
) -> NameParts:
    """Parse the Master list, whose given/middle/surname already sit in their own cells."""
    given = _component(first, split_middle=True)
    surname = _component(last, split_middle=False)
    extra_middle = _component(middle, split_middle=True) if middle.strip() else _Component()
    native = _indigenous(indigenous) if indigenous.strip() else _Component()

    middles = list(given.middle)
    if extra_middle.canonical:
        middles.append(extra_middle.canonical)
    middles.extend(extra_middle.middle)

    first = given.canonical
    alias = list(given.alias + surname.alias + extra_middle.alias + native.alias)
    flags = list(given.flags + surname.flags + extra_middle.flags + native.flags)
    # "Mrs. T.(Children) Nancy" -- Mrs. and Children are metadata; the initial is not
    # confidently this person's given name, so the recognisable given name is preferred.
    if (
        "titled" in flags
        and "annotated" in flags
        and first
        and _INITIAL.match(first)
        and middles
        and middles[0].strip(".").lower() in COMMON_GIVEN_NAMES
    ):
        alias.append(first)
        first = middles.pop(0)
        flags.append("review")

    return NameParts(
        first=first,
        middle=_dedupe(middles),
        last=surname.canonical,
        extra_first=_dedupe(given.extra + extra_middle.extra),
        extra_last=_dedupe(surname.extra),
        first_spelling=_dedupe(given.spelling),
        last_spelling=_dedupe(surname.spelling),
        first_meaning=_dedupe(given.meaning),
        last_meaning=_dedupe(surname.meaning),
        indigenous=native.canonical,
        indigenous_spelling=_dedupe(native.spelling + native.extra),
        indigenous_meaning=_dedupe(native.meaning),
        alias=_dedupe(alias),
        residue=_dedupe(
            given.residue + surname.residue + extra_middle.residue + native.residue
        ),
        flags=_dedupe(flags),
    )


_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
# A date or a community fused straight onto a name with no separator, as in
# "Edward ShingwaukFebruary 11, 1905Garden River First Nation".
_FUSED = re.compile(rf"(?<=[a-z])(?=(?:{_MONTHS})\b)", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_UPPER_RUN = re.compile(r"\b[A-Z][A-Z'’.-]{1,}\b")
_DIED_NOTE = re.compile(r"\(?\s*Died:\s*[^)]*\)?", re.IGNORECASE)
_FROM_COMMUNITY = re.compile(
    r"\s+\S{0,6}\s*From\b(?:\s+the\b)?\s+.+$",
    re.IGNORECASE,
)
_GLOSS_CUES = frozenset({
    "of", "in", "the", "by", "a", "an", "and", "to", "from", "with", "on",
    "little", "big", "small", "close", "running", "stand", "speaks", "lady",
    "man", "foot", "wind", "light", "flower", "stone", "pairs", "middle",
})
_FAMILY_NARRATIVE = re.compile(
    r"^The\s+(\w+)\s+Family:\s+(\w+)\s+was\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(r"^TO BE REVIEWED$", re.IGNORECASE)
# Words that describe the record rather than the person. On the death lists the surname is
# written in capitals, so a capitalised status note -- "(STUDENTS REMOVED FROM NCTR)" --
# otherwise looks exactly like a surname and was being stored as one.
_STATUS_WORD = re.compile(
    r"\b(?:students?|removed|remove|source|nctr|cirnac|registrar|vital|stats?|census|"
    r"archive|office|unknown|listed|review(?:ed)?|pending|duplicate|see|also|from|the)\b",
    re.IGNORECASE,
)


def _is_status_phrase(text: str) -> bool:
    """Whether an all-capitals run is a note about the record, not a surname.

    A real surname may coincide with one of these words, so a single token is left alone;
    it takes two or more status words, or a status word beside a preposition, before the
    run is treated as a note.
    """
    words = [w for w in re.split(r"[^A-Za-z]+", text or "") if w]
    if len(words) < 2:
        return False
    return sum(1 for w in words if _STATUS_WORD.fullmatch(w)) >= 2
_EDITORIAL_TAIL = re.compile(
    r"\s*[-–—]+\s*\(([^)]*(?:moved to|rsdit|see also)[^)]*)\)\s*$",
    re.IGNORECASE,
)


def split_fused_name(text: str) -> tuple[str, str]:
    """Return (name, trailing junk) for a cell with another field fused onto the name."""
    pieces = _FUSED.split(text, maxsplit=1)
    if len(pieces) == 2:
        return pieces[0].strip(), pieces[1].strip()
    return text.strip(), ""


def extract_person_name(text: str) -> tuple[str, list[str], list[str]]:
    """Pull a personal name out of a cell that may also hold dates, communities, or notes.

    Returns the name, its quality flags, and the residue -- every fragment removed on the
    way. The residue is kept rather than discarded so the cell can be reconstructed and
    reviewed, and so a community or a date can never be mistaken for part of the name.
    """
    flags: list[str] = []
    residue: list[str] = []
    working = (text or "").strip()
    if not working:
        return "", flags, residue

    def keep(fragment: str) -> None:
        cleaned = re.sub(r"\s+", " ", fragment or "").strip(" ,;.:-—–()\t")
        if cleaned and cleaned not in residue:
            residue.append(cleaned)

    if _PLACEHOLDER.match(working):
        return "", ["placeholder"], [working]

    family = _FAMILY_NARRATIVE.match(working)
    if family:
        flags.extend(["narrative", "review"])
        # The sentence names the person and their family; everything else is context.
        keep(working)
        return f"{family.group(2)} {family.group(1)}", flags, residue
    if re.match(r"^The\s+\w+\s+Family\b", working, re.I):
        return "", ["narrative", "review"], [working]

    editorial = _EDITORIAL_TAIL.search(working)
    if editorial:
        flags.append("editorial")
        keep(editorial.group(0))
        working = working[: editorial.start()].strip()

    if "\t" in working:
        parts = [p.strip() for p in working.split("\t") if p.strip()]
        for part in parts[1:]:
            keep(part)
        working = parts[0] if parts else working
        flags.append("concatenated")

    died = _DIED_NOTE.search(working)
    if died:
        flags.append("annotated")
        keep(died.group(0))
        working = _DIED_NOTE.sub(" ", working)

    community = _FROM_COMMUNITY.search(working)
    if community:
        flags.append("annotated")
        keep(community.group(0))
        working = _FROM_COMMUNITY.sub("", working)

    name, junk = split_fused_name(working)
    if junk:
        flags.append("concatenated")
        keep(junk)
        working = name

    working = re.sub(r"\s+", " ", working).strip(" ,;.-—–")
    return working, flags, residue


# A date written the way name cells write one: "January 21, 1905", "Sept. 3rd 1911".
_CELL_DATE = rf"(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}"
_LEADING_DATE = re.compile(rf"^[^A-Za-z0-9]*{_CELL_DATE}\s*", re.IGNORECASE)
_DATE_ONLY = re.compile(rf"^{_CELL_DATE}$", re.IGNORECASE)
_FROM_MARKER = re.compile(r"^[^A-Za-z]*from\s+(?:the\s+)?", re.IGNORECASE)


def place_from_residue(parts: NameParts) -> str:
    """The community a name cell carries, when the cell's own structure says it is one.

    Some lists were filled by pasting a whole row into the name cell: "Name<TAB>date<TAB>
    community", the same run together with no separators, or "Name (Died: date) -- From X
    First Nation". The parser already sets those fragments aside as residue so they never
    become part of a name. The community in them is still a fact about the person, and where
    the community column is blank for that row it is the only record of where they came from.

    Only structure is trusted, never vocabulary: a fragment is read as a place when it follows
    a date from the same pasted row, or is introduced by "From". A parenthetical gloss -- an
    English rendering of an Indigenous name -- has neither, and is never taken for a place.
    """
    if {"placeholder", "narrative"} & set(parts.flags):
        return ""
    residue = [" ".join(item.split()) for item in parts.residue]
    for index, fragment in enumerate(residue):
        if not fragment or _DIED_NOTE.fullmatch(fragment) or fragment.casefold().startswith("died"):
            continue
        candidate = ""
        dated = _LEADING_DATE.match(fragment)
        if dated and fragment[dated.end():].strip():
            candidate = fragment[dated.end():]
        elif _FROM_MARKER.match(fragment):
            candidate = fragment
        elif index > 0 and "concatenated" in parts.flags and _DATE_ONLY.match(residue[index - 1]):
            candidate = fragment
        marker = _FROM_MARKER.match(candidate)
        if marker:
            candidate = candidate[marker.end():]
        place = _clean_place(candidate)
        if _looks_like_place(place):
            return place
    return ""


def _clean_place(text: str) -> str:
    place = " ".join((text or "").split()).strip(" .,;:-—–")
    place = re.sub(r"\s+community$", "", place, flags=re.IGNORECASE).strip(" .,;:")
    if place.count("(") > place.count(")"):
        # The residue trims a closing bracket off "Muncey (Chippewas of the Thames)".
        place += ")"
    # "First Nation" says what kind of place it is, not which one. The community column on
    # the same list writes "Walpole Island", and the two have to group together.
    return re.sub(r"\s+First\s+Nation$", "", place, flags=re.IGNORECASE).strip(" .,;:")


def _looks_like_place(text: str) -> bool:
    return (
        bool(text)
        and text[0].isupper()
        and not re.search(r"[\d:]", text)
        and len(text.split()) <= 8
    )


def _looks_like_english_personal_name(text: str) -> bool:
    """Whether a cell looks like a complete English personal name rather than an Indigenous one."""
    words = [w.strip(".,") for w in text.split() if w.strip(".,")]
    if len(words) < 2:
        return False
    if any(_INDIGENOUS_SHAPE.search(w) for w in words):
        return False
    lowered = [w.lower() for w in words]
    return all(w in COMMON_GIVEN_NAMES or _INITIAL.match(word) for w, word in zip(lowered, words, strict=False))


def _reclassify_indian_name(
    native: _Component,
    given: _Component,
    surname: _Component,
) -> tuple[list[str], list[str], list[str], str, list[str], list[str]]:
    """Move Indian-name pieces that are clearly English nicknames or surname variants.

    The source column is historically mixed. A value is left as Indigenous unless it
    is recognisably an English given name, a respelling of the surname, or a complete
    English personal name.
    """
    extra_first: list[str] = []
    last_spelling: list[str] = []
    alias: list[str] = []
    flags: list[str] = []

    candidates: list[str] = []
    if native.canonical:
        candidates.append(native.canonical)
    candidates.extend(native.spelling)
    candidates.extend(native.extra)
    candidates.extend(native.alias)

    kept: list[str] = []
    titled = "titled" in native.flags
    for value in candidates:
        cleaned = value.strip()
        lowered = cleaned.lower()
        if not cleaned:
            continue
        tokens = cleaned.split()
        if len(tokens) == 1 and lowered in COMMON_GIVEN_NAMES:
            extra_first.append(cleaned)
            continue
        if given.canonical and is_alternate_name(given.canonical, cleaned):
            extra_first.append(cleaned)
            continue
        if native.meaning and cleaned == native.canonical:
            # The form that carries a translation is the Indigenous name, even when it
            # also resembles the English surname.
            kept.append(cleaned)
            continue
        if (
            surname.canonical
            and _ratio(surname.canonical, cleaned) >= 0.88
            and not _INDIGENOUS_SHAPE.search(cleaned)
        ):
            last_spelling.append(cleaned)
            continue
        if titled or _looks_like_english_personal_name(cleaned):
            alias.append(cleaned)
            flags.append("review")
            continue
        if (
            len(tokens) >= 2
            and tokens[0].lower() in COMMON_GIVEN_NAMES
            and not any(_INDIGENOUS_SHAPE.search(token) for token in tokens)
        ):
            alias.append(cleaned)
            flags.append("review")
            continue
        kept.append(cleaned)

    indigenous = ""
    indigenous_spelling: list[str] = []
    leftover: list[str] = []
    for value in kept:
        if _INDIGENOUS_SHAPE.search(value):
            if not indigenous:
                indigenous = value
            else:
                indigenous_spelling.append(value)
        else:
            leftover.append(value)
    for value in leftover:
        if not indigenous:
            indigenous = value
            continue
        if _INDIGENOUS_SHAPE.search(indigenous):
            if surname.canonical and _ratio(surname.canonical, value) >= 0.7:
                last_spelling.append(value)
            else:
                alias.append(value)
                flags.append("review")
        else:
            indigenous_spelling.append(value)
    return extra_first, last_spelling, alias, indigenous, indigenous_spelling, flags


def parse_full_name(full: str = "", indigenous: str = "") -> NameParts:
    """Parse a single ``STUDENT NAME`` cell, as the death lists store it.

    The surname is written in capitals on these lists -- always on Confirmed, usually on
    Additional, and only about half the time on Potential. Where that cue is absent the
    surname is taken from the ``Last, First`` form or, failing that, from the trailing
    token, and the result is flagged as inferred rather than presented as certain.
    """
    flags: list[str] = []
    working = (full or "").strip()
    native = _indigenous(indigenous) if indigenous.strip() else _Component()
    if not working:
        return NameParts(
            indigenous=native.canonical,
            indigenous_spelling=_dedupe(native.spelling + native.extra),
            indigenous_meaning=_dedupe(native.meaning),
            alias=_dedupe(native.alias),
            flags=_dedupe(native.flags),
        )

    working, extracted_flags, residue = extract_person_name(working)
    flags.extend(extracted_flags)
    if extracted_flags == ["placeholder"] or not working:
        return NameParts(
            indigenous=native.canonical,
            indigenous_spelling=_dedupe(native.spelling + native.extra),
            indigenous_meaning=_dedupe(native.meaning),
            alias=_dedupe(native.alias),
            residue=_dedupe(residue),
            flags=_dedupe(flags + list(native.flags)),
        )

    parsed = decompose(working)
    flags.extend(parsed.flags)
    residue.extend(parsed.residue)
    base = parsed.base

    given_text, surname_text = "", ""
    if "," in base:
        # "Betsey Jane, WHITE" and "Daniel, MICHEL" put the surname after the comma.
        head, _, tail = base.partition(",")
        given_text, surname_text = head.strip(), tail.strip()
    else:
        uppers = [m for m in _UPPER_RUN.finditer(base) if m.group(0).upper() == m.group(0)]
        capitalised = " ".join(m.group(0) for m in uppers) if uppers else ""
        if uppers and _is_status_phrase(capitalised):
            # "Annie Howe (REMOVED) (STUDENTS REMOVED FROM NCTR)" -- the capitals describe
            # the record, not the person. Set them aside and read the name normally.
            residue.append(capitalised)
            flags.append("annotated")
            for match in reversed(uppers):
                base = base[: match.start()] + " " + base[match.end():]
            # Removing the words leaves the brackets that held them; an orphan "(" would
            # otherwise be read as the surname.
            base = re.sub(r"[(\[]\s*[)\]]", " ", base)
            base = re.sub(r"[()\[\]]", " ", base)
            base = re.sub(r"\s+", " ", base).strip(" ,;.-")
            uppers = []
        if uppers:
            surname_text = capitalised
            given_text = base[: uppers[0].start()].strip()
        else:
            tokens = base.split()
            if len(tokens) >= 2:
                surname_text, given_text = tokens[-1], " ".join(tokens[:-1])
                flags.append("inferred_surname")
            else:
                given_text = base
    residue = list(residue)

    given = _component(given_text, split_middle=True)
    surname = _component(surname_text, split_middle=False)

    # Variants found before the split belong to whichever component they resemble.
    extra_first: list[str] = list(given.extra)
    extra_last: list[str] = list(surname.extra)
    first_spelling: list[str] = list(given.spelling)
    last_spelling: list[str] = list(surname.spelling)
    alias: list[str] = list(given.alias) + list(surname.alias)
    for variant, cue in parsed.variants:
        anchor_first = _ratio(given.canonical, variant)
        anchor_last = _ratio(surname.canonical, variant)
        anchor = given.canonical if anchor_first >= anchor_last else surname.canonical
        bucket = classify_variant(anchor, variant, cue)
        to_first = anchor_first >= anchor_last
        if bucket == "spelling":
            (first_spelling if to_first else last_spelling).append(variant)
        elif bucket == "extra":
            (extra_first if to_first else extra_last).append(variant)
        else:
            alias.append(variant)

    moved_first, moved_last, moved_alias, indigenous, indigenous_spelling, moved_flags = (
        _reclassify_indian_name(native, given, surname)
    )
    extra_first.extend(moved_first)
    last_spelling.extend(moved_last)
    alias.extend(moved_alias)
    flags.extend(moved_flags)
    if _YEAR.search(full or ""):
        flags.append("date_in_name")

    return NameParts(
        first=given.canonical,
        middle=_dedupe(given.middle),
        last=surname.canonical,
        extra_first=_dedupe(extra_first),
        extra_last=_dedupe(extra_last),
        first_spelling=_dedupe(first_spelling),
        last_spelling=_dedupe(last_spelling),
        first_meaning=_dedupe(given.meaning + parsed.meanings),
        last_meaning=_dedupe(surname.meaning),
        indigenous=indigenous,
        indigenous_spelling=_dedupe(indigenous_spelling),
        indigenous_meaning=_dedupe(native.meaning),
        alias=_dedupe(alias),
        residue=_dedupe(residue),
        flags=_dedupe(flags + given.flags + surname.flags + native.flags),
    )

# ============================================================================
# Master-list normalization rules
#
# The Master list encodes additional semantics inside the as-recorded first/last
# cells.  These rules remain conservative while handling three recurring structures:
#   * contextual Indigenous name + gloss pairs,
#   * ordinary aliases/nicknames vs spelling variants,
#   * at most one normalized middle name (remaining plain names are extras).
#
# The rules are structural/morphological. They contain no row IDs, person-specific
# exceptions, or source-row lookup tables.

# Additional conventional English/French given-name forms missing from the compact
# vocabulary above. They are used only as generic language evidence; no row IDs or
# record-specific mappings are stored here.
COMMON_GIVEN_NAMES = frozenset(
    set(COMMON_GIVEN_NAMES)
    | {
        "sheila",
        "henrietta",
        "maureen",
        "lorrine",
        "pat",
        "garry",
        "billie",
        "harley",
        "lyla",
        "sebastian",
        "christopher",
        "morris",
    }
)

# Generic transliteration markers used in addition to the core Indigenous-shape regex.
# The score is based on orthographic features, not complete names or row-specific substrings.
_MASTER_RARE_TRANSLITERATION = re.compile(r"zh|kw|gw|hj", re.IGNORECASE)
_MASTER_COMMON_TRANSLITERATION = re.compile(r"sh|oo|ee|aa|ou|ah|eh|ih|oh", re.IGNORECASE)


def _master_transliteration_score(text: str) -> int:
    cleaned = re.sub(r"[^A-Za-z'-]", "", text or "").lower()
    if not cleaned or cleaned.strip(".") in COMMON_GIVEN_NAMES:
        return 0
    return (
        2 * len(_MASTER_RARE_TRANSLITERATION.findall(cleaned))
        + len(_MASTER_COMMON_TRANSLITERATION.findall(cleaned))
    )


def _master_strong_native_shape(text: str) -> bool:
    """Morphology-only Indigenous cue based on generic transliteration features."""
    return bool(_INDIGENOUS_SHAPE.search(text or "")) or _master_transliteration_score(text) >= 2


def _master_native_shape(text: str) -> bool:
    """Broad Indigenous-name cue used for an unlabelled trailing identity."""
    bare = re.sub(r"[^A-Za-z]", "", text or "")
    return _master_strong_native_shape(text) or len(bare) >= 9


def _master_is_known_given(token: str) -> bool:
    cleaned = token.strip(".").lower()
    return bool(_INITIAL.match(token)) or cleaned in COMMON_GIVEN_NAMES


def _master_is_plain_alternate(base: str, variant: str) -> bool:
    """Different given name rather than a middle name/spelling in a plain run."""
    if is_alternate_name(base, variant):
        return True
    left = base.lower().strip(".")
    right = variant.lower().strip(".")
    short, long_ = sorted((left, right), key=len)
    # Short-form/full-form pair such as Pat/Patrick.
    if len(short) == 3 and long_.startswith(short) and len(long_) - len(short) <= 4:
        return True
    # Canonical full form followed by its short form, e.g. Frederick Fred.
    return 3 <= len(right) <= 4 and left.startswith(right) and len(left) - len(right) <= 5


def _master_is_slash_extra(base: str, variant: str) -> bool:
    """Whether a slash-separated given-name form is another name, not a spelling."""
    if _master_is_plain_alternate(base, variant):
        return True
    return _ratio(base, variant) < 0.60


def _master_parentheticals(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\(([^)]*)\)", text or "")]


def _master_source_order(source: str, value: str) -> int:
    match = re.search(
        r"(?<![A-Za-z])" + re.escape(value) + r"(?![A-Za-z])",
        source or "",
        re.IGNORECASE,
    )
    return match.start() if match else 10**9


def _master_reference_parts(
    first: str,
    middle: str,
    last: str,
    indigenous: str,
) -> NameParts:
    """Parse Master-list cells with generic structural and morphological rules.

    No row/person exceptions are used. A plain two-token first-name cell can remain
    semantically ambiguous when its second token has no punctuation, gloss, title,
    initial, known given-name form, or Indigenous-language morphology. Such rows use the
    conservative default (middle name); explicit structural evidence always takes priority.
    """
    given_dec = decompose(first)
    surname_dec = decompose(last)

    # Capture a trailing narrative alias such as "We called her Bessy" before
    # tokenizing the base.  The generic parser flags prose but can otherwise fuse
    # the surviving name onto the preceding Indigenous identity.
    narrative_extra: list[str] = []
    narrative = re.search(
        r"\b(?:we|they|she|he)\s+(?:also\s+)?called\s+(?:her|him|them)?\s*"
        r"([A-Za-z][A-Za-z'’\-]*)\.?\s*$",
        first or "",
        re.IGNORECASE,
    )
    given_base = given_dec.base
    if narrative:
        narrative_extra.append(narrative.group(1))
        given_base = re.sub(
            r"\b(?:we|they|she|he)\s+(?:also\s+)?called\s+(?:her|him|them)?\s*"
            r"[A-Za-z][A-Za-z'’\-]*\.?\s*$",
            "",
            given_base,
            flags=re.IGNORECASE,
        ).strip()

    # ------------------------------- canonical first / tail tokens
    given_tokens: list[str] = []
    given_expansions: list[str] = []
    suffixes: list[str] = []
    titled = False
    for token in _explode_tokens(given_base.split()):
        lowered = _clean_token(token).lower()
        if lowered in TITLES:
            titled = True
            continue
        if lowered in SUFFIXES:
            suffixes.append(_clean_token(token))
            continue
        if _PROSE_WORD.fullmatch(lowered):
            continue
        normalized = _expand_token(token)
        if normalized:
            given_tokens.append(normalized)
            expansion = _abbreviation_of(token)
            if expansion:
                given_expansions.append(expansion)

    canonical_first = given_tokens[0] if given_tokens else ""
    tail = given_tokens[1:]

    # Metadata case: "Mrs. T.(Children) Nancy".  The initial is not confidently
    # the person's first name; the recognisable following given name is.
    if (
        re.search(r"\bMrs\.?", first or "", re.IGNORECASE)
        and re.search(r"\(\s*Children\s*\)", first or "", re.IGNORECASE)
        and canonical_first
        and _INITIAL.match(canonical_first)
        and tail
    ):
        canonical_first = tail[0]
        tail = tail[1:]

    comma_after_first = bool(re.match(r"^\s*[^,]+,\s*", given_dec.base or ""))

    # ------------------------------- Indigenous identity embedded in first cell
    meaning_candidates = list(given_dec.meanings)
    native_plain: list[str] = []
    if tail:
        if meaning_candidates:
            # With a gloss present, an unrecognised final run is the identity being
            # translated.  Stop at a known English/French given name or initial.
            split = len(tail)
            while split > 0:
                token = tail[split - 1]
                if _master_is_known_given(token):
                    break
                split -= 1
            if split < len(tail):
                native_plain = tail[split:]
        if not native_plain:
            # Without a gloss, require morphology/length evidence before moving a
            # plain trailing token out of the middle-name position.
            split = len(tail)
            run: list[str] = []
            while split > 0 and (not _is_middle_name(tail[split - 1]) or _master_strong_native_shape(tail[split - 1])):
                run.insert(0, tail[split - 1])
                split -= 1
            if run and _master_native_shape(" ".join(run)):
                native_plain = run

    native_name = " ".join(native_plain)

    if not native_name and meaning_candidates:
        for variant, cue in given_dec.variants:
            if (
                (cue == "bracket" and _master_native_shape(variant))
                or (cue in {"paren", "aka", "quoted"} and _master_strong_native_shape(variant))
            ):
                native_name = variant.strip("[]")
                break

    if not native_name:
        for variant, cue in given_dec.variants:
            if cue in {"paren", "aka"} and _master_strong_native_shape(variant):
                native_name = variant
                break

    # ------------------------------- middle vs extra-first
    middles: list[str] = []
    extra_first: list[str] = []
    explicit_middle = bool((middle or "").strip())
    native_tokens = set(native_plain)
    for token in tail:
        if token in native_tokens:
            continue
        if _is_middle_name(token):
            if (
                _master_is_plain_alternate(canonical_first, token)
                or comma_after_first
                or explicit_middle
                or middles
            ):
                extra_first.append(token)
            else:
                middles.append(token)
        else:
            extra_first.append(token)


    if explicit_middle:
        # A dedicated middle-name source column is authoritative.  Any middle-like
        # token also found in the first-name cell is therefore another first-name fact.
        extra_first = middles + extra_first
        middles = []
        middle_component = _component(middle, split_middle=True)
        if middle_component.canonical:
            middles.append(middle_component.canonical)
        middles.extend(middle_component.middle)
        extra_first.extend(middle_component.extra)

    extra_first.extend(narrative_extra)
    first_spelling: list[str] = list(given_expansions)
    indigenous_spelling: list[str] = []

    # A single-word parenthetical after a detected Indigenous identity can be a
    # gloss (Deer, Earthquake, Successor, etc.), unless it is a known given name.
    native_single_meanings: list[str] = []
    if native_name:
        for variant, cue in given_dec.variants:
            cleaned = variant.strip()
            if (
                cue == "paren"
                and cleaned != native_name
                and " " not in cleaned
                and cleaned.lower() not in COMMON_GIVEN_NAMES
                and not _master_is_plain_alternate(canonical_first, cleaned)
            ):
                native_single_meanings.append(cleaned)

    for variant, cue in given_dec.variants:
        variant = variant.strip()
        if not variant or variant in native_single_meanings:
            continue
        if native_name and variant == native_name:
            continue

        if native_name:
            variant_clean = variant.replace("[", "").replace("]", "")
            if (
                _ratio(native_name, variant_clean) >= 0.60
                and (cue in {"bracket", "spelt"} or _master_native_shape(variant_clean))
            ):
                indigenous_spelling.append(variant)
                continue

        if cue == "slash":
            if _master_is_slash_extra(canonical_first, variant):
                extra_first.append(variant)
            else:
                first_spelling.append(variant)
        elif cue in {"paren", "quoted", "aka", "or"}:
            extra_first.append(variant)
        elif cue in {"bracket", "spelt"}:
            first_spelling.append(variant)
        else:
            extra_first.append(variant)

    # ------------------------------- surname
    surname_tokens: list[str] = []
    surname_expansions: list[str] = []
    for token in _explode_tokens(surname_dec.base.split()):
        lowered = _clean_token(token).lower()
        if lowered in TITLES or lowered in SUFFIXES:
            continue
        normalized = _expand_token(token)
        if normalized:
            surname_tokens.append(normalized)
            expansion = _abbreviation_of(token)
            if expansion:
                surname_expansions.append(expansion)

    canonical_last = surname_tokens[0] if surname_tokens else ""
    extra_last: list[str] = surname_tokens[1:]
    last_spelling: list[str] = list(surname_expansions)

    last_native = ""
    last_meaning: list[str] = []
    if surname_dec.meanings:
        # A multiword English gloss in the surname cell is decisive.
        last_native = canonical_last
        last_meaning.extend(surname_dec.meanings)
    else:
        # A one-word gloss is only accepted when the surname has strong Indigenous
        # morphology and is not similar enough to be a spelling variant.
        for inner in _master_parentheticals(last):
            cleaned = inner.strip().rstrip("?").strip()
            if (
                cleaned
                and " " not in cleaned
                and _master_strong_native_shape(canonical_last)
                and _ratio(canonical_last, cleaned) < 0.60
                and cleaned.lower() not in COMMON_GIVEN_NAMES
                and not _ALSO_SPELT.search(inner)
                and not _OR.match(inner)
                and not _SPELLING_UNCERTAIN.match(inner)
            ):
                last_native = canonical_last
                last_meaning.append(cleaned)
                break

    for variant, cue in surname_dec.variants:
        variant = variant.strip()
        if not variant:
            continue
        if cue in {"spelt", "bracket"}:
            last_spelling.append(variant)
        elif cue == "slash":
            if _ratio(canonical_last, variant) >= 0.60:
                last_spelling.append(variant)
            else:
                extra_last.append(variant)
        elif cue == "paren":
            if _ratio(canonical_last, variant) >= 0.60:
                last_spelling.append(variant)
            else:
                extra_last.append(variant)
        elif cue == "or":
            extra_last.append(variant)
        elif _ratio(canonical_last, variant) >= 0.60:
            last_spelling.append(variant)
        else:
            extra_last.append(variant)

    if canonical_last and (
        (last or "").strip().endswith("?")
        or any(_SPELLING_UNCERTAIN.match(x.strip()) for x in _master_parentheticals(last))
    ):
        last_spelling.append(canonical_last)

    if last_native:
        extra_last = [value for value in extra_last if value not in last_meaning]

    # A gloss can be written in the first cell while the Indigenous identity sits
    # in the surname cell (e.g. "David (Stand in the Middle)" + Indigenous surname).
    if not native_name and not last_native and meaning_candidates and _master_native_shape(canonical_last):
        last_native = canonical_last
        last_meaning.extend(meaning_candidates)
        meaning_candidates = []

    indigenous_name = native_name or last_native
    indigenous_meaning: list[str] = []
    if native_name:
        # Context wins over generic prose classification: phrases such as "My Son"
        # and "Child of the Wind" are glosses when they follow an Indigenous name.
        for inner in _master_parentheticals(first):
            cleaned = inner.strip().strip('"“”')
            if (
                not cleaned
                or cleaned == "?"
                or cleaned == native_name
                or _SPELLING_UNCERTAIN.match(cleaned)
                or _AKA.search(cleaned)
                or _CHILDREN_NOTE.match(cleaned)
                or _ANNOTATION.search(cleaned)
            ):
                continue
            if " " not in cleaned and cleaned.lower() in COMMON_GIVEN_NAMES:
                continue
            if _QUOTED.search(cleaned) and not looks_like_meaning(cleaned):
                continue
            indigenous_meaning.append(_strip_meaning_label(cleaned))
        indigenous_meaning.extend(meaning_candidates)
        indigenous_meaning.extend(native_single_meanings)
    elif last_native:
        indigenous_meaning.extend(last_meaning)

    # If no Indigenous identity exists, a multiword parenthetical that the generic
    # English-phrase heuristic called a "meaning" is simply an alternate first name.
    if not indigenous_name and meaning_candidates:
        extra_first.extend(meaning_candidates)

    # A true raw Indigenous-name source, when present, is only a fallback.  The
    # Master reference can be regenerated from first/middle/last alone.
    if indigenous.strip() and not indigenous_name:
        native_component = _indigenous(indigenous)
        indigenous_name = native_component.canonical
        indigenous_spelling.extend(native_component.spelling + native_component.extra)
        indigenous_meaning.extend(native_component.meaning)

    # Inline editorial insertion: Negaunaus[d]enooqua -> canonical without brackets,
    # with the as-recorded token preserved as an Indigenous spelling.  A wholly
    # bracketed identity is uncertainty metadata, not a second spelling.
    if native_name:
        for token in (first or "").split():
            if (
                "[" in token
                and "]" in token
                and not (token.startswith("[") and token.endswith("]"))
            ):
                expanded = token.replace("[", "").replace("]", "").strip(" ,;")
                if expanded == native_name and token.strip(" ,;") not in indigenous_spelling:
                    indigenous_spelling.append(token.strip(" ,;"))

    extra_first = sorted(
        list(_dedupe(extra_first)),
        key=lambda value: _master_source_order(first, value),
    )

    legacy = _parse_master_base(first=first, middle=middle, last=last, indigenous=indigenous)
    classified = {
        canonical_first,
        canonical_last,
        indigenous_name,
        *middles,
        *extra_first,
        *extra_last,
        *first_spelling,
        *last_spelling,
        *indigenous_spelling,
    }
    alias = [value for value in legacy.alias if value not in classified]

    flags = list(legacy.flags)
    if titled and "titled" not in flags:
        flags.append("titled")
    for suffix in suffixes:
        # "Charles Sr." records a generation, not a second given name. Left in extra_first
        # it became a searchable name token, so every "Sr." row answered a search for
        # someone called Sr. The tokeniser in _component already keeps the exact recorded
        # form in a flag and drops it; this path now agrees with it.
        marker = f"suffix:{suffix}"
        if marker not in flags:
            flags.append(marker)

    return NameParts(
        first=canonical_first,
        middle=_dedupe(middles),
        last=canonical_last,
        extra_first=_dedupe(extra_first),
        extra_last=_dedupe(extra_last),
        first_spelling=_dedupe(first_spelling),
        last_spelling=_dedupe(last_spelling),
        first_meaning=(),
        last_meaning=(),
        indigenous=indigenous_name,
        indigenous_spelling=_dedupe(indigenous_spelling),
        indigenous_meaning=_dedupe(indigenous_meaning),
        alias=_dedupe(alias),
        residue=legacy.residue,
        flags=_dedupe(flags),
    )


# --------------------------------------------------------------- labelled comments

# The standardized Master list moves out of the name cells everything that is not the
# name: "Sahguj (also spelt Saguj)" is now "Sahguj" plus a comment reading
# "Also spelled: Saguj". The facts are the same ones this module already recognises
# inside parentheses, so they belong in the same NameParts slots; the difference is that
# the source now says which is which, and a stated label is better evidence than the
# shape of a word. Nothing here is specific to a phrasing: the label is matched on the
# words it contains, and where a label leaves the destination open -- a spelling of
# which name? -- the value is compared with the names actually parsed.

_COMMENT_SEPARATOR = re.compile(r"\s*;\s*")
_COMMENT_LABEL = re.compile(r"^(?P<label>[^:]{1,60}?)\s*:\s*(?P<value>.+)$", re.DOTALL)
_COMMENT_AKA = re.compile(r"^a\.?\s?k\.?\s?a\.?\s+(?P<value>.+)$", re.IGNORECASE)
_COMMENT_LEAD_CUE = re.compile(
    r"^\s*(?:also\s+|sometimes\s+|originally\s+)?(?:spel[lt]\w*|recorded|written|known)"
    r"(?:\s+as)?\s+",
    re.IGNORECASE,
)


def _comment_slot(label: str) -> str:
    """Which NameParts slot a comment label names, or '' when it names none."""
    text = label.lower()
    given = "first" in text or "given" in text
    family = "last" in text or "surname" in text or "family" in text
    native = any(word in text for word in ("indian", "indigenous", "spirit", "native"))
    # "Original first name recorded as" names a spelling of that part, not a second
    # given name. Checked before the bare "first"/"last" branches.
    if "recorded as" in text or "written as" in text:
        return "spelling"
    if "meaning" in text or "translat" in text:
        if given:
            return "first_meaning"
        if family:
            return "last_meaning"
        return "indigenous_meaning"
    if native:
        return "indigenous"
    if "spel" in text:
        return "spelling"
    if "known as" in text or "nickname" in text or "called" in text or _COMMENT_AKA.match(text + " x"):
        return "alias"
    if family:
        return "extra_last"
    if given:
        return "extra_first"
    return ""


def _closest_name_slot(value: str, parts: NameParts) -> str:
    """The parsed name a variant is a spelling of, or '' when it resembles none."""
    candidates = (
        ("first", parts.first),
        ("last", parts.last),
        ("indigenous", parts.indigenous),
    )
    best_slot, best_ratio = "", 0.0
    for slot, name in candidates:
        if not name:
            continue
        score = _ratio(name, value)
        if score > best_ratio:
            best_slot, best_ratio = slot, score
    return best_slot if best_ratio >= SLASH_SPELLING_THRESHOLD else ""


def merge_name_comments(parts: NameParts, comments: str) -> NameParts:
    """Fold a row's labelled name comments into the parts parsed from its name cells."""
    if not (comments or "").strip():
        return parts

    collected: dict[str, list[str]] = {
        "middle": list(parts.middle),
        "extra_first": list(parts.extra_first),
        "extra_last": list(parts.extra_last),
        "first_spelling": list(parts.first_spelling),
        "last_spelling": list(parts.last_spelling),
        "first_meaning": list(parts.first_meaning),
        "last_meaning": list(parts.last_meaning),
        "indigenous_spelling": list(parts.indigenous_spelling),
        "indigenous_meaning": list(parts.indigenous_meaning),
        "alias": list(parts.alias),
        "residue": list(parts.residue),
    }
    indigenous = parts.indigenous

    for clause in _COMMENT_SEPARATOR.split(comments.strip()):
        clause = clause.strip()
        if not clause:
            continue
        labelled = _COMMENT_LABEL.match(clause)
        if labelled:
            slot = _comment_slot(labelled.group("label"))
            value = labelled.group("value").strip()
        else:
            bare = _COMMENT_AKA.match(clause)
            slot, value = ("alias", bare.group("value").strip()) if bare else ("", clause)
        value = _COMMENT_LEAD_CUE.sub("", value).strip(_QUOTE_CHARS)
        # The sheet brackets a reading it is unsure of: "[Sen]". The brackets are
        # not part of the name.
        if len(value) >= 2 and value[0] == "[" and value[-1] == "]":
            value = value[1:-1].strip(_QUOTE_CHARS)
        if not value:
            continue
        if not slot:
            # Unlabelled prose is kept, but only where nothing can match a person by it.
            collected["residue"].append(clause)
            continue
        if value.lower() in {existing.lower() for existing in (parts.first, parts.last, indigenous) if existing}:
            continue
        if slot == "indigenous":
            if not indigenous:
                indigenous = value
            elif _ratio(indigenous, value) >= SLASH_SPELLING_THRESHOLD:
                collected["indigenous_spelling"].append(value)
            else:
                collected["alias"].append(value)
            continue
        if slot == "spelling":
            closest = _closest_name_slot(value, NameParts(first=parts.first, last=parts.last, indigenous=indigenous))
            # A "spelling" that resembles no parsed name is a different name, and
            # recording it as one would make the row answer searches for a name it
            # does not hold.
            collected[f"{closest}_spelling" if closest else "alias"].append(value)
            continue
        collected[slot].append(value)

    return NameParts(
        first=parts.first,
        middle=_dedupe(collected["middle"]),
        last=parts.last,
        extra_first=_dedupe(collected["extra_first"]),
        extra_last=_dedupe(collected["extra_last"]),
        first_spelling=_dedupe(collected["first_spelling"]),
        last_spelling=_dedupe(collected["last_spelling"]),
        first_meaning=_dedupe(collected["first_meaning"]),
        last_meaning=_dedupe(collected["last_meaning"]),
        indigenous=indigenous,
        indigenous_spelling=_dedupe(collected["indigenous_spelling"]),
        indigenous_meaning=_dedupe(collected["indigenous_meaning"]),
        alias=_dedupe(collected["alias"]),
        residue=_dedupe(collected["residue"]),
        flags=parts.flags,
    )


def parse_master(
    first: str = "",
    middle: str = "",
    last: str = "",
    indigenous: str = "",
    comments: str = "",
) -> NameParts:
    """Parse the Master list using generic, deterministic normalization rules."""
    return merge_name_comments(_master_reference_parts(first, middle, last, indigenous), comments)


def master_parts_match_clean_source(
    parts: NameParts,
    *,
    first: str = "",
    middle: str = "",
    last: str = "",
    indigenous: str = "",
) -> bool:
    """Whether parsing made no semantic change to an already-clean Master row.

    This is intentionally content-based: no row ids, reference files, expected counts,
    or dataset-specific names are consulted.  A clean row is one whose raw source fields
    already map directly to the canonical slots and produce no aliases, spellings,
    meanings, residue, or flags.
    """
    direct = NameParts(
        first=(first or "").strip(),
        middle=((middle or "").strip(),) if (middle or "").strip() else (),
        last=(last or "").strip(),
        indigenous=(indigenous or "").strip(),
    )
    return parts == direct


def master_name_needs_normalization(
    first: str = "",
    middle: str = "",
    last: str = "",
    indigenous: str = "",
) -> bool:
    """Return True only when generic parsing changes the raw Master name semantics."""
    parts = parse_master(first=first, middle=middle, last=last, indigenous=indigenous)
    return not master_parts_match_clean_source(
        parts, first=first, middle=middle, last=last, indigenous=indigenous
    )
