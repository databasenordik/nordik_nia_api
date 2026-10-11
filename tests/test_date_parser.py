"""Behaviour of the preprocessed date columns.

The point of these columns is that a date is stored at the precision it was recorded to.
The production SQL parser flattens everything to a DATE, which forces it to invent the
components the source never gave -- "June 1882" becomes 1882-01-01. These lock the
decision not to do that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.normalization.date_parser import parse_date  # noqa: E402


def test_full_date_keeps_all_three_components():
    parts = parse_date("1889-12-24")
    assert (parts.year, parts.month, parts.day) == (1889, 12, 24)
    assert parts.precision == "day"
    assert parts.iso == "1889-12-24"


def test_month_precision_does_not_invent_a_day():
    parts = parse_date("June 1882")
    assert (parts.year, parts.month, parts.day) == (1882, 6, None)
    assert parts.precision == "month"
    assert parts.iso == "1882-06"


def test_year_precision_does_not_invent_a_month_or_day():
    parts = parse_date("1874")
    assert (parts.year, parts.month, parts.day) == (1874, None, None)
    assert parts.precision == "year"
    assert parts.iso == "1874"


def test_zero_components_mean_not_recorded():
    # "1868-00-00" records only a year; the zeros are placeholders, not a January date.
    assert parse_date("1868-00-00").iso == "1868"
    assert parse_date("00.09.1962").iso == "1962-09"


def test_day_first_dotted_format():
    parts = parse_date("23.04.1940")
    assert (parts.year, parts.month, parts.day) == (1940, 4, 23)


def test_month_name_without_separators():
    parts = parse_date("9Aug1877")
    assert (parts.year, parts.month, parts.day) == (1877, 8, 9)


def test_a_second_admission_is_kept_not_discarded():
    parts = parse_date("28 August 1879 2nd time August 1881")
    assert parts.iso == "1879-08-28"
    assert parts.extra == ("1881-08",)
    assert "multiple" in parts.flags


def test_impossible_day_falls_back_to_the_month_it_is_sure_of():
    # September has 30 days. Asserting the 1st, as the SQL parser does, invents a date.
    parts = parse_date("31.09.1944")
    assert (parts.year, parts.month, parts.day) == (1944, 9, None)
    assert parts.precision == "month"
    assert "impossible_day" in parts.flags


def test_provenance_note_is_flagged_and_stripped():
    parts = parse_date("1902-09-14 (NCTR SOURCE)")
    assert parts.iso == "1902-09-14"
    assert "annotated" in parts.flags


def test_approximate_and_uncertain_markers_are_recorded():
    assert "approximate" in parse_date("1868-00-00~").flags
    assert "approximate" in parse_date("1895 est").flags
    assert "uncertain" in parse_date("1960-61 (?)").flags


def test_text_without_a_date_is_reported_not_guessed():
    for value in ("Unknown", "Not Known", "Not listed", "algoma"):
        parts = parse_date(value)
        assert parts.year is None
        assert "unparsed" in parts.flags


def test_hyphenated_day_month_year_keeps_all_parts():
    parts = parse_date("26-Apr-1946")
    assert (parts.year, parts.month, parts.day) == (1946, 4, 26)
    assert parts.iso == "1946-04-26"
    assert parts.precision == "day"


def test_unambiguous_mdy_slash_is_not_collapsed_to_a_year():
    parts = parse_date("12/14/1890")
    assert (parts.year, parts.month, parts.day) == (1890, 12, 14)
    assert parts.iso == "1890-12-14"
    parts = parse_date("08/18/1889")
    assert (parts.year, parts.month, parts.day) == (1889, 8, 18)


def test_ambiguous_slash_date_is_flagged_not_silently_flipped():
    parts = parse_date("10/9/1951")
    assert parts.iso == "1951-09-10"
    assert "ambiguous_order" in parts.flags


def test_space_separated_ymd_keeps_month_and_day():
    parts = parse_date("1943 09 14  (NCTR SOURCE)")
    assert parts.iso == "1943-09-14"
    assert "annotated" in parts.flags


def test_year_range_keeps_both_ends():
    parts = parse_date("1960-61 (?)")
    assert parts.iso == "1960"
    assert "1961" in parts.extra
    assert "range" in parts.flags
    assert "uncertain" in parts.flags


def test_year_month_is_not_a_range():
    parts = parse_date("1943-11-~")
    assert parts.iso == "1943-11"
    assert "range" not in parts.flags
    assert "approximate" in parts.flags


def test_post_and_ca_are_qualifiers():
    assert "after" in parse_date("1918-09-17 (post)").flags
    assert "after" in parse_date("Post 1911").flags
    assert parse_date("Post 1911").iso == "1911"
    assert "approximate" in parse_date("ca. 1929").flags
    assert parse_date("ca. 1929").iso == "1929"


def test_alternative_months_are_both_kept():
    parts = parse_date("1894- May or July")
    assert parts.iso == "1894"
    assert "1894-05" in parts.extra
    assert "1894-07" in parts.extra
    assert "alternative" in parts.flags


def test_month_only_without_year_is_kept():
    parts = parse_date("Dec")
    assert parts.year is None
    assert parts.month == 12
    assert parts.iso == "--12"
    assert parts.precision == "month"


def test_month_day_without_year_is_kept():
    parts = parse_date("Sent to hospital Nov. 8 (CIRNAC SOURCE)")
    assert parts.year is None
    assert (parts.month, parts.day) == (11, 8)
    assert parts.iso == "--11-08"
    assert "annotated" in parts.flags


def test_excel_serial_converts_only_non_year_integers():
    parts = parse_date("11314")
    assert parts.iso == "1930-12-22"
    assert "excel_serial" in parts.flags
    year_only = parse_date("1908")
    assert year_only.iso == "1908"
    assert "excel_serial" not in year_only.flags


def test_empty_is_safe():
    parts = parse_date("")
    assert parts.year is None and parts.iso == "" and parts.flags == ()


def test_sort_key_orders_unknown_components_first():
    assert parse_date("1882").sort_key == "1882-00-00"
    assert parse_date("1882-06").sort_key == "1882-06-00"
    assert parse_date("1882-06-05").sort_key == "1882-06-05"
    assert parse_date("1882").sort_key < parse_date("1882-06-05").sort_key


def test_backfill_writes_only_recorded_components():
    from app.normalization.backfill_names import date_block

    row = {"row_data_normalized": {"fields": {"Admitted": {"raw": "June 1882"}}}}
    block = date_block(row, ["admitted_date"], {"admitted_date": ["fields.Admitted"]})
    entry = block["admitted_date"]
    assert entry == {"year": 1882, "month": 6, "iso": "1882-06", "precision": "month"}
    assert "day" not in entry
