from app.services.text_cleaning import clean_text_segments


def test_clean_text_segments_normalizes_spaces() -> None:
    full_text, segments = clean_text_segments(["Hello   world\nthis is\na test"])

    assert "Hello world" in full_text
    assert segments
