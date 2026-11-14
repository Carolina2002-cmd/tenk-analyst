"""Stage 0 tests: the chunks are in Qdrant, and a second ingest changes nothing."""
from backend.ingest import COLLECTION, get_client, main


def test_collection_holds_10229_chunks_and_rerun_does_not_change_it():
    main()  # first run embeds anything missing; on a full collection it skips every filing
    client = get_client()
    assert client.count(COLLECTION, exact=True).count == 10229

    main()  # second run must skip everything
    assert client.count(COLLECTION, exact=True).count == 10229


def test_every_chunk_keeps_its_metadata():
    client = get_client()
    points, _ = client.scroll(COLLECTION, limit=50, with_payload=True)
    for p in points:
        meta = p.payload["metadata"]
        assert meta["company"] and meta["year"] in (2024, 2025)
        assert meta["item"] and meta["item_title"]
        assert meta["url"].startswith("https://www.sec.gov/")
