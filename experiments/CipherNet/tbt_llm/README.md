# TBT-LLM

Thousand Brains Theory applied to sequence modeling and language.

This is a self-contained experiment. It imports shared utilities from
`src/` (columns, reference frames, modalities) but does NOT modify them.
The parent CipherNet vision system is the fallback baseline.

## Status: DESIGN PHASE

See `../SEQUENCE_MODELING.md` for the full planning document.

## Intended structure (to be created as phases are implemented)

    tbt_llm/
        data/           tokenised corpora, cached SDR encodings
        src/
            temporal_memory.py   cells-within-columns, distal dendrites, bursting
            token_modality.py    SensorModality for discrete tokens -> SDR
            temporal_frame.py    multi-scale temporal reference frame
            sequence_cortex.py   Cortex subclass wired for sequences
        experiments/
            s1_counting.py       Phase S1: next-symbol on deterministic sequences
            s2_wikitext.py       Phase S3: character prediction on WikiText-2
        tests/
