# Roadmap: method extraction and reproducibility audit

The current system focuses on provenance classification: whether a biomedical paper mainly uses author-generated data or reused public datasets.

Next modules should be added in this order:

1. Repository metadata connector for GEO/GSM/SRA/ENA/BioSample.
2. Biological experiment schema extraction.
3. Computational method schema extraction.
4. Source harmonization across article text and repository metadata.
5. Missing-detail and contradiction audit.
6. Reviewer-editable protocol reconstruction.

The first assay-specific target should be narrow, for example ChIP-seq, because the schema can be made concrete:

- organism
- tissue or cell type
- disease condition
- assay target
- antibody information
- biological replicates
- sequencing platform
- reference genome
- alignment tool
- peak calling tool
- normalization method
- processed data availability
- code availability

Each extracted field should store value, evidence, source, status, confidence, and review status.
