# Synthetic Pressure Equipment Standard (SPES-1)

## What is this?

A **synthetic engineering standard** that mirrors the structure and style of the
ASME Boiler & Pressure Vessel Code (BPVC), Section VIII, Division 1, without
reproducing any copyrighted material.

## Why synthetic?

The real ASME BPVC is copyrighted. This project demonstrates techniques over a
representative-but-fictional analog so it can be distributed freely and tested
end-to-end without license concerns.

## Naming conventions

| Concept | Real ASME | Synthetic SPES-1 |
|---|---|---|
| Material spec prefix | `SA-` | `SM-` |
| General requirements | `UG-` | `A-` |
| Welding requirements | `UW-` | `B-` |
| Material properties tables | `Section II Part D` | `Part M` |
| Stress table | `Table 1A` | `Table M-1` |
| Joint efficiency table | `UW-12` | `B-12` |

## Structure

- `standard.md` — The full synthetic standard (sections, formulas, tables, cross-refs)
- `table_1a_materials.json` — Material property table (stress vs. temperature)
- `joint_efficiency_table.json` — Joint efficiency matrix (type × examination)
- `design_examples.json` — Ground truth worked calculations (the eval oracle)

## Relationship to pinned data

`src/anvil/pinned/` contains the same numeric ground truth in Python form. The
generation layer prefers pinned data over RAG retrieval for critical lookups to
eliminate the risk of hallucinated material properties. The JSON files here
exist so the parsing/retrieval layers have a document to ingest.
