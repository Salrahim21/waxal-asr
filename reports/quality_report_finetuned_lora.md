# Quality Report: finetuned_lora

**Submission:** `submission_finetuned.csv`
**Total predictions:** 4253

## Global Metrics

| Metric | Count | % |
|--------|-------|---|
| Empty predictions | 0 | 0.0% |
| Single character | 13 | 0.3% |
| Punctuation only | 13 | 0.3% |
| Garbled script | 0 | 0.0% |
| Repetitive | 0 | 0.0% |
| Duplicate transcriptions | 78 | 1.8% |
| Unique transcriptions | 4175 | 98.2% |

**Average transcript length:** 166 chars

## Per-Language Breakdown

### LIN (1866 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 0 (0.0%) |
| Repetitive | 0 (0.0%) |
| Duplicates | 75 |
| Avg pred length | 145 chars |
| Avg GT length | 150 chars |
| Length ratio | 0.96x |

**Sample predictions:**

- `lin_24758`: Toza komona ndako etaji ya dini vo. Ikolo bako mikombo na se toza omona matiti na vanzete. Eza na kulera pembe.
- `lin_32255`: Boto elakisi biso mwa sale mokombo mene penza eza na ba kiti moko ya kitogo. Mdiri eza nakati ya vio pe sale oyo emunami
- `lin_69423`: E namoni oyo eza ba biki ya ndenge na ndenge ya kokoma.

### LUG (638 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 0 (0.0%) |
| Repetitive | 0 (0.0%) |
| Duplicates | 0 |
| Avg pred length | 179 chars |
| Avg GT length | 208 chars |
| Length ratio | 0.86x |

**Sample predictions:**

- `lug_96114`: Omukyalo omudugavu yeesivye ekitambala kumutwegwe aweso omwana,omwana yeevasi emabega waliyo ebimera ebya kiragala.
- `lug_75987`: Mechifana nchirabikira mungu omutemi wenyama. Yali ate mateme enyama yaonga bwa pima wali wono omulala yali alabika nti 
- `lug_75903`: Ndaba etinyoni. Echinyoni kino china omumomu wavu. China kkala ez'enjawulo mubyoya byacho. Mune mukkala enzirugavu eya c

### SNA (1749 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 0 (0.0%) |
| Repetitive | 0 (0.0%) |
| Duplicates | 1 |
| Avg pred length | 184 chars |
| Avg GT length | 192 chars |
| Length ratio | 0.96x |

**Sample predictions:**

- `sna_19435`: Nzvimbo yezvitoro iri kutanga kubudirira. Chitoro chekutanga achisati chanyazo pera chine zvitina zvitsvuku. Chepakati c
- `sna_60933`: Pane imwe nzvimbo. Pane vakadzi vakagara musvigaro. Vachitarisa mumwe mudzimai ari mberipawo anuita seari kutamba. Vakad
- `sna_60986`: Mudzimai amire kumberi kwemango mazvimai akapfeka zvichena panoratidza kuti pamuchato pane mabharuma nezvinji.

## Submission Validation

**PASS:** All 4253 IDs present
