# Quality Report: beam_search_small

**Submission:** `submission_beam_small.csv`
**Total predictions:** 4253

## Global Metrics

| Metric | Count | % |
|--------|-------|---|
| Empty predictions | 0 | 0.0% |
| Single character | 19 | 0.4% |
| Punctuation only | 28 | 0.7% |
| Garbled script | 359 | 8.4% |
| Repetitive | 1 | 0.0% |
| Duplicate transcriptions | 79 | 1.9% |
| Unique transcriptions | 4174 | 98.1% |

**Average transcript length:** 142 chars

## Per-Language Breakdown

### LIN (1866 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 6 (0.3%) |
| Repetitive | 1 (0.1%) |
| Duplicates | 27 |
| Avg pred length | 144 chars |
| Avg GT length | 150 chars |
| Length ratio | 0.96x |

**Sample predictions:**

- `lin_24758`: Dosa qu'on mounan dha qu'e tagia di nivo, pico loba qu'o mico bonas e dosa qu mona matiti n'avanzete. Ezanquillera pemme
- `lin_32255`: Oto e l'axibis o moa sal e moa komun ben e pilsa, eza na va kiti moa kiti togo. On diria eza l'akati avyo, ben eza le oy
- `lin_69423`: nà moun you you zaba biki yandengi nandengie yako koma.

### LUG (638 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 0 (0.0%) |
| Repetitive | 0 (0.0%) |
| Duplicates | 2 |
| Avg pred length | 173 chars |
| Avg GT length | 208 chars |
| Length ratio | 0.83x |

**Sample predictions:**

- `lug_96114`: Omu cha aromu du gavu e yehsivi e chitambara kumudwegwe. Awe so umuana. Umuana hiye vasse e mawe gawa wali yo ebi mera e
- `lug_75987`: kwa? kwa kwa. kwa Kwa.
- `lug_75903`: Ndaba etinyoni. Etinyoni kino tina umumu hafu. Tina kala ezenja ulu wiyawia wiyacho. Onimu kala enziru gafu. Eya tukusik

### SNA (1749 samples)

| Metric | Value |
|--------|-------|
| Empty | 0 (0.0%) |
| Garbled | 353 (20.2%) |
| Repetitive | 0 (0.0%) |
| Duplicates | 49 |
| Avg pred length | 128 chars |
| Avg GT length | 192 chars |
| Length ratio | 0.67x |

**Sample predictions:**

- `sna_60933`: ὀ ὁ ὃ ὅ ὇ ὄ Ὁ ὒ Ὅ Ὀ ὂ ὘ ὗ ὡ ὔ ὕ ὤ ὦ ὜ ὣ Ὓ ὎ ὓ ὑ ὖ Ὦ ὢ ὠ Ὗ ὏ Ὂ Ὕ ὐ Ὑ Ὃ Ὄ ὚ ὧ Ὣ ὥ ὆ Ὠ Ὡ ὶ ὲ ὸ Ὧ ὰ ά ί Ὤ ὼ ώ Ὢ ὾ ύ ό ὴ έ Ὥ 
- `sna_60986`: ḍᵉᶦᵃ ᵃᵗᵒᶜᶠᶤᵍᶔᶉ ᶀᶼᶰᶶᶯᶸᶱᶕᶗᶒ
- `sna_60960`: Mūrumi āre mūmu kwa gwa un e ue tāra un oratitza kūti ārekū fanbisa ākāp fe kā muttepe mutteima āka bere kā beke dema ā 

## Submission Validation

**PASS:** All 4253 IDs present
