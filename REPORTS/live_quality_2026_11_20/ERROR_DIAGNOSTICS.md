# Cham OCR Live: Detailed Failure Audit

All results newly sampled at production URL; see REPORT.md for methods.

## Categorized character errors

### v24_1_r4

Total edits: 456 (substitutions 128 / deletions 245 / insertions 83)

By source-character category: Cham vowel / diacritic: 139, Spacing / linebreak: 69, Cham coda: 61, Cham base consonant / vowel: 55, ASCII punctuation: 55, Cham punctuation: 41, Other: 19, Cham digit: 12, Cham medial: 5

Top confusions:

| Operation | GT | OCR | Count |
|---|---|---|---:|
| I | ∅ | SPACE | 36 |
| D | ꩝ | ∅ | 31 |
| D | ꨮ | ∅ | 30 |
| D | SPACE | ∅ | 27 |
| D | ꩌ | ∅ | 18 |
| D | ꨰ | ∅ | 15 |
| D | ? | ∅ | 15 |
| D | ꩃ | ∅ | 12 |
| S | : | / | 10 |
| S | ꨩ | ꩃ | 9 |
| D | – | ∅ | 9 |
| I | ∅ | ꩌ | 9 |
| D | ꨯ | ∅ | 9 |
| S | ꨮ | ꨩ | 8 |
| S | ꨁ | ꨄ | 8 |

Worst three documents:

- synthetic_142 (tier 4), CER 25.75%, edits 94/365, lines 7/8
  GT first line: ['ꨗꨆꨓꨯꨮꨩ ꨚꨣꨤꨯꨩ ꨣꨪꩌ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ (ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꩑꩐ ꨧꨆꨥꨪ ꨀꨳꨩ']
  OCR first line: ['ꨗꨆꨓꨯꨩ ꨚꨣꨤꨲꨩ ꨣꨪꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꨅꨭ ꨧꨆꨥꨪ ꨀꨳꨩ']
- synthetic_147 (tier 4), CER 12.31%, edits 8/65, lines 2/2
  GT first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨁꩆ']
  OCR first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨄ꩓ꩆ']
- synthetic_196 (tier 5), CER 12.27%, edits 20/163, lines 3/3
  GT first line: ['– ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ: ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ꩝ ꨚꨓꨆꨪꩀ ꨀꨊꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ꩞']
  OCR first line: ['ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ, ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ ꨚꨓꨆꨪꩀ ꨀꨋꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ ꩞']

### v24_1_r3_1

Total edits: 443 (substitutions 147 / deletions 195 / insertions 101)

By source-character category: Cham vowel / diacritic: 125, Spacing / linebreak: 74, Cham coda: 65, ASCII punctuation: 64, Cham base consonant / vowel: 57, Cham punctuation: 26, Other: 19, Cham digit: 7, Cham medial: 6

Top confusions:

| Operation | GT | OCR | Count |
|---|---|---|---:|
| I | ∅ | SPACE | 50 |
| D | ꨮ | ∅ | 29 |
| D | SPACE | ∅ | 21 |
| D | ꩝ | ∅ | 20 |
| S | – | - | 14 |
| D | ? | ∅ | 14 |
| D | ꩌ | ∅ | 12 |
| S | ꨩ | ꨱ | 12 |
| I | ∅ | ꨯ | 12 |
| S | ꨩ | ꩃ | 11 |
| D | ꩃ | ∅ | 11 |
| S | ꨁ | ꨄ | 9 |
| S | ꨮ | ꨩ | 8 |
| S | ꨮ | ꨱ | 7 |
| D | ; | ∅ | 6 |

Worst three documents:

- synthetic_142 (tier 4), CER 25.21%, edits 92/365, lines 7/8
  GT first line: ['ꨗꨆꨓꨯꨮꨩ ꨚꨣꨤꨯꨩ ꨣꨪꩌ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ (ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꩑꩐ ꨧꨆꨥꨪ ꨀꨳꨩ']
  OCR first line: ['ꨗꨆꨓꨯꨩ ꨚꨣꨤꨯꨩ ꨣꨪꨔꨭꩆ ꨓꨠꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꨅꨭ꩐ ꨧꨆꨥꨪ ꨀꨳꨩ']
- synthetic_196 (tier 5), CER 12.88%, edits 21/163, lines 3/3
  GT first line: ['– ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ: ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ꩝ ꨚꨓꨆꨪꩀ ꨀꨊꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ꩞']
  OCR first line: ['ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ* ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ] ꨚꨓꨆꨪꩀ ꨀꨋꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬꩌ ꨧꨩ ꨮꩃ ꩞']
- synthetic_147 (tier 4), CER 10.77%, edits 7/65, lines 2/2
  GT first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨁꩆ']
  OCR first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨄ꩓ꩆ']

### v24_1_r3

Total edits: 489 (substitutions 159 / deletions 185 / insertions 145)

By source-character category: Cham vowel / diacritic: 152, Spacing / linebreak: 87, Cham coda: 65, ASCII punctuation: 64, Cham base consonant / vowel: 54, Cham punctuation: 32, Other: 19, Cham digit: 8, Cham medial: 8

Top confusions:

| Operation | GT | OCR | Count |
|---|---|---|---:|
| I | ∅ | SPACE | 66 |
| D | ꨮ | ∅ | 30 |
| I | ∅ | ꨯ | 20 |
| D | SPACE | ∅ | 19 |
| D | ꩝ | ∅ | 18 |
| S | ꨩ | ꨱ | 18 |
| D | ? | ∅ | 15 |
| D | ꩌ | ∅ | 15 |
| D | ꩃ | ∅ | 12 |
| I | ∅ | ꨮ | 11 |
| S | – | - | 10 |
| S | ꨩ | ꩃ | 9 |
| S | : | / | 8 |
| S | ꨮ | ꨱ | 8 |
| I | ∅ | ꨰ | 7 |

Worst three documents:

- synthetic_142 (tier 4), CER 25.75%, edits 94/365, lines 7/8
  GT first line: ['ꨗꨆꨓꨯꨮꨩ ꨚꨣꨤꨯꨩ ꨣꨪꩌ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ (ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꩑꩐ ꨧꨆꨥꨪ ꨀꨳꨩ']
  OCR first line: ['ꨗꨆꨓꨯꨩ ꨚꨣꨤꨩ ꨣꨪ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꨪꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꨭ꩐ ꨧꨆꨶꨪ ꨀꨳꨩ']
- synthetic_196 (tier 5), CER 14.72%, edits 24/163, lines 3/3
  GT first line: ['– ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ: ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ꩝ ꨚꨓꨆꨪꩀ ꨀꨊꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ꩞']
  OCR first line: ['ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ* ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꨰꩀ ꨚꨓꨆꨪꩀ ꨀꨋꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ ꩞']
- synthetic_134 (tier 4), CER 10.37%, edits 14/135, lines 3/3
  GT first line: ['ꨨꨕꨯꩌ ꨝꨣꨪꩍ ꨆꨣꨬ ꨌꨪꩍ ꨙꩌ ꨀꨕꩅ']
  OCR first line: ['ꨨꨕꨯꩌ ꨝꨣꨪꩍ ꨆꨣꨬ ꨌꨪꩍ ꨙꩌ ꨀꨕꩅ']

### v24_1_r2_1_6

Total edits: 475 (substitutions 183 / deletions 179 / insertions 113)

By source-character category: Cham vowel / diacritic: 146, Spacing / linebreak: 68, Cham base consonant / vowel: 67, ASCII punctuation: 65, Cham coda: 61, Cham punctuation: 26, Other: 19, Cham digit: 12, Cham medial: 11

Top confusions:

| Operation | GT | OCR | Count |
|---|---|---|---:|
| I | ∅ | SPACE | 44 |
| D | ꨮ | ∅ | 28 |
| D | SPACE | ∅ | 21 |
| S | ꨩ | ꩃ | 15 |
| S | ꨮ | ꨱ | 14 |
| D | ? | ∅ | 14 |
| S | ꨩ | ꨱ | 13 |
| D | ꩌ | ∅ | 13 |
| I | ∅ | ꨯ | 13 |
| S | – | - | 12 |
| D | ꩝ | ∅ | 11 |
| I | ∅ | ꨮ | 10 |
| D | ꩃ | ∅ | 9 |
| I | ∅ | ꨰ | 8 |
| S | : | / | 7 |

Worst three documents:

- synthetic_142 (tier 4), CER 27.12%, edits 99/365, lines 7/8
  GT first line: ['ꨗꨆꨓꨯꨮꨩ ꨚꨣꨤꨯꨩ ꨣꨪꩌ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨥꨪ ꨌꩌ (ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꩑꩐ ꨧꨆꨥꨪ ꨀꨳꨩ']
  OCR first line: ['ꨗꨆꨓꨯꨩ ꨚꨣꨤꨲꨩ ꨣꨪ ꨔꨭꩆ ꨓꨟꨩ ꨝꨪꨤꩆ ꨓꨎꨭꩍ ꨧꨆꨶꨪ ꨌꩌ ꨗꨶꨮꩄ ꨝꨪꨤꩆ ꨭꩆ ꨧꨆꨶꨪ ꨀꨳꨩ']
- synthetic_147 (tier 4), CER 13.85%, edits 9/65, lines 2/2
  GT first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨁꩆ']
  OCR first line: ['꩓꩞꩓꩞ ꨞꨩ ꨙꩌ ꨟꨄ꩓ꩆ']
- synthetic_196 (tier 5), CER 13.50%, edits 22/163, lines 3/3
  GT first line: ['– ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ: ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꩀ꩝ ꨚꨓꨆꨪꩀ ꨀꨊꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨡꨮꩃ꩞']
  OCR first line: ['ꨇꨪꩀ ꨠꩊ ꨌꨪꩍ/ ꨅꨩ ꨆꨵꩀ ꨀꨟꨣꨰꩀ] ꨚꨓꨆꨪꩀ ꨀꨋꨶꨬ ꨆꨢꨩ ꨆꨓꩀ ꨀꨊꨶꨬ ꨧꨩ ꨕꨮꩃ꩞']

## Paired comparison on identical inputs

| Newer model | Comparator | Same images | Newer wins | Ties | Newer loses | Mean CER gain (points) |
|---|---|---:|---:|---:|---:|---:|
| v24_1_r4 | v24_1_r3_1 | 29 | 8 | 3 | 18 | -0.61 |
| v24_1_r4 | v24_1_r3 | 29 | 11 | 6 | 12 | 0.02 |
| v24_1_r4 | v24_1_r2_1_6 | 29 | 13 | 4 | 12 | -0.09 |
