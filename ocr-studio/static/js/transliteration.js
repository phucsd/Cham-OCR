/**
 * Cham Paleographic Unicode Inspector & Scholarly Transliteration Engine
 * Covers Unicode Cham Block (U+AA00..U+AA5F) in Canonical Brahmic Logical Order.
 */
(function (global) {
    "use strict";

    const CHAM_UNICODE_TABLE = {
        // Independent Vowels (U+AA00..U+AA05)
        0xAA00: { name: "CHAM LETTER A", cat: "Independent Vowel", rom: "a", skt: "a" },
        0xAA01: { name: "CHAM LETTER I", cat: "Independent Vowel", rom: "i", skt: "i" },
        0xAA02: { name: "CHAM LETTER U", cat: "Independent Vowel", rom: "u", skt: "u" },
        0xAA03: { name: "CHAM LETTER E", cat: "Independent Vowel", rom: "é", skt: "e" },
        0xAA04: { name: "CHAM LETTER AI", cat: "Independent Vowel", rom: "ai", skt: "ai" },
        0xAA05: { name: "CHAM LETTER O", cat: "Independent Vowel", rom: "o", skt: "o" },

        // Consonants (U+AA06..U+AA28)
        0xAA06: { name: "CHAM LETTER KA", cat: "Base Consonant", base: "k", inherent: "a", skt: "ka" },
        0xAA07: { name: "CHAM LETTER KHA", cat: "Base Consonant", base: "kh", inherent: "a", skt: "kha" },
        0xAA08: { name: "CHAM LETTER GA", cat: "Base Consonant", base: "g", inherent: "a", skt: "ga", pair: "ꨓ (U+AA13 TA) / ꨤ (U+AA24 LA)" },
        0xAA09: { name: "CHAM LETTER GHA", cat: "Base Consonant", base: "gh", inherent: "a", skt: "gha" },
        0xAA0A: { name: "CHAM LETTER NGUE", cat: "Base Consonant", base: "ng", inherent: "â", skt: "ṅa", pair: "ꨕ (U+AA15 DA)" },
        0xAA0B: { name: "CHAM LETTER NGA", cat: "Base Consonant", base: "ng", inherent: "a", skt: "ṅga" },
        0xAA0C: { name: "CHAM LETTER CHA", cat: "Base Consonant", base: "c", inherent: "a", skt: "ca" },
        0xAA0D: { name: "CHAM LETTER CHHA", cat: "Base Consonant", base: "ch", inherent: "a", skt: "cha" },
        0xAA0E: { name: "CHAM LETTER JA", cat: "Base Consonant", base: "j", inherent: "a", skt: "ja" },
        0xAA0F: { name: "CHAM LETTER JHA", cat: "Base Consonant", base: "jh", inherent: "a", skt: "jha" },
        0xAA10: { name: "CHAM LETTER NHUE", cat: "Base Consonant", base: "ny", inherent: "â", skt: "ña" },
        0xAA11: { name: "CHAM LETTER NHA", cat: "Base Consonant", base: "ny", inherent: "a", skt: "ñja", pair: "ꨒ (U+AA12 NHJA)" },
        0xAA12: { name: "CHAM LETTER NHJA", cat: "Base Consonant", base: "nj", inherent: "a", skt: "ṇḍa", pair: "ꨑ (U+AA11 NHA)" },
        0xAA13: { name: "CHAM LETTER TA", cat: "Base Consonant", base: "t", inherent: "a", skt: "ta", pair: "ꨈ (U+AA08 GA)" },
        0xAA14: { name: "CHAM LETTER THA", cat: "Base Consonant", base: "th", inherent: "a", skt: "tha", pair: "ꨟ (U+AA1F MA)" },
        0xAA15: { name: "CHAM LETTER DA", cat: "Base Consonant", base: "d", inherent: "a", skt: "da", pair: "ꨊ (U+AA0A NGUE)" },
        0xAA16: { name: "CHAM LETTER DHA", cat: "Base Consonant", base: "dh", inherent: "a", skt: "dha" },
        0xAA17: { name: "CHAM LETTER NUE", cat: "Base Consonant", base: "n", inherent: "â", skt: "na" },
        0xAA18: { name: "CHAM LETTER NA", cat: "Base Consonant", base: "n", inherent: "a", skt: "ṇa" },
        0xAA19: { name: "CHAM LETTER DDA", cat: "Base Consonant", base: "nd", inherent: "a", skt: "nda" },
        0xAA1A: { name: "CHAM LETTER PA", cat: "Base Consonant", base: "p", inherent: "a", skt: "pa" },
        0xAA1B: { name: "CHAM LETTER PPA", cat: "Base Consonant", base: "p", inherent: "a", skt: "ppa" },
        0xAA1C: { name: "CHAM LETTER PHA", cat: "Base Consonant", base: "ph", inherent: "a", skt: "pha" },
        0xAA1D: { name: "CHAM LETTER BA", cat: "Base Consonant", base: "b", inherent: "a", skt: "ba", pair: "ꨟ (U+AA1F MA)" },
        0xAA1E: { name: "CHAM LETTER BHA", cat: "Base Consonant", base: "bh", inherent: "a", skt: "bha" },
        0xAA1F: { name: "CHAM LETTER MUE", cat: "Base Consonant", base: "m", inherent: "â", skt: "ma", pair: "ꨔ (U+AA14 THA) / ꨝ (U+AA1D BA)" },
        0xAA20: { name: "CHAM LETTER MA", cat: "Base Consonant", base: "m", inherent: "a", skt: "mma" },
        0xAA21: { name: "CHAM LETTER BBA", cat: "Base Consonant", base: "mb", inherent: "a", skt: "mba", pair: "ꨥ (U+AA25 VA)" },
        0xAA22: { name: "CHAM LETTER YA", cat: "Base Consonant", base: "y", inherent: "a", skt: "ya" },
        0xAA23: { name: "CHAM LETTER RA", cat: "Base Consonant", base: "r", inherent: "a", skt: "ra" },
        0xAA24: { name: "CHAM LETTER LA", cat: "Base Consonant", base: "l", inherent: "a", skt: "la", pair: "ꨈ (U+AA08 GA)" },
        0xAA25: { name: "CHAM LETTER VA", cat: "Base / Final Consonant", base: "w", inherent: "a", finalRom: "w", skt: "va", pair: "ꨡ (U+AA21 BBA)" },
        0xAA26: { name: "CHAM LETTER SSA", cat: "Base Consonant", base: "s", inherent: "a", skt: "śa" },
        0xAA27: { name: "CHAM LETTER SA", cat: "Base Consonant", base: "s", inherent: "a", skt: "sa" },
        0xAA28: { name: "CHAM LETTER HA", cat: "Base Consonant", base: "h", inherent: "a", skt: "ha" },

        // Dependent Vowel Signs (U+AA29..U+AA32)
        0xAA29: { name: "CHAM VOWEL SIGN AA", cat: "Vowel Lengthener", vowel: "ā", skt: "ā", order: 5 },
        0xAA2A: { name: "CHAM VOWEL SIGN I", cat: "Dependent Vowel (Above)", vowel: "i", skt: "i", order: 4, pair: "ꨫ (U+AA2B II)" },
        0xAA2B: { name: "CHAM VOWEL SIGN II", cat: "Dependent Vowel (Above)", vowel: "ī", skt: "ī", order: 4, pair: "ꨪ (U+AA2A I)" },
        0xAA2C: { name: "CHAM VOWEL SIGN EI", cat: "Dependent Vowel (Above)", vowel: "ei", skt: "ei", order: 4 },
        0xAA2D: { name: "CHAM VOWEL SIGN U", cat: "Dependent Vowel (Below)", vowel: "u", skt: "u", order: 4 },
        0xAA2E: { name: "CHAM VOWEL SIGN OE", cat: "Dependent Vowel (Post)", vowel: "ơ", skt: "ē", order: 4 },
        0xAA2F: { name: "CHAM VOWEL SIGN O", cat: "Pre-Base Vowel (Left)", vowel: "o", skt: "o", order: 3 },
        0xAA30: { name: "CHAM VOWEL SIGN AI", cat: "Pre-Base Vowel (Left)", vowel: "ai", skt: "ai", order: 3 },
        0xAA31: { name: "CHAM VOWEL SIGN AU", cat: "Dependent Vowel (Post)", vowel: "au", skt: "ao", order: 4 },
        0xAA32: { name: "CHAM VOWEL SIGN UE", cat: "Dependent Vowel (Below)", vowel: "ư", skt: "au", order: 4, pair: "ꨶ (U+AA36 Medial WA)" },

        // Consonant Signs / Medials (U+AA33..U+AA36)
        0xAA33: { name: "CHAM CONSONANT SIGN YA", cat: "Subjoined Medial", medial: "i", skt: "y", order: 2 },
        0xAA34: { name: "CHAM CONSONANT SIGN RA", cat: "Pre/Subjoined Medial", medial: "r", skt: "r", order: 1 },
        0xAA35: { name: "CHAM CONSONANT SIGN LA", cat: "Subjoined Medial", medial: "l", skt: "l", order: 1 },
        0xAA36: { name: "CHAM CONSONANT SIGN WA", cat: "Subjoined Medial", medial: "u", skt: "v", order: 2, pair: "ꨲ (U+AA32 Vowel UE)" },

        // Final Consonants & Signs (U+AA40..U+AA4D)
        0xAA40: { name: "CHAM LETTER FINAL K", cat: "Final Consonant", coda: "k", order: 6 },
        0xAA41: { name: "CHAM LETTER FINAL G", cat: "Final Consonant", coda: "g", order: 6 },
        0xAA42: { name: "CHAM LETTER FINAL NG", cat: "Final Consonant", coda: "ng", order: 6, pair: "ꩍ (U+AA4D Final H)" },
        0xAA43: { name: "CHAM CONSONANT SIGN FINAL NG", cat: "Combining Coda (Above)", coda: "ng", order: 6, pair: "꩝ (U+AA5D Danda) / ꩌ (U+AA4C Final M)" },
        0xAA44: { name: "CHAM LETTER FINAL CH", cat: "Final Consonant", coda: "c", order: 6 },
        0xAA45: { name: "CHAM LETTER FINAL T", cat: "Final Consonant", coda: "t", order: 6, pair: "ꩊ (U+AA4A Final L)" },
        0xAA46: { name: "CHAM LETTER FINAL N", cat: "Final Consonant", coda: "n", order: 6 },
        0xAA47: { name: "CHAM LETTER FINAL P", cat: "Final Consonant", coda: "p", order: 6 },
        0xAA48: { name: "CHAM LETTER FINAL Y", cat: "Final Consonant", coda: "y", order: 6 },
        0xAA49: { name: "CHAM LETTER FINAL R", cat: "Final Consonant", coda: "r", order: 6 },
        0xAA4A: { name: "CHAM LETTER FINAL L", cat: "Final Consonant", coda: "l", order: 6, pair: "ꩅ (U+AA45 Final T)" },
        0xAA4B: { name: "CHAM LETTER FINAL SS", cat: "Final Consonant", coda: "s", order: 6 },
        0xAA4C: { name: "CHAM CONSONANT SIGN FINAL M", cat: "Combining Coda (Anusvara)", coda: "m", order: 6, pair: "ꩃ (U+AA43 Final NG)" },
        0xAA4D: { name: "CHAM CONSONANT SIGN FINAL H", cat: "Combining Coda (Visarga)", coda: "h", order: 6, pair: "ꩂ (U+AA42 Final NG)" },

        // Digits (U+AA50..U+AA59)
        0xAA50: { name: "CHAM DIGIT ZERO", cat: "Cham Numeral", rom: "0" },
        0xAA51: { name: "CHAM DIGIT ONE", cat: "Cham Numeral", rom: "1" },
        0xAA52: { name: "CHAM DIGIT TWO", cat: "Cham Numeral", rom: "2" },
        0xAA53: { name: "CHAM DIGIT THREE", cat: "Cham Numeral", rom: "3" },
        0xAA54: { name: "CHAM DIGIT FOUR", cat: "Cham Numeral", rom: "4" },
        0xAA55: { name: "CHAM DIGIT FIVE", cat: "Cham Numeral", rom: "5" },
        0xAA56: { name: "CHAM DIGIT SIX", cat: "Cham Numeral", rom: "6" },
        0xAA57: { name: "CHAM DIGIT SEVEN", cat: "Cham Numeral", rom: "7" },
        0xAA58: { name: "CHAM DIGIT EIGHT", cat: "Cham Numeral", rom: "8" },
        0xAA59: { name: "CHAM DIGIT NINE", cat: "Cham Numeral", rom: "9" },

        // Punctuation (U+AA5C..U+AA5F)
        0xAA5C: { name: "CHAM PUNCTUATION SPIRAL", cat: "Section / Invocation", rom: "§" },
        0xAA5D: { name: "CHAM PUNCTUATION DANDA", cat: "Clause Punctuation", rom: " | " },
        0xAA5E: { name: "CHAM PUNCTUATION DOUBLE DANDA", cat: "Stanza / Section Mark", rom: " || " },
        0xAA5F: { name: "CHAM PUNCTUATION TRIPLE DANDA", cat: "Major Closure Mark", rom: " ||| " }
    };

    function formatCodepoint(cp) {
        return "U+" + cp.toString(16).toUpperCase().padStart(4, "0");
    }

    function isChamCodepoint(cp) {
        return cp >= 0xAA00 && cp <= 0xAA5F;
    }

    function isCombiningCham(cp) {
        return (cp >= 0xAA29 && cp <= 0xAA36) || cp === 0xAA43 || cp === 0xAA4C || cp === 0xAA4D;
    }

    /**
     * Inspects a grapheme cluster or substring and returns detailed paleographic metadata.
     */
    function inspectCluster(clusterStr) {
        if (!clusterStr) return { cluster: "", codepoints: [], isCanonicalOrder: true, romanized: "", warnings: [] };
        const chars = Array.from(clusterStr);
        const codepoints = chars.map((ch) => {
            const cp = ch.codePointAt(0);
            const meta = CHAM_UNICODE_TABLE[cp] || null;
            return {
                char: ch,
                display: isCombiningCham(cp) ? ("◌" + ch) : ch,
                hex: formatCodepoint(cp),
                cp: cp,
                name: meta ? meta.name : (cp === 0x20 ? "SPACE" : `UNICODE CHARACTER (${formatCodepoint(cp)})`),
                category: meta ? meta.cat : (cp < 128 ? "ASCII / Punctuation" : "Extended Script"),
                minimalPair: meta && meta.pair ? meta.pair : null,
                orderSlot: meta && typeof meta.order === "number" ? meta.order : 0
            };
        });

        // Verify canonical Brahmic ordering among combining marks attached to a base
        let isCanonical = true;
        const warnings = [];
        let maxSlotSeen = 0;
        for (let i = 0; i < codepoints.length; i++) {
            const item = codepoints[i];
            if (i === 0 && isCombiningCham(item.cp)) {
                isCanonical = false;
                warnings.push(`Detached combining mark ${item.hex} (${item.name}) without preceding base consonant.`);
            }
            if (item.orderSlot > 0) {
                if (item.orderSlot < maxSlotSeen) {
                    isCanonical = false;
                    warnings.push(`Non-canonical mark sequence at ${item.hex}: expected Base → Medials → Pre-Vowels → Vowels → Finals.`);
                }
                maxSlotSeen = Math.max(maxSlotSeen, item.orderSlot);
            }
        }

        return {
            cluster: clusterStr,
            codepoints: codepoints,
            isCanonicalOrder: isCanonical,
            romanized: transliterateCham(clusterStr),
            warnings: warnings
        };
    }

    /**
     * Converts Unicode Eastern Cham (Akhar Thrah) text into readable scholarly EFEO/Inrasara-style Romanization.
     */
    function transliterateCham(text) {
        if (!text) return "";
        const cps = Array.from(text).map((c) => c.codePointAt(0));
        let out = "";
        let i = 0;

        while (i < cps.length) {
            const cp = cps[i];
            const info = CHAM_UNICODE_TABLE[cp];

            if (!info) {
                out += String.fromCodePoint(cp);
                i++;
                continue;
            }

            // Standalone vowel, digit, or punctuation
            if (info.rom !== undefined && !info.base) {
                // Attach combining coda if attached directly to an independent vowel (e.g. ꨀꩌ -> am)
                let seg = info.rom;
                i++;
                while (i < cps.length) {
                    const nextInfo = CHAM_UNICODE_TABLE[cps[i]];
                    if (nextInfo && (cps[i] === 0xAA29 || cps[i] === 0xAA43 || cps[i] === 0xAA4C || cps[i] === 0xAA4D)) {
                        if (cps[i] === 0xAA29 && seg === "a") seg = "ā";
                        else if (nextInfo.coda) seg += nextInfo.coda;
                        i++;
                    } else {
                        break;
                    }
                }
                out += seg;
                continue;
            }

            // Standalone final consonant glyph (if unattached or following syllable)
            if (info.coda && !info.base) {
                out += info.coda;
                i++;
                continue;
            }

            // Base Consonant syllable assembly
            if (info.base) {
                let onset = info.base;
                let medials = "";
                let preVowel = "";
                let depVowel = "";
                let hasAA = false;
                let coda = "";
                i++;

                while (i < cps.length) {
                    const mCp = cps[i];
                    const mInfo = CHAM_UNICODE_TABLE[mCp];
                    if (!mInfo) break;

                    if (mInfo.medial) {
                        medials += mInfo.medial;
                        i++;
                    } else if (mCp === 0xAA2F || mCp === 0xAA30) {
                        preVowel = mInfo.vowel;
                        i++;
                    } else if (mCp >= 0xAA2A && mCp <= 0xAA32) {
                        depVowel = mInfo.vowel;
                        i++;
                    } else if (mCp === 0xAA29) {
                        hasAA = true;
                        i++;
                    } else if (mCp === 0xAA43 || mCp === 0xAA4C || mCp === 0xAA4D) {
                        coda += mInfo.coda;
                        i++;
                    } else if (mCp >= 0xAA40 && mCp <= 0xAA4B) {
                        coda += mInfo.coda;
                        i++;
                        break;
                    } else if (mCp === 0xAA25) {
                        // U+AA25 VA acts as final -w only when NOT followed by any dependent vowel/medial
                        const lookahead = i + 1 < cps.length ? cps[i + 1] : 0;
                        const laCombining = isCombiningCham(lookahead);
                        if (!laCombining) {
                            coda += "w";
                            i++;
                            break;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Compose vowel nucleus (including compound Cham diphthongs like ꨯ + ꨮ -> o / ô, ꨯ + ꨱ -> au)
                let nucleus = "";
                if (preVowel === "o" && depVowel === "ơ") {
                    nucleus = "ô";
                } else if (preVowel === "o" && depVowel === "au") {
                    nucleus = "ao";
                } else if (preVowel && depVowel) {
                    nucleus = preVowel + depVowel;
                } else if (preVowel) {
                    nucleus = hasAA ? (preVowel === "o" ? "ô" : preVowel) : preVowel;
                } else if (depVowel) {
                    nucleus = depVowel;
                } else if (hasAA) {
                    nucleus = "ā";
                } else if (medials === "i") {
                    nucleus = "ia";
                    medials = "";
                } else if (medials === "u") {
                    nucleus = "ua";
                    medials = "";
                } else {
                    nucleus = info.inherent || "a";
                }

                out += onset + medials + nucleus + coda;
                continue;
            }

            out += String.fromCodePoint(cp);
            i++;
        }

        return out.replace(/\s{2,}/g, " ").trim();
    }

    /**
     * Client-side canonical Logical Order normalizer fallback (mirrors scripts/generate_data.py).
     * Canonical order inside a consonant cluster:
     *   Base + Medial Ra/La (AA34, AA35) + Medial Ya/Wa (AA33, AA36) + Pre-Vowels (AA2F, AA30)
     *   + Other Dependent Vowels (AA2A..AA2E, AA31, AA32) + Lengthener AA (AA29) + Combining Codas (AA43, AA4C, AA4D)
     */
    function normalizeChamLogicalClient(text) {
        if (!text) return "";
        const cps = Array.from(text).map((c) => c.codePointAt(0));
        const rankOf = (cp) => {
            if (cp === 0xAA34 || cp === 0xAA35) return 1;
            if (cp === 0xAA33 || cp === 0xAA36) return 2;
            if (cp === 0xAA2F || cp === 0xAA30) return 3;
            if ((cp >= 0xAA2A && cp <= 0xAA2E) || cp === 0xAA31 || cp === 0xAA32) return 4;
            if (cp === 0xAA29) return 5;
            if (cp === 0xAA43 || cp === 0xAA4C || cp === 0xAA4D) return 6;
            return 0;
        };

        const result = [];
        let i = 0;
        while (i < cps.length) {
            const cp = cps[i];
            if ((cp >= 0xAA06 && cp <= 0xAA28) || (cp >= 0xAA00 && cp <= 0xAA05)) {
                result.push(cp);
                i++;
                const marks = [];
                while (i < cps.length && rankOf(cps[i]) > 0) {
                    marks.push(cps[i]);
                    i++;
                }
                if (marks.length > 1) {
                    marks.sort((a, b) => rankOf(a) - rankOf(b));
                }
                // Deduplicate accidental adjacent identical combining marks
                for (let k = 0; k < marks.length; k++) {
                    if (k === 0 || marks[k] !== marks[k - 1]) {
                        result.push(marks[k]);
                    }
                }
            } else {
                result.push(cp);
                i++;
            }
        }
        return String.fromCodePoint(...result);
    }

    global.ChamLinguistics = {
        CHAM_UNICODE_TABLE,
        formatCodepoint,
        isChamCodepoint,
        isCombiningCham,
        inspectCluster,
        transliterateCham,
        normalizeChamLogicalClient
    };
})(window);
