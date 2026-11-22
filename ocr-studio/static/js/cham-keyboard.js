/**
 * Cham Paleographic Virtual Keyboard (Akhar Thrah) — Pure Academic English Edition
 * Supports 5-row (Compact), 6-row (Balanced), and 9-row (Canonical) layouts,
 * Traditional & Sanskrit phonetic legends, draggable floating window, and non-blur cursor insertion.
 */
(function (global) {
    "use strict";

    const CHAM_KEYBOARD_LAYOUTS = {
        "5": [
            [
                { char: "꩑", display: "꩑", trad: "1", sanskrit: "1", title: "U+AA51 Cham Digit One" },
                { char: "꩒", display: "꩒", trad: "2", sanskrit: "2", title: "U+AA52 Cham Digit Two" },
                { char: "꩓", display: "꩓", trad: "3", sanskrit: "3", title: "U+AA53 Cham Digit Three" },
                { char: "꩔", display: "꩔", trad: "4", sanskrit: "4", title: "U+AA54 Cham Digit Four" },
                { char: "꩕", display: "꩕", trad: "5", sanskrit: "5", title: "U+AA55 Cham Digit Five" },
                { char: "꩖", display: "꩖", trad: "6", sanskrit: "6", title: "U+AA56 Cham Digit Six" },
                { char: "꩗", display: "꩗", trad: "7", sanskrit: "7", title: "U+AA57 Cham Digit Seven" },
                { char: "꩘", display: "꩘", trad: "8", sanskrit: "8", title: "U+AA58 Cham Digit Eight" },
                { char: "꩙", display: "꩙", trad: "9", sanskrit: "9", title: "U+AA59 Cham Digit Nine" },
                { char: "꩐", display: "꩐", trad: "0", sanskrit: "0", title: "U+AA50 Cham Digit Zero" },
                { char: "ꨀ", display: "ꨀ", trad: "a", sanskrit: "a", title: "U+AA00 Cham Letter A" },
                { char: "ꨁ", display: "ꨁ", trad: "i", sanskrit: "i", title: "U+AA01 Cham Letter I" },
                { char: "ꨂ", display: "ꨂ", trad: "u", sanskrit: "u", title: "U+AA02 Cham Letter U" },
                { char: "ꨃ", display: "ꨃ", trad: "e", sanskrit: "e", title: "U+AA03 Cham Letter E" },
                { char: "ꨄ", display: "ꨄ", trad: "ai", sanskrit: "ai", title: "U+AA04 Cham Letter Ai" },
                { char: "ꨅ", display: "ꨅ", trad: "o", sanskrit: "o", title: "U+AA05 Cham Letter O" },
                { char: "'", display: "'", trad: "'", sanskrit: "'", title: "Apostrophe" }
            ],
            [
                { char: "ꨆ", display: "ꨆ", trad: "ka", sanskrit: "ka", title: "U+AA06 Cham Letter Ka" },
                { char: "ꨇ", display: "ꨇ", trad: "kha", sanskrit: "kha", title: "U+AA07 Cham Letter Kha" },
                { char: "ꨈ", display: "ꨈ", trad: "ga", sanskrit: "ga", title: "U+AA08 Cham Letter Ga" },
                { char: "ꨉ", display: "ꨉ", trad: "gha", sanskrit: "gha", title: "U+AA09 Cham Letter Gha" },
                { char: "ꨊ", display: "ꨊ", trad: "ngâ", sanskrit: "ṅa", title: "U+AA0A Cham Letter Ngue" },
                { char: "ꨋ", display: "ꨋ", trad: "nga", sanskrit: "ṅga", title: "U+AA0B Cham Letter Nga" },
                { char: "ꨌ", display: "ꨌ", trad: "ca", sanskrit: "ca", title: "U+AA0C Cham Letter Cha" },
                { char: "ꨍ", display: "ꨍ", trad: "cha", sanskrit: "cha", title: "U+AA0D Cham Letter Chha" },
                { char: "ꨎ", display: "ꨎ", trad: "ja", sanskrit: "ja", title: "U+AA0E Cham Letter Ja" },
                { char: "ꨏ", display: "ꨏ", trad: "jha", sanskrit: "jha", title: "U+AA0F Cham Letter Jha" },
                { char: "ꨐ", display: "ꨐ", trad: "nyâ", sanskrit: "ña", title: "U+AA10 Cham Letter Nhue" },
                { char: "ꨑ", display: "ꨑ", trad: "nya", sanskrit: "ñja", title: "U+AA11 Cham Letter Nha" },
                { char: "ꨒ", display: "ꨒ", trad: "nja", sanskrit: "ṇḍa", title: "U+AA12 Cham Letter Nhja" },
                { char: "ꨓ", display: "ꨓ", trad: "ta", sanskrit: "ta", title: "U+AA13 Cham Letter Ta" },
                { char: "ꨔ", display: "ꨔ", trad: "tha", sanskrit: "tha", title: "U+AA14 Cham Letter Tha" },
                { char: "ꨕ", display: "ꨕ", trad: "da", sanskrit: "da", title: "U+AA15 Cham Letter Da" },
                { char: "ꨖ", display: "ꨖ", trad: "dha", sanskrit: "dha", title: "U+AA16 Cham Letter Dha" },
                { char: "ꨗ", display: "ꨗ", trad: "nâ", sanskrit: "na", title: "U+AA17 Cham Letter Nue" }
            ],
            [
                { char: "ꨘ", display: "ꨘ", trad: "na", sanskrit: "ṇa", title: "U+AA18 Cham Letter Na" },
                { char: "ꨙ", display: "ꨙ", trad: "nda", sanskrit: "nda", title: "U+AA19 Cham Letter Dda" },
                { char: "ꨚ", display: "ꨚ", trad: "pa", sanskrit: "pa", title: "U+AA1A Cham Letter Pa" },
                { char: "ꨛ", display: "ꨛ", trad: "ppa", sanskrit: "ppa", title: "U+AA1B Cham Letter Ppa" },
                { char: "ꨜ", display: "ꨜ", trad: "pha", sanskrit: "pha", title: "U+AA1C Cham Letter Pha" },
                { char: "ꨝ", display: "ꨝ", trad: "ba", sanskrit: "ba", title: "U+AA1D Cham Letter Ba" },
                { char: "ꨞ", display: "ꨞ", trad: "bha", sanskrit: "bha", title: "U+AA1E Cham Letter Bha" },
                { char: "ꨟ", display: "ꨟ", trad: "mâ", sanskrit: "ma", title: "U+AA1F Cham Letter Mue" },
                { char: "ꨠ", display: "ꨠ", trad: "ma", sanskrit: "mma", title: "U+AA20 Cham Letter Ma" },
                { char: "ꨡ", display: "ꨡ", trad: "mba", sanskrit: "mba", title: "U+AA21 Cham Letter Bba" },
                { char: "ꨢ", display: "ꨢ", trad: "ya", sanskrit: "ya", title: "U+AA22 Cham Letter Ya" },
                { char: "ꨣ", display: "ꨣ", trad: "ra", sanskrit: "ra", title: "U+AA23 Cham Letter Ra" },
                { char: "ꨤ", display: "ꨤ", trad: "la", sanskrit: "la", title: "U+AA24 Cham Letter La" },
                { char: "ꨥ", display: "ꨥ", trad: "wa", sanskrit: "va", title: "U+AA25 Cham Letter Va" },
                { char: "ꨦ", display: "ꨦ", trad: "śa", sanskrit: "śa", title: "U+AA26 Cham Letter Ssa" },
                { char: "ꨧ", display: "ꨧ", trad: "sa", sanskrit: "sa", title: "U+AA27 Cham Letter Sa" },
                { char: "ꨨ", display: "ꨨ", trad: "ha", sanskrit: "ha", title: "U+AA28 Cham Letter Ha" },
                { type: "backspace", display: "⌫", trad: "Back", sanskrit: "Back", title: "Backspace (Delete preceding character)" }
            ],
            [
                { char: "ꨩ", display: '<span class="dotted-circle">◌</span>ꨩ', trad: "-ā", sanskrit: "-ā", title: "U+AA29 Vowel Sign Aa" },
                { char: "ꨪ", display: '<span class="dotted-circle">◌</span>ꨪ', trad: "-i", sanskrit: "-i", title: "U+AA2A Vowel Sign I" },
                { char: "ꨫ", display: '<span class="dotted-circle">◌</span>ꨫ', trad: "-ī", sanskrit: "-ī", title: "U+AA2B Vowel Sign Ii" },
                { char: "ꨬ", display: '<span class="dotted-circle">◌</span>ꨬ', trad: "-ei", sanskrit: "-ei", title: "U+AA2C Vowel Sign Ei" },
                { char: "ꨭ", display: '<span class="dotted-circle">◌</span>ꨭ', trad: "-u", sanskrit: "-u", title: "U+AA2D Vowel Sign U" },
                { char: "ꨮ", display: '<span class="dotted-circle">◌</span>ꨮ', trad: "-ơ", sanskrit: "-ē", title: "U+AA2E Vowel Sign Oe" },
                { char: "ꨯ", display: 'ꨯ<span class="dotted-circle">◌</span>', trad: "-o", sanskrit: "-o", title: "U+AA2F Vowel Sign O (Pre-base)" },
                { char: "ꨰ", display: 'ꨰ<span class="dotted-circle">◌</span>', trad: "-ai", sanskrit: "-ai", title: "U+AA30 Vowel Sign Ai (Pre-base)" },
                { char: "ꨱ", display: '<span class="dotted-circle">◌</span>ꨱ', trad: "-au", sanskrit: "-ao", title: "U+AA31 Vowel Sign Au" },
                { char: "ꨲ", display: '<span class="dotted-circle">◌</span>ꨲ', trad: "-ư", sanskrit: "-au", title: "U+AA32 Vowel Sign Ue" },
                { char: "ꨳ", display: '<span class="dotted-circle">◌</span>ꨳ', trad: "-ia", sanskrit: "-ya", title: "U+AA33 Medial Ya" },
                { char: "ꨴ", display: '<span class="dotted-circle">◌</span>ꨴ', trad: "-r", sanskrit: "-ra", title: "U+AA34 Medial Ra" },
                { char: "ꨵ", display: '<span class="dotted-circle">◌</span>ꨵ', trad: "-l", sanskrit: "-la", title: "U+AA35 Medial La" },
                { char: "ꨶ", display: '<span class="dotted-circle">◌</span>ꨶ', trad: "-ua", sanskrit: "-va", title: "U+AA36 Medial Wa" },
                { char: "꩜", display: "꩜", trad: "§", sanskrit: "§", title: "U+AA5C Cham Spiral" },
                { char: "꩝", display: "꩝", trad: "|", sanskrit: "|", title: "U+AA5D Cham Danda" },
                { char: "꩞", display: "꩞", trad: "||", sanskrit: "||", title: "U+AA5E Cham Double Danda" },
                { char: "꩟", display: "꩟", trad: "|||", sanskrit: "|||", title: "U+AA5F Cham Triple Danda" }
            ],
            [
                { char: "ꩀ", display: "ꩀ", trad: "-k", sanskrit: "-k", title: "U+AA40 Final K" },
                { char: "ꩁ", display: "ꩁ", trad: "-g", sanskrit: "-g", title: "U+AA41 Final G" },
                { char: "ꩂ", display: "ꩂ", trad: "-ng", sanskrit: "-ṅ", title: "U+AA42 Final Ng" },
                { char: "ꩃ", display: '<span class="dotted-circle">◌</span>ꩃ', trad: "-ng", sanskrit: "-ṅg", title: "U+AA43 Sign Final Ng" },
                { char: "ꩄ", display: "ꩄ", trad: "-c", sanskrit: "-c", title: "U+AA44 Final Ch" },
                { char: "ꩅ", display: "ꩅ", trad: "-t", sanskrit: "-t", title: "U+AA45 Final T" },
                { char: "ꩆ", display: "ꩆ", trad: "-n", sanskrit: "-n", title: "U+AA46 Final N" },
                { char: "ꩇ", display: "ꩇ", trad: "-p", sanskrit: "-p", title: "U+AA47 Final P" },
                { char: "ꩈ", display: "ꩈ", trad: "-y", sanskrit: "-y", title: "U+AA48 Final Y" },
                { char: "ꩉ", display: "ꩉ", trad: "-r", sanskrit: "-r", title: "U+AA49 Final R" },
                { char: "ꩊ", display: "ꩊ", trad: "-l", sanskrit: "-l", title: "U+AA4A Final L" },
                { char: "ꩋ", display: "ꩋ", trad: "-s", sanskrit: "-s", title: "U+AA4B Final Ss" },
                { char: "ꩌ", display: '<span class="dotted-circle">◌</span>ꩌ', trad: "-m", sanskrit: "-ṃ", title: "U+AA4C Sign Final M (Anusvara)" },
                { char: "ꩍ", display: '<span class="dotted-circle">◌</span>ꩍ', trad: "-h", sanskrit: "-ḥ", title: "U+AA4D Sign Final H (Visarga)" },
                { char: "ꨥ", display: "ꨥ", trad: "-w", sanskrit: "-v", title: "U+AA25 Syllable-Final Va" },
                { char: "(", display: "(", trad: "(", sanskrit: "(", title: "Left Parenthesis" },
                { char: ")", display: ")", trad: ")", sanskrit: ")", title: "Right Parenthesis" },
                { type: "space", char: " ", display: "ꨌꩌꨛꨩ", trad: "Space", sanskrit: "Space", title: "Insert Whitespace" },
                { type: "enter", display: "↵", trad: "Commit", sanskrit: "Commit", title: "Commit Active Line Edit" }
            ]
        ],
        "6": [
            [
                { char: "꩑", display: "꩑", trad: "1", sanskrit: "1", title: "U+AA51 Cham Digit One" },
                { char: "꩒", display: "꩒", trad: "2", sanskrit: "2", title: "U+AA52 Cham Digit Two" },
                { char: "꩓", display: "꩓", trad: "3", sanskrit: "3", title: "U+AA53 Cham Digit Three" },
                { char: "꩔", display: "꩔", trad: "4", sanskrit: "4", title: "U+AA54 Cham Digit Four" },
                { char: "꩕", display: "꩕", trad: "5", sanskrit: "5", title: "U+AA55 Cham Digit Five" },
                { char: "꩖", display: "꩖", trad: "6", sanskrit: "6", title: "U+AA56 Cham Digit Six" },
                { char: "꩗", display: "꩗", trad: "7", sanskrit: "7", title: "U+AA57 Cham Digit Seven" },
                { char: "꩘", display: "꩘", trad: "8", sanskrit: "8", title: "U+AA58 Cham Digit Eight" },
                { char: "꩙", display: "꩙", trad: "9", sanskrit: "9", title: "U+AA59 Cham Digit Nine" },
                { char: "꩐", display: "꩐", trad: "0", sanskrit: "0", title: "U+AA50 Cham Digit Zero" },
                { char: "ꨀ", display: "ꨀ", trad: "a", sanskrit: "a", title: "U+AA00 Cham Letter A" },
                { char: "ꨁ", display: "ꨁ", trad: "i", sanskrit: "i", title: "U+AA01 Cham Letter I" },
                { char: "ꨂ", display: "ꨂ", trad: "u", sanskrit: "u", title: "U+AA02 Cham Letter U" },
                { char: "ꨃ", display: "ꨃ", trad: "e", sanskrit: "e", title: "U+AA03 Cham Letter E" }
            ],
            [
                { char: "ꨄ", display: "ꨄ", trad: "ai", sanskrit: "ai", title: "U+AA04 Cham Letter Ai" },
                { char: "ꨅ", display: "ꨅ", trad: "o", sanskrit: "o", title: "U+AA05 Cham Letter O" },
                { char: "ꨆ", display: "ꨆ", trad: "ka", sanskrit: "ka", title: "U+AA06 Cham Letter Ka" },
                { char: "ꨇ", display: "ꨇ", trad: "kha", sanskrit: "kha", title: "U+AA07 Cham Letter Kha" },
                { char: "ꨈ", display: "ꨈ", trad: "ga", sanskrit: "ga", title: "U+AA08 Cham Letter Ga" },
                { char: "ꨉ", display: "ꨉ", trad: "gha", sanskrit: "gha", title: "U+AA09 Cham Letter Gha" },
                { char: "ꨊ", display: "ꨊ", trad: "ngâ", sanskrit: "ṅa", title: "U+AA0A Cham Letter Ngue" },
                { char: "ꨋ", display: "ꨋ", trad: "nga", sanskrit: "ṅga", title: "U+AA0B Cham Letter Nga" },
                { char: "ꨌ", display: "ꨌ", trad: "ca", sanskrit: "ca", title: "U+AA0C Cham Letter Cha" },
                { char: "ꨍ", display: "ꨍ", trad: "cha", sanskrit: "cha", title: "U+AA0D Cham Letter Chha" },
                { char: "ꨎ", display: "ꨎ", trad: "ja", sanskrit: "ja", title: "U+AA0E Cham Letter Ja" },
                { char: "ꨏ", display: "ꨏ", trad: "jha", sanskrit: "jha", title: "U+AA0F Cham Letter Jha" },
                { char: "ꨐ", display: "ꨐ", trad: "nyâ", sanskrit: "ña", title: "U+AA10 Cham Letter Nhue" },
                { char: "ꨑ", display: "ꨑ", trad: "nya", sanskrit: "ñja", title: "U+AA11 Cham Letter Nha" }
            ],
            [
                { char: "ꨒ", display: "ꨒ", trad: "nja", sanskrit: "ṇḍa", title: "U+AA12 Cham Letter Nhja" },
                { char: "ꨓ", display: "ꨓ", trad: "ta", sanskrit: "ta", title: "U+AA13 Cham Letter Ta" },
                { char: "ꨔ", display: "ꨔ", trad: "tha", sanskrit: "tha", title: "U+AA14 Cham Letter Tha" },
                { char: "ꨕ", display: "ꨕ", trad: "da", sanskrit: "da", title: "U+AA15 Cham Letter Da" },
                { char: "ꨖ", display: "ꨖ", trad: "dha", sanskrit: "dha", title: "U+AA16 Cham Letter Dha" },
                { char: "ꨗ", display: "ꨗ", trad: "nâ", sanskrit: "na", title: "U+AA17 Cham Letter Nue" },
                { char: "ꨘ", display: "ꨘ", trad: "na", sanskrit: "ṇa", title: "U+AA18 Cham Letter Na" },
                { char: "ꨙ", display: "ꨙ", trad: "nda", sanskrit: "nda", title: "U+AA19 Cham Letter Dda" },
                { char: "ꨚ", display: "ꨚ", trad: "pa", sanskrit: "pa", title: "U+AA1A Cham Letter Pa" },
                { char: "ꨛ", display: "ꨛ", trad: "ppa", sanskrit: "ppa", title: "U+AA1B Cham Letter Ppa" },
                { char: "ꨜ", display: "ꨜ", trad: "pha", sanskrit: "pha", title: "U+AA1C Cham Letter Pha" },
                { char: "ꨝ", display: "ꨝ", trad: "ba", sanskrit: "ba", title: "U+AA1D Cham Letter Ba" },
                { char: "ꨞ", display: "ꨞ", trad: "bha", sanskrit: "bha", title: "U+AA1E Cham Letter Bha" },
                { char: "ꨟ", display: "ꨟ", trad: "mâ", sanskrit: "ma", title: "U+AA1F Cham Letter Mue" }
            ],
            [
                { char: "ꨠ", display: "ꨠ", trad: "ma", sanskrit: "mma", title: "U+AA20 Cham Letter Ma" },
                { char: "ꨡ", display: "ꨡ", trad: "mba", sanskrit: "mba", title: "U+AA21 Cham Letter Bba" },
                { char: "ꨢ", display: "ꨢ", trad: "ya", sanskrit: "ya", title: "U+AA22 Cham Letter Ya" },
                { char: "ꨣ", display: "ꨣ", trad: "ra", sanskrit: "ra", title: "U+AA23 Cham Letter Ra" },
                { char: "ꨤ", display: "ꨤ", trad: "la", sanskrit: "la", title: "U+AA24 Cham Letter La" },
                { char: "ꨥ", display: "ꨥ", trad: "wa", sanskrit: "va", title: "U+AA25 Cham Letter Va" },
                { char: "ꨦ", display: "ꨦ", trad: "śa", sanskrit: "śa", title: "U+AA26 Cham Letter Ssa" },
                { char: "ꨧ", display: "ꨧ", trad: "sa", sanskrit: "sa", title: "U+AA27 Cham Letter Sa" },
                { char: "ꨨ", display: "ꨨ", trad: "ha", sanskrit: "ha", title: "U+AA28 Cham Letter Ha" },
                { char: "ꨳ", display: '<span class="dotted-circle">◌</span>ꨳ', trad: "-ia", sanskrit: "-ya", title: "U+AA33 Medial Ya" },
                { char: "ꨴ", display: '<span class="dotted-circle">◌</span>ꨴ', trad: "-r", sanskrit: "-ra", title: "U+AA34 Medial Ra" },
                { char: "ꨵ", display: '<span class="dotted-circle">◌</span>ꨵ', trad: "-l", sanskrit: "-la", title: "U+AA35 Medial La" },
                { char: "ꨶ", display: '<span class="dotted-circle">◌</span>ꨶ', trad: "-ua", sanskrit: "-va", title: "U+AA36 Medial Wa" },
                { type: "backspace", display: "⌫", trad: "Back", sanskrit: "Back", title: "Backspace" }
            ],
            [
                { char: "ꨩ", display: '<span class="dotted-circle">◌</span>ꨩ', trad: "-ā", sanskrit: "-ā", title: "U+AA29 Vowel Sign Aa" },
                { char: "ꨪ", display: '<span class="dotted-circle">◌</span>ꨪ', trad: "-i", sanskrit: "-i", title: "U+AA2A Vowel Sign I" },
                { char: "ꨫ", display: '<span class="dotted-circle">◌</span>ꨫ', trad: "-ī", sanskrit: "-ī", title: "U+AA2B Vowel Sign Ii" },
                { char: "ꨬ", display: '<span class="dotted-circle">◌</span>ꨬ', trad: "-ei", sanskrit: "-ei", title: "U+AA2C Vowel Sign Ei" },
                { char: "ꨭ", display: '<span class="dotted-circle">◌</span>ꨭ', trad: "-u", sanskrit: "-u", title: "U+AA2D Vowel Sign U" },
                { char: "ꨮ", display: '<span class="dotted-circle">◌</span>ꨮ', trad: "-ơ", sanskrit: "-ē", title: "U+AA2E Vowel Sign Oe" },
                { char: "ꨯ", display: 'ꨯ<span class="dotted-circle">◌</span>', trad: "-o", sanskrit: "-o", title: "U+AA2F Vowel Sign O" },
                { char: "ꨰ", display: 'ꨰ<span class="dotted-circle">◌</span>', trad: "-ai", sanskrit: "-ai", title: "U+AA30 Vowel Sign Ai" },
                { char: "ꨱ", display: '<span class="dotted-circle">◌</span>ꨱ', trad: "-au", sanskrit: "-ao", title: "U+AA31 Vowel Sign Au" },
                { char: "ꨲ", display: '<span class="dotted-circle">◌</span>ꨲ', trad: "-ư", sanskrit: "-au", title: "U+AA32 Vowel Sign Ue" },
                { char: "꩜", display: "꩜", trad: "§", sanskrit: "§", title: "U+AA5C Cham Spiral" },
                { char: "꩝", display: "꩝", trad: "|", sanskrit: "|", title: "U+AA5D Cham Danda" },
                { char: "꩞", display: "꩞", trad: "||", sanskrit: "||", title: "U+AA5E Cham Double Danda" },
                { char: "꩟", display: "꩟", trad: "|||", sanskrit: "|||", title: "U+AA5F Cham Triple Danda" }
            ],
            [
                { char: "ꩀ", display: "ꩀ", trad: "-k", sanskrit: "-k", title: "U+AA40 Final K" },
                { char: "ꩁ", display: "ꩁ", trad: "-g", sanskrit: "-g", title: "U+AA41 Final G" },
                { char: "ꩂ", display: "ꩂ", trad: "-ng", sanskrit: "-ṅ", title: "U+AA42 Final Ng" },
                { char: "ꩃ", display: '<span class="dotted-circle">◌</span>ꩃ', trad: "-ng", sanskrit: "-ṅg", title: "U+AA43 Sign Final Ng" },
                { char: "ꩄ", display: "ꩄ", trad: "-c", sanskrit: "-c", title: "U+AA44 Final Ch" },
                { char: "ꩅ", display: "ꩅ", trad: "-t", sanskrit: "-t", title: "U+AA45 Final T" },
                { char: "ꩆ", display: "ꩆ", trad: "-n", sanskrit: "-n", title: "U+AA46 Final N" },
                { char: "ꩇ", display: "ꩇ", trad: "-p", sanskrit: "-p", title: "U+AA47 Final P" },
                { char: "ꩈ", display: "ꩈ", trad: "-y", sanskrit: "-y", title: "U+AA48 Final Y" },
                { char: "ꩉ", display: "ꩉ", trad: "-r", sanskrit: "-r", title: "U+AA49 Final R" },
                { char: "ꩊ", display: "ꩊ", trad: "-l", sanskrit: "-l", title: "U+AA4A Final L" },
                { char: "ꩋ", display: "ꩋ", trad: "-s", sanskrit: "-s", title: "U+AA4B Final Ss" },
                { char: "ꩌ", display: '<span class="dotted-circle">◌</span>ꩌ', trad: "-m", sanskrit: "-ṃ", title: "U+AA4C Sign Final M" },
                { char: "ꩍ", display: '<span class="dotted-circle">◌</span>ꩍ', trad: "-h", sanskrit: "-ḥ", title: "U+AA4D Sign Final H" },
                { char: "ꨥ", display: "ꨥ", trad: "-w", sanskrit: "-v", title: "U+AA25 Final Va" },
                { char: "(", display: "(", trad: "(", sanskrit: "(", title: "Left Parenthesis" },
                { char: ")", display: ")", trad: ")", sanskrit: ")", title: "Right Parenthesis" },
                { char: "'", display: "'", trad: "'", sanskrit: "'", title: "Apostrophe" },
                { type: "space", char: " ", display: "ꨌꩌꨛꨩ", trad: "Space", sanskrit: "Space", title: "Space" },
                { type: "enter", display: "↵", trad: "Commit", sanskrit: "Commit", title: "Commit Active Line Edit" }
            ]
        ],
        "9": [
            [
                { char: "꩑", display: "꩑", trad: "1", sanskrit: "1", title: "U+AA51 Cham Digit One" },
                { char: "꩒", display: "꩒", trad: "2", sanskrit: "2", title: "U+AA52 Cham Digit Two" },
                { char: "꩓", display: "꩓", trad: "3", sanskrit: "3", title: "U+AA53 Cham Digit Three" },
                { char: "꩔", display: "꩔", trad: "4", sanskrit: "4", title: "U+AA54 Cham Digit Four" },
                { char: "꩕", display: "꩕", trad: "5", sanskrit: "5", title: "U+AA55 Cham Digit Five" },
                { char: "꩖", display: "꩖", trad: "6", sanskrit: "6", title: "U+AA56 Cham Digit Six" },
                { char: "꩗", display: "꩗", trad: "7", sanskrit: "7", title: "U+AA57 Cham Digit Seven" },
                { char: "꩘", display: "꩘", trad: "8", sanskrit: "8", title: "U+AA58 Cham Digit Eight" },
                { char: "꩙", display: "꩙", trad: "9", sanskrit: "9", title: "U+AA59 Cham Digit Nine" },
                { char: "꩐", display: "꩐", trad: "0", sanskrit: "0", title: "U+AA50 Cham Digit Zero" }
            ],
            [
                { char: "ꨀ", display: "ꨀ", trad: "a", sanskrit: "a", title: "U+AA00 Cham Letter A" },
                { char: "ꨁ", display: "ꨁ", trad: "i", sanskrit: "i", title: "U+AA01 Cham Letter I" },
                { char: "ꨂ", display: "ꨂ", trad: "u", sanskrit: "u", title: "U+AA02 Cham Letter U" },
                { char: "ꨃ", display: "ꨃ", trad: "e", sanskrit: "e", title: "U+AA03 Cham Letter E" },
                { char: "ꨄ", display: "ꨄ", trad: "ai", sanskrit: "ai", title: "U+AA04 Cham Letter Ai" },
                { char: "ꨅ", display: "ꨅ", trad: "o", sanskrit: "o", title: "U+AA05 Cham Letter O" },
                { char: "ꨆ", display: "ꨆ", trad: "ka", sanskrit: "ka", title: "U+AA06 Cham Letter Ka" },
                { char: "ꨇ", display: "ꨇ", trad: "kha", sanskrit: "kha", title: "U+AA07 Cham Letter Kha" },
                { char: "ꨈ", display: "ꨈ", trad: "ga", sanskrit: "ga", title: "U+AA08 Cham Letter Ga" },
                { char: "ꨉ", display: "ꨉ", trad: "gha", sanskrit: "gha", title: "U+AA09 Cham Letter Gha" }
            ],
            [
                { char: "ꨊ", display: "ꨊ", trad: "ngâ", sanskrit: "ṅa", title: "U+AA0A Cham Letter Ngue" },
                { char: "ꨋ", display: "ꨋ", trad: "nga", sanskrit: "ṅga", title: "U+AA0B Cham Letter Nga" },
                { char: "ꨌ", display: "ꨌ", trad: "ca", sanskrit: "ca", title: "U+AA0C Cham Letter Cha" },
                { char: "ꨍ", display: "ꨍ", trad: "cha", sanskrit: "cha", title: "U+AA0D Cham Letter Chha" },
                { char: "ꨎ", display: "ꨎ", trad: "ja", sanskrit: "ja", title: "U+AA0E Cham Letter Ja" },
                { char: "ꨏ", display: "ꨏ", trad: "jha", sanskrit: "jha", title: "U+AA0F Cham Letter Jha" },
                { char: "ꨐ", display: "ꨐ", trad: "nyâ", sanskrit: "ña", title: "U+AA10 Cham Letter Nhue" },
                { char: "ꨑ", display: "ꨑ", trad: "nya", sanskrit: "ñja", title: "U+AA11 Cham Letter Nha" },
                { char: "ꨒ", display: "ꨒ", trad: "nja", sanskrit: "ṇḍa", title: "U+AA12 Cham Letter Nhja" },
                { char: "ꨓ", display: "ꨓ", trad: "ta", sanskrit: "ta", title: "U+AA13 Cham Letter Ta" }
            ],
            [
                { char: "ꨔ", display: "ꨔ", trad: "tha", sanskrit: "tha", title: "U+AA14 Cham Letter Tha" },
                { char: "ꨕ", display: "ꨕ", trad: "da", sanskrit: "da", title: "U+AA15 Cham Letter Da" },
                { char: "ꨖ", display: "ꨖ", trad: "dha", sanskrit: "dha", title: "U+AA16 Cham Letter Dha" },
                { char: "ꨗ", display: "ꨗ", trad: "nâ", sanskrit: "na", title: "U+AA17 Cham Letter Nue" },
                { char: "ꨘ", display: "ꨘ", trad: "na", sanskrit: "ṇa", title: "U+AA18 Cham Letter Na" },
                { char: "ꨙ", display: "ꨙ", trad: "nda", sanskrit: "nda", title: "U+AA19 Cham Letter Dda" },
                { char: "ꨚ", display: "ꨚ", trad: "pa", sanskrit: "pa", title: "U+AA1A Cham Letter Pa" },
                { char: "ꨛ", display: "ꨛ", trad: "ppa", sanskrit: "ppa", title: "U+AA1B Cham Letter Ppa" },
                { char: "ꨜ", display: "ꨜ", trad: "pha", sanskrit: "pha", title: "U+AA1C Cham Letter Pha" },
                { char: "ꨝ", display: "ꨝ", trad: "ba", sanskrit: "ba", title: "U+AA1D Cham Letter Ba" }
            ],
            [
                { char: "ꨞ", display: "ꨞ", trad: "bha", sanskrit: "bha", title: "U+AA1E Cham Letter Bha" },
                { char: "ꨟ", display: "ꨟ", trad: "mâ", sanskrit: "ma", title: "U+AA1F Cham Letter Mue" },
                { char: "ꨠ", display: "ꨠ", trad: "ma", sanskrit: "mma", title: "U+AA20 Cham Letter Ma" },
                { char: "ꨡ", display: "ꨡ", trad: "mba", sanskrit: "mba", title: "U+AA21 Cham Letter Bba" },
                { char: "ꨢ", display: "ꨢ", trad: "ya", sanskrit: "ya", title: "U+AA22 Cham Letter Ya" },
                { char: "ꨣ", display: "ꨣ", trad: "ra", sanskrit: "ra", title: "U+AA23 Cham Letter Ra" },
                { char: "ꨤ", display: "ꨤ", trad: "la", sanskrit: "la", title: "U+AA24 Cham Letter La" },
                { char: "ꨥ", display: "ꨥ", trad: "wa", sanskrit: "va", title: "U+AA25 Cham Letter Va" },
                { char: "ꨦ", display: "ꨦ", trad: "śa", sanskrit: "śa", title: "U+AA26 Cham Letter Ssa" },
                { char: "ꨧ", display: "ꨧ", trad: "sa", sanskrit: "sa", title: "U+AA27 Cham Letter Sa" }
            ],
            [
                { char: "ꨨ", display: "ꨨ", trad: "ha", sanskrit: "ha", title: "U+AA28 Cham Letter Ha" },
                { char: "ꨩ", display: '<span class="dotted-circle">◌</span>ꨩ', trad: "-ā", sanskrit: "-ā", title: "U+AA29 Vowel Sign Aa" },
                { char: "ꨪ", display: '<span class="dotted-circle">◌</span>ꨪ', trad: "-i", sanskrit: "-i", title: "U+AA2A Vowel Sign I" },
                { char: "ꨫ", display: '<span class="dotted-circle">◌</span>ꨫ', trad: "-ī", sanskrit: "-ī", title: "U+AA2B Vowel Sign Ii" },
                { char: "ꨬ", display: '<span class="dotted-circle">◌</span>ꨬ', trad: "-ei", sanskrit: "-ei", title: "U+AA2C Vowel Sign Ei" },
                { char: "ꨭ", display: '<span class="dotted-circle">◌</span>ꨭ', trad: "-u", sanskrit: "-u", title: "U+AA2D Vowel Sign U" },
                { char: "ꨮ", display: '<span class="dotted-circle">◌</span>ꨮ', trad: "-ơ", sanskrit: "-ē", title: "U+AA2E Vowel Sign Oe" },
                { char: "ꨯ", display: 'ꨯ<span class="dotted-circle">◌</span>', trad: "-o", sanskrit: "-o", title: "U+AA2F Vowel Sign O" },
                { char: "ꨰ", display: 'ꨰ<span class="dotted-circle">◌</span>', trad: "-ai", sanskrit: "-ai", title: "U+AA30 Vowel Sign Ai" },
                { char: "ꨱ", display: '<span class="dotted-circle">◌</span>ꨱ', trad: "-au", sanskrit: "-ao", title: "U+AA31 Vowel Sign Au" }
            ],
            [
                { char: "ꨲ", display: '<span class="dotted-circle">◌</span>ꨲ', trad: "-ư", sanskrit: "-au", title: "U+AA32 Vowel Sign Ue" },
                { char: "ꨳ", display: '<span class="dotted-circle">◌</span>ꨳ', trad: "-ia", sanskrit: "-ya", title: "U+AA33 Medial Ya" },
                { char: "ꨴ", display: '<span class="dotted-circle">◌</span>ꨴ', trad: "-r", sanskrit: "-ra", title: "U+AA34 Medial Ra" },
                { char: "ꨵ", display: '<span class="dotted-circle">◌</span>ꨵ', trad: "-l", sanskrit: "-la", title: "U+AA35 Medial La" },
                { char: "ꨶ", display: '<span class="dotted-circle">◌</span>ꨶ', trad: "-ua", sanskrit: "-va", title: "U+AA36 Medial Wa" },
                { char: "ꩀ", display: "ꩀ", trad: "-k", sanskrit: "-k", title: "U+AA40 Final K" },
                { char: "ꩁ", display: "ꩁ", trad: "-g", sanskrit: "-g", title: "U+AA41 Final G" },
                { char: "ꩂ", display: "ꩂ", trad: "-ng", sanskrit: "-ṅ", title: "U+AA42 Final Ng" },
                { char: "ꩃ", display: '<span class="dotted-circle">◌</span>ꩃ', trad: "-ng", sanskrit: "-ṅg", title: "U+AA43 Sign Final Ng" },
                { char: "ꩄ", display: "ꩄ", trad: "-c", sanskrit: "-c", title: "U+AA44 Final Ch" }
            ],
            [
                { char: "ꩅ", display: "ꩅ", trad: "-t", sanskrit: "-t", title: "U+AA45 Final T" },
                { char: "ꩆ", display: "ꩆ", trad: "-n", sanskrit: "-n", title: "U+AA46 Final N" },
                { char: "ꩇ", display: "ꩇ", trad: "-p", sanskrit: "-p", title: "U+AA47 Final P" },
                { char: "ꩈ", display: "ꩈ", trad: "-y", sanskrit: "-y", title: "U+AA48 Final Y" },
                { char: "ꩉ", display: "ꩉ", trad: "-r", sanskrit: "-r", title: "U+AA49 Final R" },
                { char: "ꩊ", display: "ꩊ", trad: "-l", sanskrit: "-l", title: "U+AA4A Final L" },
                { char: "ꩋ", display: "ꩋ", trad: "-s", sanskrit: "-s", title: "U+AA4B Final Ss" },
                { char: "ꩌ", display: '<span class="dotted-circle">◌</span>ꩌ', trad: "-m", sanskrit: "-ṃ", title: "U+AA4C Sign Final M" },
                { char: "ꩍ", display: '<span class="dotted-circle">◌</span>ꩍ', trad: "-h", sanskrit: "-ḥ", title: "U+AA4D Sign Final H" },
                { char: "ꨥ", display: "ꨥ", trad: "-w", sanskrit: "-v", title: "U+AA25 Final Va" },
                { type: "backspace", display: "⌫", trad: "Back", sanskrit: "Back", title: "Backspace" }
            ],
            [
                { char: "꩜", display: "꩜", trad: "§", sanskrit: "§", title: "U+AA5C Cham Spiral" },
                { char: "꩝", display: "꩝", trad: "|", sanskrit: "|", title: "U+AA5D Cham Danda" },
                { char: "꩞", display: "꩞", trad: "||", sanskrit: "||", title: "U+AA5E Cham Double Danda" },
                { char: "꩟", display: "꩟", trad: "|||", sanskrit: "|||", title: "U+AA5F Cham Triple Danda" },
                { type: "space", char: " ", display: "ꨌꩌꨛꨩ", trad: "Space", sanskrit: "Space", title: "Space" },
                { char: "(", display: "(", trad: "(", sanskrit: "(", title: "Left Parenthesis" },
                { char: ")", display: ")", trad: ")", sanskrit: ")", title: "Right Parenthesis" },
                { char: "'", display: "'", trad: "'", sanskrit: "'", title: "Apostrophe" },
                { type: "enter", display: "↵", trad: "Commit", sanskrit: "Commit", title: "Commit Active Line Edit" }
            ]
        ]
    };

    const state = {
        isOpen: false,
        rowLayout: localStorage.getItem("cham_kb_row_layout") || "5",
        phoneticMode: localStorage.getItem("cham_kb_phonetic") || "traditional",
        fontSize: parseInt(localStorage.getItem("cham_kb_font_size") || "18", 10),
        dragInit: false
    };

    function renderGrid() {
        const body = document.getElementById("cham-kb-body");
        if (!body) return;
        const rows = CHAM_KEYBOARD_LAYOUTS[state.rowLayout] || CHAM_KEYBOARD_LAYOUTS["5"];
        body.innerHTML = "";

        rows.forEach((rowKeys) => {
            const rowEl = document.createElement("div");
            rowEl.className = "cham-kb-row";
            rowKeys.forEach((key) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "cham-kb-key";
                if (key.type === "space") btn.classList.add("key-space");
                if (key.type === "backspace" || key.type === "enter") btn.classList.add("key-action");
                btn.title = key.title || "";

                // Prevent stealing focus from active transcription line input!
                btn.addEventListener("mousedown", (e) => e.preventDefault());
                btn.addEventListener("click", () => handleVirtualKeyPress(key));

                let subLabel = "";
                if (state.phoneticMode === "traditional" && key.trad) subLabel = key.trad;
                else if (state.phoneticMode === "sanskrit" && key.sanskrit) subLabel = key.sanskrit;

                btn.innerHTML = `
                    <span class="cham-kb-glyph" style="font-size:${state.fontSize}px">${key.display}</span>
                    ${subLabel ? `<span class="cham-kb-sub">${subLabel}</span>` : ""}
                `;
                rowEl.appendChild(btn);
            });
            body.appendChild(rowEl);
        });
    }

    function getTargetInput() {
        const activeTarget = global.StudioApp && global.StudioApp.getActiveLineInput
            ? global.StudioApp.getActiveLineInput()
            : null;
        if (activeTarget && document.body.contains(activeTarget)) return activeTarget;
        return document.getElementById("cham-kb-scratchpad");
    }

    function handleVirtualKeyPress(key) {
        if (key.type === "enter") {
            if (global.StudioApp && global.StudioApp.commitActiveEdit) {
                global.StudioApp.commitActiveEdit();
            }
            return;
        }

        const input = getTargetInput();
        if (!input) return;

        const start = typeof input.selectionStart === "number" ? input.selectionStart : input.value.length;
        const end = typeof input.selectionEnd === "number" ? input.selectionEnd : input.value.length;
        const val = input.value || "";

        if (key.type === "backspace") {
            if (start !== end) {
                input.value = val.slice(0, start) + val.slice(end);
                input.setSelectionRange(start, start);
            } else if (start > 0) {
                const chars = Array.from(val.slice(0, start));
                chars.pop();
                const prefix = chars.join("");
                input.value = prefix + val.slice(end);
                input.setSelectionRange(prefix.length, prefix.length);
            }
        } else {
            const insertStr = key.char || "";
            input.value = val.slice(0, start) + insertStr + val.slice(end);
            const nextPos = start + insertStr.length;
            input.setSelectionRange(nextPos, nextPos);
        }

        input.focus();
        input.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function toggle(forceState) {
        const panel = document.getElementById("cham-kb-panel");
        const triggerBtn = document.getElementById("btn-header-keyboard");
        if (!panel) return;

        state.isOpen = typeof forceState === "boolean" ? forceState : !state.isOpen;
        if (state.isOpen) {
            renderGrid();
            initDrag();
            panel.classList.remove("closed");
            panel.classList.add("open");
            if (triggerBtn) triggerBtn.classList.add("active");
        } else {
            panel.classList.remove("open");
            panel.classList.add("closed");
            if (triggerBtn) triggerBtn.classList.remove("active");
        }
    }

    function setRowLayout(val) {
        state.rowLayout = String(val);
        localStorage.setItem("cham_kb_row_layout", state.rowLayout);
        renderGrid();
    }

    function setPhoneticMode(val) {
        state.phoneticMode = String(val);
        localStorage.setItem("cham_kb_phonetic", state.phoneticMode);
        renderGrid();
    }

    function setFontSize(val) {
        state.fontSize = parseInt(val, 10) || 18;
        localStorage.setItem("cham_kb_font_size", String(state.fontSize));
        renderGrid();
    }

    function resetPosition() {
        const panel = document.getElementById("cham-kb-panel");
        if (!panel) return;
        panel.style.top = "";
        panel.style.left = "";
        panel.style.bottom = "20px";
        panel.style.right = "20px";
        if (global.StudioApp && global.StudioApp.toast) {
            global.StudioApp.toast("Keyboard docked to default corner");
        }
    }

    function copyScratchpad() {
        const sp = document.getElementById("cham-kb-scratchpad");
        if (!sp || !sp.value) return;
        navigator.clipboard.writeText(sp.value).then(() => {
            if (global.StudioApp) global.StudioApp.toast("Scratchpad copied to clipboard");
        });
    }

    function insertScratchpadToLine() {
        const sp = document.getElementById("cham-kb-scratchpad");
        if (!sp || !sp.value) return;
        if (global.StudioApp && global.StudioApp.insertTextIntoSelectedLine) {
            global.StudioApp.insertTextIntoSelectedLine(sp.value);
        }
    }

    function clearScratchpad() {
        const sp = document.getElementById("cham-kb-scratchpad");
        if (sp) sp.value = "";
    }

    function normalizeScratchpad() {
        const sp = document.getElementById("cham-kb-scratchpad");
        if (!sp || !sp.value || !global.ChamLinguistics) return;
        sp.value = global.ChamLinguistics.normalizeChamLogicalClient(sp.value);
        if (global.StudioApp) global.StudioApp.toast("Normalized to canonical Brahmic Logical Order");
    }

    function initDrag() {
        if (state.dragInit) return;
        const header = document.getElementById("cham-kb-header");
        const panel = document.getElementById("cham-kb-panel");
        if (!header || !panel) return;
        state.dragInit = true;

        let dragging = false;
        let sx = 0, sy = 0, initL = 0, initT = 0;

        header.addEventListener("mousedown", (e) => {
            if (e.target.closest("select, button, input")) return;
            dragging = true;
            const r = panel.getBoundingClientRect();
            sx = e.clientX;
            sy = e.clientY;
            initL = r.left;
            initT = r.top;
            panel.style.bottom = "auto";
            panel.style.right = "auto";
            panel.style.left = `${r.left}px`;
            panel.style.top = `${r.top}px`;
            document.body.style.userSelect = "none";
        });

        document.addEventListener("mousemove", (e) => {
            if (!dragging) return;
            const maxL = Math.max(8, window.innerWidth - panel.offsetWidth - 8);
            const maxT = Math.max(8, window.innerHeight - panel.offsetHeight - 8);
            const nl = Math.max(8, Math.min(maxL, initL + (e.clientX - sx)));
            const nt = Math.max(8, Math.min(maxT, initT + (e.clientY - sy)));
            panel.style.left = `${nl}px`;
            panel.style.top = `${nt}px`;
        });

        document.addEventListener("mouseup", () => {
            if (dragging) {
                dragging = false;
                document.body.style.userSelect = "";
            }
        });
    }

    global.ChamKeyboard = {
        toggle,
        isOpen: () => state.isOpen,
        setRowLayout,
        setPhoneticMode,
        setFontSize,
        resetPosition,
        copyScratchpad,
        insertScratchpadToLine,
        clearScratchpad,
        normalizeScratchpad,
        renderGrid
    };
})(window);
