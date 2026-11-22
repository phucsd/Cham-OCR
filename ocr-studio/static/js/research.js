/**
 * Cham-OCR Technical Report — Interactive Navigation, Reading Progress & Citation Helper
 */
(function () {
    "use strict";

    function showPaperToast(msg) {
        const el = document.createElement("div");
        el.className = "paper-toast";
        el.innerText = msg;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 2200);
    }

    window.copyBibtexCitation = function () {
        const codeEl = document.getElementById("bibtex-code");
        if (!codeEl) return;
        navigator.clipboard.writeText(codeEl.innerText).then(() => {
            showPaperToast("Copied BibTeX citation to clipboard");
        });
    };

    window.toggleMobileToc = function () {
        const sb = document.getElementById("paper-sidebar");
        if (sb) sb.classList.toggle("mobile-open");
    };

    window.filterTocSections = function (query) {
        const q = String(query || "").trim().toLowerCase();
        document.querySelectorAll("#paper-sidebar .toc-item").forEach((li) => {
            if (!q) {
                li.style.display = "";
                return;
            }
            const text = li.innerText.toLowerCase();
            li.style.display = text.includes(q) ? "" : "none";
        });
    };

    function initResearchPage() {
        const progressBar = document.getElementById("reading-progress-bar");
        const links = Array.from(document.querySelectorAll(".toc-link, .toc-sublink"));
        const NAV_OFFSET = 84;
        const sections = [];

        links.forEach((link) => {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                const target = document.querySelector(href);
                if (target) {
                    sections.push({ link, target });
                    link.addEventListener("click", (e) => {
                        e.preventDefault();
                        const top = target.getBoundingClientRect().top + window.pageYOffset - NAV_OFFSET;
                        window.scrollTo({ top, behavior: "smooth" });
                        history.pushState(null, "", href);
                        document.getElementById("paper-sidebar")?.classList.remove("mobile-open");
                    });
                }
            }
        });

        function updateScrollState() {
            const scrollTop = Math.max(
                window.scrollY || 0,
                document.documentElement ? document.documentElement.scrollTop : 0,
                document.body ? document.body.scrollTop : 0
            );
            const totalScrollable = Math.max(
                1,
                (document.documentElement ? document.documentElement.scrollHeight : document.body.scrollHeight) - window.innerHeight
            );
            if (progressBar) {
                const pct = Math.min(100, Math.max(2, (scrollTop / totalScrollable) * 100));
                progressBar.style.width = `${pct.toFixed(2)}%`;
            }

            const probe = scrollTop + 125;
            for (let i = sections.length - 1; i >= 0; i--) {
                const top = sections[i].target.getBoundingClientRect().top + window.pageYOffset;
                if (top <= probe) {
                    links.forEach((l) => l.classList.remove("active"));
                    sections[i].link.classList.add("active");
                    break;
                }
            }
        }

        window.addEventListener("scroll", updateScrollState, { passive: true });
        document.addEventListener("scroll", updateScrollState, { passive: true });
        window.addEventListener("resize", updateScrollState, { passive: true });
        updateScrollState();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initResearchPage);
    } else {
        initResearchPage();
    }
})();
