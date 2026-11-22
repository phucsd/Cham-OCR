/**
 * Cham OCR Studio — Interactive User Guide & Feature Manual Controller (/help)
 * Provides ScrollSpy TOC, Reading Progress Bar, Mobile TOC Drawer, and Live Guide Search.
 */
(function () {
    "use strict";

    window.toggleHelpMobileToc = function () {
        const sb = document.getElementById("help-sidebar");
        if (sb) sb.classList.toggle("mobile-open");
    };

    window.filterHelpGuide = function (rawQuery) {
        const q = String(rawQuery || "").trim().toLowerCase();

        // Filter TOC items
        document.querySelectorAll("#help-sidebar .toc-nav-item").forEach((li) => {
            if (!q) {
                li.style.display = "";
                return;
            }
            li.style.display = li.innerText.toLowerCase().includes(q) ? "" : "none";
        });

        // Filter shortcut rows if searching inside shortcuts
        document.querySelectorAll("#shortcuts-table tbody tr").forEach((row) => {
            if (!q) {
                row.style.display = "";
                return;
            }
            row.style.display = row.innerText.toLowerCase().includes(q) ? "" : "none";
        });
    };

    document.addEventListener("DOMContentLoaded", () => {
        const progressBar = document.getElementById("help-progress-bar");
        const links = Array.from(document.querySelectorAll(".toc-nav-link, .toc-sub-link"));
        const NAV_OFFSET = 82;
        const tracked = [];

        links.forEach((link) => {
            const href = link.getAttribute("href");
            if (href && href.startsWith("#")) {
                const target = document.querySelector(href);
                if (target) {
                    tracked.push({ link, target });
                    link.addEventListener("click", (e) => {
                        e.preventDefault();
                        const top = target.getBoundingClientRect().top + window.pageYOffset - NAV_OFFSET;
                        window.scrollTo({ top, behavior: "smooth" });
                        history.pushState(null, "", href);
                        document.getElementById("help-sidebar")?.classList.remove("mobile-open");
                    });
                }
            }
        });

        function onScroll() {
            const scrollTop = window.scrollY || document.documentElement.scrollTop;
            const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
            if (progressBar) {
                progressBar.style.width = `${Math.min(100, Math.max(0, (scrollTop / maxScroll) * 100))}%`;
            }

            const probe = scrollTop + 120;
            for (let i = tracked.length - 1; i >= 0; i--) {
                const top = tracked[i].target.getBoundingClientRect().top + window.pageYOffset;
                if (top <= probe) {
                    links.forEach((l) => l.classList.remove("active"));
                    tracked[i].link.classList.add("active");
                    break;
                }
            }
        }

        window.addEventListener("scroll", onScroll, { passive: true });
        onScroll();

        // Allow '/' or Ctrl+K on /help to focus the guide search input
        document.addEventListener("keydown", (e) => {
            if (((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) || (e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName || ""))) {
                e.preventDefault();
                const inp = document.getElementById("help-search-input");
                if (inp) {
                    inp.focus();
                    inp.select();
                }
            }
        });
    });
})();
