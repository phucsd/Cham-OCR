/**
 * Cham OCR Studio — Paleographic Transcription & Diagnostic Workbench Engine
 * Pure Academic English | Preserves Claude Warm Light Design System (DESIGN.md)
 */
(function (global) {
    "use strict";

    // Global client-side diagnostic error hooks
    window.onerror = function (message, source, lineno, colno) {
        const msg = `${message} (${source}:${lineno}:${colno})`;
        appendLog(`Client Runtime Error: ${msg}`, "error");
        fetch("/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: msg })
        }).catch(() => {});
        return false;
    };

    window.onunhandledrejection = function (event) {
        const reason = event && event.reason ? String(event.reason) : "Unknown rejection";
        appendLog(`Unhandled Promise Rejection: ${reason}`, "error");
        fetch("/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "Unhandled rejection: " + reason })
        }).catch(() => {});
    };

    // =========================================================================
    // Application State
    // =========================================================================
    const state = {
        layoutMode: localStorage.getItem("cham_studio_layout") || "workbench", // workbench | side | stacked | focus
        sidebarWidth: parseInt(localStorage.getItem("cham_studio_sb_w") || "288", 10),
        editorWidth: parseInt(localStorage.getItem("cham_studio_ed_w") || "470", 10),
        stackedViewerPct: parseFloat(localStorage.getItem("cham_studio_stack_pct") || "50"),

        // Queue & Active Folio
        queue: [],
        queueFilter: "all", // all | needs_review | ocr_done | verified | unprocessed
        queueSearch: "",
        autoTranscribeOnUpload: localStorage.getItem("cham_studio_auto_ocr") === "1",
        batchRunning: false,
        batchAbortRequested: false,

        // Active Folio Image & OCR Data
        activeFolioId: null,
        currentImageBase64: "",
        imageWidth: 0,
        imageHeight: 0,
        lines: [],
        paragraphs: [],
        documentMeta: null,
        profilerMeta: null,
        modelInfoMeta: null,

        // Undo / Redo Stacks (per active folio in memory)
        undoStack: [],
        redoStack: [],

        // Engine Configuration
        segMethod: "dbnet",      // dbnet | valley | adaptive
        readingOrder: "auto",    // auto | rows
        valleyGap: 12,
        valleyWindow: 25,

        // Viewer State
        sidebarUserPinned: null, // null = smart auto (collapsed for <=1 folio, expanded for >=2)
        scale: 1.0,
        panX: 0,
        panY: 0,
        isPanning: false,
        panStartX: 0,
        panStartY: 0,
        bboxMode: localStorage.getItem("cham_studio_bbox_mode") || "spotlight", // spotlight | all | off
        showBboxOverlay: true,
        showPolygonOverlay: true,
        isDrawBboxMode: false,
        isDrawingBbox: false,
        drawStartPt: null,
        resizingBbox: null, // { lineIdx, corner, startX, startY, origBox }
        magnifierEnabled: false,
        filters: {
            contrast: 100,
            brightness: 100,
            grayscale: 0,
            invert: false
        },

        // Editor & Linguistic State
        editorMode: "lines",     // lines | paragraphs
        cropMode: localStorage.getItem("cham_studio_crop_mode") || "focus", // focus | all | off
        showLineCrops: false,
        showRomanized: localStorage.getItem("cham_studio_roman") === "1",
        showInspector: false,    // Closed by default (Progressive Disclosure — toggle via U+ / Alt+U)
        findBarOpen: false,
        searchQuery: "",
        editingLineIdx: null,
        activeInputEl: null,
        selection: {
            level: "line",       // line | char
            lineIdx: null,
            charIdx: null
        },

        // Cached Samples & Pending PDF
        samples: [],
        pendingPdfDoc: null,
        pendingPdfFileName: "",
        cmdkSelectedIndex: 0,
        cmdkFilteredItems: []
    };

    // =========================================================================
    // IndexedDB Persistence for Large Manuscript Folios
    // =========================================================================
    const DB_NAME = "ChamOCRStudioDB";
    const STORE_NAME = "images";

    function openDB() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, 1);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME, { keyPath: "id" });
                }
            };
            req.onsuccess = (e) => resolve(e.target.result);
            req.onerror = (e) => reject(e.target.error);
        });
    }

    async function dbPutFolio(item) {
        try {
            const db = await openDB();
            const tx = db.transaction(STORE_NAME, "readwrite");
            tx.objectStore(STORE_NAME).put({ ...item });
            return new Promise((res, rej) => {
                tx.oncomplete = () => res();
                tx.onerror = () => rej(tx.error);
            });
        } catch (err) {
            console.warn("IndexedDB save warning:", err);
        }
    }

    async function dbGetFolio(id) {
        try {
            const db = await openDB();
            const tx = db.transaction(STORE_NAME, "readonly");
            const req = tx.objectStore(STORE_NAME).get(id);
            return new Promise((res, rej) => {
                req.onsuccess = () => res(req.result || null);
                req.onerror = () => rej(req.error);
            });
        } catch (err) {
            console.warn("IndexedDB load warning:", err);
            return null;
        }
    }

    async function dbDeleteFolio(id) {
        try {
            const db = await openDB();
            const tx = db.transaction(STORE_NAME, "readwrite");
            tx.objectStore(STORE_NAME).delete(id);
        } catch (err) {
            console.warn("IndexedDB delete warning:", err);
        }
    }

    async function dbClearAll() {
        try {
            const db = await openDB();
            const tx = db.transaction(STORE_NAME, "readwrite");
            tx.objectStore(STORE_NAME).clear();
        } catch (err) {
            console.warn("IndexedDB clear warning:", err);
        }
    }

    function persistQueueIndex() {
        try {
            const slim = state.queue.map((item) => ({
                id: item.id,
                name: item.name,
                thumb: item.thumb || "",
                time: item.time || "",
                model: item.model || "v24_1_r4",
                status: item.status || "Unprocessed",
                lineCount: Array.isArray(item.lines) ? item.lines.length : (item.lineCount || 0),
                verifiedCount: Array.isArray(item.lines)
                    ? item.lines.filter((l) => l.human_verified).length
                    : (item.verifiedCount || 0),
                document: item.document || null,
                profiler: item.profiler || null,
                active: item.id === state.activeFolioId
            }));
            localStorage.setItem("cham_ocr_image_queue", JSON.stringify(slim));
        } catch (err) {
            console.warn("Queue localStorage save warning:", err);
        }
    }

    async function restoreQueueFromStorage() {
        try {
            const raw = localStorage.getItem("cham_ocr_image_queue");
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (!Array.isArray(parsed)) return;
            state.queue = parsed.map((x) => ({
                ...x,
                lines: Array.isArray(x.lines) ? x.lines : null,
                active: false
            }));
            renderQueue();
            updateCorpusFooter();
            if (state.queue.length > 0) {
                await selectFolio(state.queue[0].id);
            }
        } catch (err) {
            console.warn("Queue restore warning:", err);
            state.queue = [];
        }
    }

    // =========================================================================
    // Document Status Calculation (Strictly aligned with DESIGN.md)
    // =========================================================================
    function computeDocStatus(lines, docInfo) {
        if (!lines || lines.length === 0) {
            return docInfo && docInfo.status === "needs_review" ? "Needs review" : "No text";
        }
        if (lines.every((l) => l.human_verified === true)) {
            return "Verified";
        }
        const anyLineFlagged = lines.some(
            (l) => l.human_verified !== true && l.segmentation && l.segmentation.needs_review
        );
        if (anyLineFlagged || (docInfo && docInfo.needs_review)) {
            return "Needs review";
        }
        return "OCR done";
    }

    function statusToSlug(statusStr) {
        const s = String(statusStr || "unprocessed").toLowerCase().trim();
        if (s === "ocr done") return "ocr_done";
        if (s === "needs review") return "needs_review";
        if (s === "verified") return "verified";
        if (s === "no text") return "no_text";
        if (s === "processing") return "processing";
        return "unprocessed";
    }

    // =========================================================================
    // Layout Modes & Resizable Splitters
    // =========================================================================
    function applyLayoutVars() {
        document.documentElement.style.setProperty("--sidebar-width", `${state.sidebarWidth}px`);
        document.documentElement.style.setProperty("--editor-width", `${state.editorWidth}px`);
        document.documentElement.style.setProperty("--stacked-viewer-height", `${state.stackedViewerPct}%`);
    }

    function updateWorkspaceChromeState() {
        const ws = document.getElementById("main-workspace");
        const sb = document.getElementById("sidebar-pane");
        const topPill = document.getElementById("canvas-top-pill");
        const floatHud = document.getElementById("canvas-floating-hud");
        const qCountBadge = document.getElementById("canvas-queue-count");
        const qBtn = document.getElementById("btn-canvas-queue");
        const postOcrTools = document.getElementById("editor-post-ocr-tools");
        const actionBar = document.getElementById("line-action-bar");
        const inspectorDrawer = document.getElementById("inspector-drawer");

        const isEmptyLanding = state.queue.length === 0 && !state.currentImageBase64;
        if (ws) ws.classList.toggle("is-empty-landing", isEmptyLanding);

        // Smart Auto-Collapsing Corpus Queue (<= 1 folio = collapsed, >= 2 folios = expanded unless user pinned)
        if (sb) {
            if (state.layoutMode === "side" || state.layoutMode === "focus" || isEmptyLanding) {
                sb.classList.add("collapsed");
            } else if (state.sidebarUserPinned !== null) {
                sb.classList.toggle("collapsed", !state.sidebarUserPinned);
            } else {
                sb.classList.toggle("collapsed", state.queue.length <= 1);
            }
        }

        if (qCountBadge) qCountBadge.innerText = String(state.queue.length);
        if (qBtn && sb) qBtn.classList.toggle("active", !sb.classList.contains("collapsed"));

        const hasImage = Boolean(state.currentImageBase64);
        if (topPill) topPill.style.display = hasImage ? "inline-flex" : "none";
        if (floatHud) floatHud.style.display = hasImage ? "inline-flex" : "none";

        const hasLines = Array.isArray(state.lines) && state.lines.length > 0;
        if (postOcrTools) postOcrTools.style.display = hasLines ? "inline-flex" : "none";
        if (actionBar) actionBar.style.display = hasLines ? "flex" : "none";
        if (!hasLines && inspectorDrawer) {
            inspectorDrawer.classList.remove("open");
        } else if (hasLines && inspectorDrawer) {
            inspectorDrawer.classList.toggle("open", state.showInspector);
        }
    }

    function setLayoutMode(mode) {
        const valid = ["workbench", "side", "stacked", "focus"];
        if (!valid.includes(mode)) mode = "workbench";
        state.layoutMode = mode;
        localStorage.setItem("cham_studio_layout", mode);

        const ws = document.getElementById("main-workspace");
        if (ws) {
            ws.classList.remove("layout-workbench", "layout-side", "layout-stacked", "layout-focus");
            ws.classList.add(`layout-${mode}`);
        }
        if (mode === "workbench" && state.queue.length >= 2 && state.sidebarUserPinned === false) {
            state.sidebarUserPinned = null;
        }
        updateWorkspaceChromeState();

        const labels = {
            workbench: "3-Pane",
            side: "Compare",
            stacked: "Stacked",
            focus: "Reader"
        };
        const activeLbl = document.getElementById("active-layout-label");
        if (activeLbl) activeLbl.innerText = labels[mode] || "Layout";

        document.querySelectorAll(".view-btn, .layout-opt").forEach((b) => {
            b.classList.toggle("active", b.dataset.view === mode);
        });
        closeAllPopovers();

        setTimeout(() => {
            if (mode !== "focus" && state.currentImageBase64) zoomToFit();
        }, 60);
    }

    function toggleSidebarPane() {
        const sb = document.getElementById("sidebar-pane");
        if (!sb) return;
        if (state.layoutMode === "side") {
            state.sidebarUserPinned = true;
            setLayoutMode("workbench");
            return;
        }
        const willExpand = sb.classList.contains("collapsed");
        state.sidebarUserPinned = willExpand;
        updateWorkspaceChromeState();
        setTimeout(() => {
            if (state.currentImageBase64) zoomToFit();
        }, 60);
    }

    function initSplitResizers() {
        const sbResizer = document.getElementById("resizer-sidebar");
        const edResizerV = document.getElementById("resizer-editor-v");
        const edResizerH = document.getElementById("resizer-editor-h");
        const stage = document.getElementById("main-split-stage");

        if (sbResizer) {
            let active = false;
            sbResizer.addEventListener("mousedown", (e) => {
                active = true;
                sbResizer.classList.add("dragging");
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";
                e.preventDefault();
            });
            sbResizer.addEventListener("dblclick", () => {
                state.sidebarWidth = 288;
                localStorage.setItem("cham_studio_sb_w", "288");
                applyLayoutVars();
            });
            document.addEventListener("mousemove", (e) => {
                if (!active) return;
                const w = Math.max(220, Math.min(450, e.clientX));
                state.sidebarWidth = w;
                applyLayoutVars();
            });
            document.addEventListener("mouseup", () => {
                if (active) {
                    active = false;
                    sbResizer.classList.remove("dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    localStorage.setItem("cham_studio_sb_w", String(state.sidebarWidth));
                }
            });
        }

        if (edResizerV) {
            let active = false;
            edResizerV.addEventListener("mousedown", (e) => {
                active = true;
                edResizerV.classList.add("dragging");
                document.body.style.cursor = "col-resize";
                document.body.style.userSelect = "none";
                e.preventDefault();
            });
            edResizerV.addEventListener("dblclick", () => {
                state.editorWidth = 470;
                localStorage.setItem("cham_studio_ed_w", "470");
                applyLayoutVars();
            });
            document.addEventListener("mousemove", (e) => {
                if (!active) return;
                const w = Math.max(320, Math.min(Math.round(window.innerWidth * 0.62), window.innerWidth - e.clientX));
                state.editorWidth = w;
                applyLayoutVars();
            });
            document.addEventListener("mouseup", () => {
                if (active) {
                    active = false;
                    edResizerV.classList.remove("dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    localStorage.setItem("cham_studio_ed_w", String(state.editorWidth));
                }
            });
        }

        if (edResizerH && stage) {
            let active = false;
            edResizerH.addEventListener("mousedown", (e) => {
                active = true;
                edResizerH.classList.add("dragging");
                document.body.style.cursor = "row-resize";
                document.body.style.userSelect = "none";
                e.preventDefault();
            });
            edResizerH.addEventListener("dblclick", () => {
                state.stackedViewerPct = 50;
                localStorage.setItem("cham_studio_stack_pct", "50");
                applyLayoutVars();
            });
            document.addEventListener("mousemove", (e) => {
                if (!active) return;
                const rect = stage.getBoundingClientRect();
                const pct = ((e.clientY - rect.top) / Math.max(1, rect.height)) * 100;
                state.stackedViewerPct = Math.max(24, Math.min(76, pct));
                applyLayoutVars();
            });
            document.addEventListener("mouseup", () => {
                if (active) {
                    active = false;
                    edResizerH.classList.remove("dragging");
                    document.body.style.cursor = "";
                    document.body.style.userSelect = "";
                    localStorage.setItem("cham_studio_stack_pct", String(Math.round(state.stackedViewerPct)));
                }
            });
        }
    }

    // =========================================================================
    // Multi-File Image, Folder & Multi-Page PDF Intake
    // =========================================================================
    function generateThumbnail(base64Str, maxW = 132, maxH = 132) {
        return new Promise((resolve) => {
            const im = new Image();
            im.onload = () => {
                const c = document.createElement("canvas");
                let w = im.width, h = im.height;
                if (w > maxW || h > maxH) {
                    const r = Math.min(maxW / w, maxH / h);
                    w = Math.max(1, Math.round(w * r));
                    h = Math.max(1, Math.round(h * r));
                }
                c.width = w;
                c.height = h;
                c.getContext("2d").drawImage(im, 0, 0, w, h);
                resolve(c.toDataURL("image/jpeg", 0.68).split(",")[1]);
            };
            im.onerror = () => resolve("");
            im.src = "data:image/png;base64," + base64Str;
        });
    }

    function readFileAsBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                const res = String(ev.target.result || "");
                resolve(res.includes(",") ? res.split(",")[1] : res);
            };
            reader.onerror = (err) => reject(err);
            reader.readAsDataURL(file);
        });
    }

    async function enqueueBase64Folio(name, b64Data, options = {}) {
        const thumb = await generateThumbnail(b64Data);
        const id = "folio_" + Date.now() + "_" + Math.random().toString(36).slice(2, 7);
        const modelSel = document.getElementById("model-select");
        const folio = {
            id,
            name: name || `clipboard_manuscript_${Date.now()}.png`,
            image: b64Data,
            thumb,
            time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
            model: options.model || (modelSel ? modelSel.value : "v24_1_r4"),
            status: "Unprocessed",
            lines: [],
            lineCount: 0,
            verifiedCount: 0,
            document: null,
            profiler: null
        };

        state.queue.push(folio);
        await dbPutFolio(folio);
        persistQueueIndex();
        renderQueue();
        updateCorpusFooter();

        if (options.makeActive !== false) {
            await selectFolio(folio.id);
        }
        return folio;
    }

    function readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (ev) => resolve(String(ev.target.result || ""));
            reader.onerror = (err) => reject(err);
            reader.readAsText(file, "utf-8");
        });
    }

    function stemName(filename) {
        return String(filename || "")
            .replace(/^.*[\\/]/, "")
            .replace(/\.studio\.json$/i, "")
            .replace(/\.[^.]+$/, "")
            .trim()
            .toLowerCase();
    }

    async function applyImportedJsonPayload(payload) {
        const entries = Array.isArray(payload && payload.documents) ? payload.documents : [payload];
        let hydratedCount = 0;
        for (const entry of entries) {
            if (!entry || !Array.isArray(entry.lines)) continue;
            const targetStem = stemName(entry.doc_id || entry.image_name || entry.name || "");
            let folio = state.queue.find((q) => stemName(q.name) === targetStem);
            if (!folio && entries.length === 1 && state.activeFolioId) {
                folio = state.queue.find((q) => q.id === state.activeFolioId);
            }
            if (!folio) continue;
            folio.lines = entry.lines.map((ln, idx) => ({
                ...ln,
                index: ln.index || ln.line_id || idx + 1,
                text: ln.text ?? ln.draft_prediction ?? "",
                confidence: Number(ln.confidence ?? ln.draft_confidence ?? 0),
                calibrated_confidence: Number(ln.calibrated_confidence ?? ln.draft_calibrated_confidence ?? ln.confidence ?? 0),
                human_verified: Boolean(ln.human_verified)
            }));
            if (entry.document) folio.document = entry.document;
            if (entry.profiler) folio.profiler = entry.profiler;
            folio.lineCount = folio.lines.length;
            folio.verifiedCount = folio.lines.filter((l) => l.human_verified).length;
            folio.model = entry.model || folio.model;
            folio.status = computeDocStatus(folio.lines, folio.document);
            if (folio.id === state.activeFolioId) {
                state.lines = folio.lines;
                state.documentMeta = folio.document || null;
                state.profilerMeta = folio.profiler || null;
            }
            await dbPutFolio(folio);
            hydratedCount++;
            if (folio.id === state.activeFolioId) {
                await selectFolio(folio.id, { skipSyncActive: true });
            }
        }
        if (hydratedCount > 0) {
            persistQueueIndex();
            renderQueue();
            updateCorpusFooter();
        }
        return hydratedCount;
    }

    async function handleIncomingFiles(fileList) {
        if (!fileList || fileList.length === 0) return;
        const files = Array.from(fileList);
        const pdfFiles = files.filter((f) => (f.type === "application/pdf") || (f.name && f.name.toLowerCase().endsWith(".pdf")));
        const jsonFiles = files.filter((f) => (f.type === "application/json") || /\.json$/i.test(f.name || ""));
        const imgFiles = files.filter((f) => (f.type && f.type.startsWith("image/")) || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name || ""));

        const newlyAdded = [];
        for (let i = 0; i < imgFiles.length; i++) {
            const f = imgFiles[i];
            const safeName = f.name || `clipboard_image_${Date.now()}_${i + 1}.png`;
            try {
                const b64 = await readFileAsBase64(f);
                const item = await enqueueBase64Folio(safeName, b64, { makeActive: i === 0 && !state.activeFolioId });
                newlyAdded.push(item);
                appendLog(`Queued manuscript image: ${safeName}`, "info");
            } catch (err) {
                appendLog(`Failed to read ${safeName}: ${err}`, "error");
            }
        }

        let hydratedFolios = 0;
        for (const jf of jsonFiles) {
            try {
                const rawTxt = await readFileAsText(jf);
                const parsed = JSON.parse(rawTxt);
                hydratedFolios += await applyImportedJsonPayload(parsed);
                appendLog(`Imported transcription JSON: ${jf.name}`, "success");
            } catch (err) {
                toast(`JSON import error (${jf.name}): ${err.message}`);
                appendLog(`Failed to parse JSON ${jf.name}: ${err}`, "error");
            }
        }

        if (newlyAdded.length > 0 && (!state.activeFolioId || newlyAdded.length === 1)) {
            await selectFolio(newlyAdded[0].id, { skipSyncActive: hydratedFolios > 0 });
        }
        if (jsonFiles.length > 0 && hydratedFolios === 0) {
            toast("No matching folio found in queue for imported JSON");
        } else if (hydratedFolios > 0) {
            toast(`Hydrated ${hydratedFolios} folio(s) from Review Queue JSON`);
        } else if (newlyAdded.length > 1) {
            toast(`Added ${newlyAdded.length} manuscript folios to queue`);
        } else if (newlyAdded.length === 1) {
            toast(`Loaded ${newlyAdded[0].name}`);
        }

        const hasUnprocessedNew = newlyAdded.some((it) => it.status === "Unprocessed" && (!it.lines || it.lines.length === 0));
        if (pdfFiles.length > 0) {
            await openPdfImportDialog(pdfFiles[0]);
        } else if (state.autoTranscribeOnUpload && hasUnprocessedNew) {
            runBatchOCR("unprocessed");
        }
    }

    // =========================================================================
    // Multi-Page PDF Rasterizer (Client-side PDF.js)
    // =========================================================================
    let pdfJsPromise = null;
    function ensurePdfJsLoaded() {
        if (window.pdfjsLib) return Promise.resolve(window.pdfjsLib);
        if (pdfJsPromise) return pdfJsPromise;
        pdfJsPromise = new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
            script.onload = () => {
                if (window.pdfjsLib) {
                    window.pdfjsLib.GlobalWorkerOptions.workerSrc =
                        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
                    resolve(window.pdfjsLib);
                } else {
                    reject(new Error("PDF.js failed to initialize"));
                }
            };
            script.onerror = () => {
                pdfJsPromise = null;
                reject(new Error("Unable to load PDF.js library from CDN"));
            };
            document.head.appendChild(script);
        });
        return pdfJsPromise;
    }

    async function openPdfImportDialog(pdfFile) {
        toast("Inspecting PDF document...");
        try {
            const pdfjsLib = await ensurePdfJsLoaded();
            const arrayBuf = await pdfFile.arrayBuffer();
            const pdfDoc = await pdfjsLib.getDocument({ data: arrayBuf }).promise;
            state.pendingPdfDoc = pdfDoc;
            state.pendingPdfFileName = (pdfFile.name || "manuscript.pdf").replace(/\.pdf$/i, "");

            const infoEl = document.getElementById("pdf-modal-info");
            const rangeEl = document.getElementById("pdf-page-range");
            const statusEl = document.getElementById("pdf-import-status");
            if (infoEl) {
                infoEl.innerHTML = `<strong>${pdfFile.name}</strong> — <strong>${pdfDoc.numPages}</strong> page(s) detected.`;
            }
            if (rangeEl) {
                rangeEl.value = pdfDoc.numPages <= 15 ? `1-${pdfDoc.numPages}` : "1-10";
                rangeEl.placeholder = `e.g. 1-${pdfDoc.numPages} or 1,3,5-8`;
            }
            if (statusEl) statusEl.innerText = "";
            openModal("pdf-modal");
        } catch (err) {
            toast("PDF import error: " + err.message);
            appendLog("PDF import error: " + err, "error");
        }
    }

    function parsePageRange(spec, maxPages) {
        const clean = String(spec || "").trim().toLowerCase();
        if (!clean || clean === "all" || clean === "*") {
            return Array.from({ length: maxPages }, (_, i) => i + 1);
        }
        const pages = new Set();
        clean.split(",").forEach((part) => {
            const seg = part.trim();
            if (!seg) return;
            if (seg.includes("-")) {
                const [aStr, bStr] = seg.split("-");
                const a = Math.max(1, parseInt(aStr, 10) || 1);
                const b = Math.min(maxPages, parseInt(bStr, 10) || maxPages);
                for (let p = Math.min(a, b); p <= Math.max(a, b); p++) pages.add(p);
            } else {
                const p = parseInt(seg, 10);
                if (p >= 1 && p <= maxPages) pages.add(p);
            }
        });
        return Array.from(pages).sort((a, b) => a - b);
    }

    async function confirmPdfImport() {
        if (!state.pendingPdfDoc) return;
        const pdfDoc = state.pendingPdfDoc;
        const rangeInput = document.getElementById("pdf-page-range");
        const scaleInput = document.getElementById("pdf-dpi-select");
        const statusEl = document.getElementById("pdf-import-status");
        const importBtn = document.getElementById("btn-confirm-pdf");

        const pages = parsePageRange(rangeInput ? rangeInput.value : "all", pdfDoc.numPages);
        if (pages.length === 0) {
            toast("Please enter a valid page range");
            return;
        }

        const renderScale = parseFloat(scaleInput ? scaleInput.value : "2.0") || 2.0;
        if (importBtn) importBtn.disabled = true;

        const imported = [];
        try {
            for (let i = 0; i < pages.length; i++) {
                const pageNum = pages[i];
                if (statusEl) statusEl.innerText = `Rasterizing page ${pageNum} (${i + 1} of ${pages.length})...`;
                const page = await pdfDoc.getPage(pageNum);
                const viewport = page.getViewport({ scale: renderScale });
                const canvas = document.createElement("canvas");
                canvas.width = Math.round(viewport.width);
                canvas.height = Math.round(viewport.height);
                const ctx = canvas.getContext("2d");
                await page.render({ canvasContext: ctx, viewport }).promise;
                const b64 = canvas.toDataURL("image/png").split(",")[1];
                const folioName = `${state.pendingPdfFileName}_p${String(pageNum).padStart(3, "0")}.png`;
                const item = await enqueueBase64Folio(folioName, b64, { makeActive: i === 0 });
                imported.push(item);
            }
            closeModal("pdf-modal");
            toast(`Imported ${imported.length} PDF page(s) into Corpus Queue`);
            appendLog(`Imported ${imported.length} pages from ${state.pendingPdfFileName}.pdf`, "success");
            if (state.autoTranscribeOnUpload && imported.length > 0) {
                runBatchOCR("unprocessed");
            }
        } catch (err) {
            toast("Error rendering PDF page: " + err.message);
        } finally {
            if (importBtn) importBtn.disabled = false;
            state.pendingPdfDoc = null;
        }
    }

    // =========================================================================
    // Built-in Demo Manuscript Specimens (/api/samples)
    // =========================================================================
    async function fetchCuratedSamples() {
        try {
            const res = await fetch("/api/samples");
            if (!res.ok) return;
            const data = await res.json();
            state.samples = data.samples || [];
            renderQuickEmptySamples();
            renderSamplesModalGrid();
        } catch (err) {
            console.warn("Samples fetch skipped:", err);
        }
    }

    function renderQuickEmptySamples() {
        const grid = document.getElementById("quick-samples-grid");
        const wrap = document.getElementById("quick-samples-strip");
        if (!grid || !wrap) return;
        if (!state.samples.length) {
            wrap.style.display = "none";
            return;
        }
        wrap.style.display = "flex";
        grid.innerHTML = "";
        state.samples.slice(0, 3).forEach((sp) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "quick-sample-item";
            btn.onclick = () => loadSampleById(sp.id, true);
            btn.innerHTML = `
                <div class="quick-sample-thumb">
                    ${sp.thumb ? `<img src="data:image/jpeg;base64,${sp.thumb}" alt="${escapeHtml(sp.name)}">` : ""}
                </div>
                <span class="quick-sample-cat">${escapeHtml(sp.category)}</span>
                <span class="quick-sample-label">${escapeHtml(sp.name)}</span>
            `;
            grid.appendChild(btn);
        });
    }

    function renderSamplesModalGrid() {
        const grid = document.getElementById("samples-modal-grid");
        if (!grid) return;
        if (!state.samples.length) {
            grid.innerHTML = `<div style="color:var(--text-tertiary);font-size:13px;">No local benchmark specimens detected on disk.</div>`;
            return;
        }
        grid.innerHTML = "";
        state.samples.forEach((sp) => {
            const card = document.createElement("div");
            card.className = "sample-gallery-card";
            card.onclick = () => {
                closeModal("samples-modal");
                loadSampleById(sp.id, true);
            };
            card.innerHTML = `
                <div class="sample-gallery-thumb">
                    ${sp.thumb ? `<img src="data:image/jpeg;base64,${sp.thumb}" alt="${escapeHtml(sp.name)}">` : ""}
                </div>
                <div style="display:flex;flex-direction:column;gap:4px;min-width:0;">
                    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                        <span class="count-pill">${escapeHtml(sp.category)}</span>
                        <span class="micro-tag">Model: ${escapeHtml(sp.recommended_model)}</span>
                    </div>
                    <div style="font-size:13px;font-weight:700;color:var(--text-primary);">${escapeHtml(sp.name)}</div>
                    <div style="font-size:11.5px;color:var(--text-secondary);line-height:1.4;">${escapeHtml(sp.description)}</div>
                </div>
            `;
            grid.appendChild(card);
        });
    }

    async function loadSampleById(sampleId, autoSelectModel = true) {
        toast("Loading archival manuscript specimen...");
        try {
            const res = await fetch(`/api/sample/${encodeURIComponent(sampleId)}`);
            const data = await res.json();
            if (data.error || !data.image) {
                toast(data.error || "Specimen unavailable");
                return null;
            }
            if (autoSelectModel && data.recommended_model) {
                const sel = document.getElementById("model-select");
                if (sel && Array.from(sel.options).some((o) => o.value === data.recommended_model)) {
                    sel.value = data.recommended_model;
                }
            }
            const item = await enqueueBase64Folio(data.filename || `${sampleId}.png`, data.image, {
                model: data.recommended_model || "v24_1_r4",
                makeActive: true
            });
            toast(`Loaded specimen: ${data.name}`);
            appendLog(`Loaded curated specimen: ${data.name} (${data.width}×${data.height}px)`, "info");
            if (state.autoTranscribeOnUpload) {
                await runSingleFolioOCR(item.id);
            }
            return item;
        } catch (err) {
            toast("Failed to load specimen");
            return null;
        }
    }

    async function loadDemoBatchSet() {
        closeModal("samples-modal");
        if (!state.samples.length) {
            toast("No curated specimens available");
            return;
        }
        const slice = state.samples.slice(0, 4);
        toast(`Loading ${slice.length}-folio demo batch...`);
        let firstId = null;
        for (let i = 0; i < slice.length; i++) {
            const res = await fetch(`/api/sample/${encodeURIComponent(slice[i].id)}`);
            const data = await res.json();
            if (data && data.image) {
                const folio = await enqueueBase64Folio(data.filename, data.image, {
                    model: data.recommended_model || "v24_1_r4",
                    makeActive: i === 0
                });
                if (i === 0) firstId = folio.id;
            }
        }
        if (firstId) {
            state.sidebarUserPinned = null;
            await selectFolio(firstId);
        }
        toast(`Loaded ${slice.length} demo folios — click "Batch" in Corpus Queue to transcribe all!`);
    }

    // =========================================================================
    // Queue Selection, Filtering, Drag-Reordering & Deletion
    // =========================================================================
    async function selectFolio(folioId, options = {}) {
        const item = state.queue.find((x) => x.id === folioId);
        if (!item) return;

        // Save current active folio lines back into queue item before switching (unless hydrating new state onto active folio)
        if (!options.skipSyncActive) {
            syncActiveFolioToQueue();
        }

        state.activeFolioId = folioId;
        state.queue.forEach((x) => { x.active = (x.id === folioId); });
        state.undoStack = [];
        state.redoStack = [];
        state.editingLineIdx = null;
        state.selection = { level: "line", lineIdx: null, charIdx: null };

        if (!item.image || !Array.isArray(item.lines)) {
            const stored = await dbGetFolio(folioId);
            if (stored) {
                if (stored.image) item.image = stored.image;
                if (Array.isArray(stored.lines)) item.lines = stored.lines;
                if (stored.document) item.document = stored.document;
                if (stored.profiler) item.profiler = stored.profiler;
            }
        }

        state.currentImageBase64 = item.image || "";
        state.lines = Array.isArray(item.lines) ? item.lines : [];
        if (state.lines.length > 0 && state.selection.lineIdx === null) {
            state.selection = { level: "line", lineIdx: 0, charIdx: null };
        }
        state.paragraphs = [];
        state.documentMeta = item.document || null;
        state.profilerMeta = item.profiler || null;

        updateWorkspaceChromeState();
        updateHeaderFolioBadge(item);
        renderQueue();

        const emptyCard = document.getElementById("empty-stage-card");
        const imgEl = document.getElementById("document-image");
        const runBtn = document.getElementById("btn-run");

        if (state.currentImageBase64) {
            if (emptyCard) emptyCard.style.display = "none";
            if (runBtn) runBtn.disabled = false;
            imgEl.onload = () => {
                state.imageWidth = imgEl.naturalWidth;
                state.imageHeight = imgEl.naturalHeight;
                applyCanvasFilters();
                zoomToFit();
                renderOverlays();
                renderEditor();
                renderDocTelemetry();
                if (state.editorMode === "paragraphs" && state.lines.length > 0) {
                    refreshParagraphFlow();
                }
            };
            imgEl.style.display = "block";
            imgEl.src = "data:image/png;base64," + state.currentImageBase64;
        } else {
            imgEl.style.display = "none";
            if (emptyCard) emptyCard.style.display = "flex";
            if (runBtn) runBtn.disabled = true;
            renderOverlays();
            renderEditor();
            renderDocTelemetry();
        }
    }

    function syncActiveFolioToQueue() {
        if (!state.activeFolioId) return;
        const item = state.queue.find((x) => x.id === state.activeFolioId);
        if (!item) return;
        item.lines = state.lines;
        item.lineCount = state.lines.length;
        item.verifiedCount = state.lines.filter((l) => l.human_verified).length;
        item.document = state.documentMeta;
        item.profiler = state.profilerMeta;
        if (item.status !== "Unprocessed" || state.lines.length > 0) {
            item.status = computeDocStatus(state.lines, state.documentMeta);
        }
        dbPutFolio(item);
        persistQueueIndex();
        updateHeaderFolioBadge(item);
        updateCorpusFooter();
    }

    function updateHeaderFolioBadge(item) {
        const chipEl = document.getElementById("viewer-folio-chip");
        const nameEl = document.getElementById("header-folio-name");
        const dotEl = document.getElementById("header-folio-dot");
        const statusBadge = document.getElementById("header-doc-badge");
        if (!item) {
            if (chipEl) chipEl.style.display = "none";
            if (nameEl) nameEl.innerText = "";
            if (dotEl) dotEl.className = "folio-dot";
            if (statusBadge) {
                statusBadge.style.display = "none";
                statusBadge.innerText = "";
            }
            return;
        }
        if (chipEl) chipEl.style.display = "inline-flex";
        if (nameEl) {
            nameEl.innerText = item.name;
            nameEl.title = item.name;
        }
        const slug = statusToSlug(item.status);
        if (dotEl) {
            dotEl.className = "folio-dot " + (slug === "needs_review" ? "review" : (slug === "unprocessed" ? "" : "ready"));
        }
        if (statusBadge) {
            statusBadge.style.display = "inline-flex";
            statusBadge.className = `doc-badge ${slug}`;
            statusBadge.innerText = item.status || "Unprocessed";
        }
    }

    async function removeFolio(folioId, ev) {
        if (ev) ev.stopPropagation();
        const idx = state.queue.findIndex((x) => x.id === folioId);
        if (idx === -1) return;
        const removed = state.queue[idx];
        state.queue.splice(idx, 1);
        await dbDeleteFolio(folioId);

        if (state.activeFolioId === folioId) {
            state.activeFolioId = null;
            if (state.queue.length > 0) {
                const nextIdx = Math.min(idx, state.queue.length - 1);
                await selectFolio(state.queue[nextIdx].id);
            } else {
                resetEmptyWorkspace();
            }
        }
        if (state.queue.length <= 1) state.sidebarUserPinned = null;
        updateWorkspaceChromeState();
        persistQueueIndex();
        renderQueue();
        updateCorpusFooter();
        toast(`Removed ${removed.name}`);
    }

    function clearEntireQueue() {
        if (!state.queue.length) return;
        if (!confirm(`Remove all ${state.queue.length} manuscript folio(s) from the workspace queue?`)) return;
        state.queue = [];
        state.activeFolioId = null;
        state.sidebarUserPinned = null;
        dbClearAll();
        persistQueueIndex();
        resetEmptyWorkspace();
        renderQueue();
        updateCorpusFooter();
        toast("Cleared manuscript corpus queue");
    }

    function resetEmptyWorkspace() {
        state.currentImageBase64 = "";
        state.lines = [];
        state.paragraphs = [];
        state.documentMeta = null;
        state.profilerMeta = null;
        const imgEl = document.getElementById("document-image");
        if (imgEl) {
            imgEl.src = "";
            imgEl.style.display = "none";
        }
        const emptyCard = document.getElementById("empty-stage-card");
        if (emptyCard) emptyCard.style.display = "flex";
        const runBtn = document.getElementById("btn-run");
        if (runBtn) runBtn.disabled = true;
        updateWorkspaceChromeState();
        updateHeaderFolioBadge(null);
        renderOverlays();
        renderEditor();
        renderDocTelemetry();
    }

    function setQueueFilter(filterVal) {
        state.queueFilter = filterVal;
        document.querySelectorAll(".filter-pill").forEach((b) => {
            b.classList.toggle("active", b.dataset.filter === filterVal);
        });
        renderQueue();
    }

    function setQueueSearch(q) {
        state.queueSearch = String(q || "").trim().toLowerCase();
        renderQueue();
    }

    let draggedQueueIdx = null;

    function renderQueue() {
        const listEl = document.getElementById("queue-list");
        const countEl = document.getElementById("queue-total-pill");
        const batchBtn = document.getElementById("btn-batch-run");
        if (countEl) countEl.innerText = String(state.queue.length);
        if (batchBtn) batchBtn.disabled = state.queue.length === 0 || state.batchRunning;
        updateWorkspaceChromeState();

        // Update filter badge counts
        const counts = { all: state.queue.length, needs_review: 0, ocr_done: 0, verified: 0, unprocessed: 0 };
        state.queue.forEach((it) => {
            const sl = statusToSlug(it.status);
            if (counts[sl] !== undefined) counts[sl]++;
        });
        document.querySelectorAll(".filter-pill").forEach((pill) => {
            const f = pill.dataset.filter;
            const base = pill.dataset.label || f;
            const c = counts[f] || 0;
            pill.innerText = f === "all" ? `All (${c})` : (c > 0 ? `${base} (${c})` : base);
        });

        if (!listEl) return;
        listEl.innerHTML = "";

        const visible = state.queue.filter((item) => {
            if (state.queueFilter !== "all" && statusToSlug(item.status) !== state.queueFilter) return false;
            if (state.queueSearch && !item.name.toLowerCase().includes(state.queueSearch)) return false;
            return true;
        });

        if (visible.length === 0) {
            listEl.innerHTML = `<div class="queue-empty">${
                state.queue.length === 0
                    ? "Corpus queue is empty.<br>Drop multiple manuscript images, a multi-page PDF, or load a curated specimen."
                    : "No folios match the active status filter."
            }</div>`;
            return;
        }

        visible.forEach((item) => {
            const realIdx = state.queue.findIndex((q) => q.id === item.id);
            const card = document.createElement("div");
            card.className = `folio-card ${item.id === state.activeFolioId ? "active" : ""}`;
            card.draggable = true;
            card.onclick = () => selectFolio(item.id);

            card.addEventListener("dragstart", () => { draggedQueueIdx = realIdx; });
            card.addEventListener("dragover", (e) => {
                e.preventDefault();
                card.classList.add("drag-target");
            });
            card.addEventListener("dragleave", () => card.classList.remove("drag-target"));
            card.addEventListener("drop", (e) => {
                e.preventDefault();
                card.classList.remove("drag-target");
                if (draggedQueueIdx !== null && draggedQueueIdx !== realIdx) {
                    const [moved] = state.queue.splice(draggedQueueIdx, 1);
                    state.queue.splice(realIdx, 0, moved);
                    persistQueueIndex();
                    renderQueue();
                }
                draggedQueueIdx = null;
            });

            const slug = statusToSlug(item.status);
            const lCount = Array.isArray(item.lines) ? item.lines.length : (item.lineCount || 0);

            card.innerHTML = `
                <div class="folio-thumb">
                    ${item.thumb ? `<img src="data:image/jpeg;base64,${item.thumb}" alt="">` : ""}
                    <span class="folio-index-tag">#${realIdx + 1}</span>
                </div>
                <div class="folio-meta">
                    <div class="folio-title" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
                    <div class="folio-sub">
                        <span class="doc-badge ${slug}">${escapeHtml(item.status || "Unprocessed")}</span>
                        ${lCount > 0 ? `<span>${lCount} ln</span>` : ""}
                    </div>
                </div>
                <div class="folio-actions">
                    <button type="button" class="folio-mini-btn" title="Transcribe this folio" data-act="ocr">▶</button>
                    <button type="button" class="folio-mini-btn danger" title="Remove folio" data-act="del">✕</button>
                </div>
            `;

            card.querySelector('[data-act="ocr"]').onclick = (ev) => {
                ev.stopPropagation();
                runSingleFolioOCR(item.id);
            };
            card.querySelector('[data-act="del"]').onclick = (ev) => removeFolio(item.id, ev);

            listEl.appendChild(card);
        });
    }

    function updateCorpusFooter() {
        let totalLines = 0;
        let verifiedLines = 0;
        state.queue.forEach((it) => {
            const lc = Array.isArray(it.lines) ? it.lines.length : (it.lineCount || 0);
            const vc = Array.isArray(it.lines) ? it.lines.filter((l) => l.human_verified).length : (it.verifiedCount || 0);
            totalLines += lc;
            verifiedLines += vc;
        });
        const pct = totalLines > 0 ? Math.round((verifiedLines / totalLines) * 100) : 0;
        const summaryEl = document.getElementById("corpus-footer-summary");
        const pctEl = document.getElementById("corpus-footer-pct");
        const fillEl = document.getElementById("corpus-verified-fill");
        if (summaryEl) summaryEl.innerHTML = `<strong>${state.queue.length}</strong> folios · <strong>${totalLines}</strong> lines`;
        if (pctEl) pctEl.innerText = `${pct}% verified`;
        if (fillEl) fillEl.style.width = `${pct}%`;
    }

    // =========================================================================
    // Single-Folio & Automated Batch OCR Execution
    // =========================================================================
    async function executeOcrRequestForFolio(folioItem) {
        if (!folioItem.image) {
            const dbRec = await dbGetFolio(folioItem.id);
            if (dbRec && dbRec.image) folioItem.image = dbRec.image;
        }
        if (!folioItem.image) throw new Error("Missing image data for folio");

        const modelSelect = document.getElementById("model-select");
        const chosenModel = modelSelect ? modelSelect.value : (folioItem.model || "v24_1_r4");

        const res = await fetch("/ocr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image: folioItem.image,
                model: chosenModel,
                method: state.segMethod,
                reading_order: state.readingOrder,
                gap: state.valleyGap,
                window: state.valleyWindow
            })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        const mappedLines = (data.lines || []).map((ln) => {
            const copy = { ...ln };
            copy.human_verified = false;
            copy.original_text = copy.text || "";
            copy.original_confidence = typeof copy.confidence === "number" ? copy.confidence : null;
            copy.original_calibrated_confidence =
                typeof copy.calibrated_confidence === "number" ? copy.calibrated_confidence : null;
            if (copy.segmentation && !Array.isArray(copy.segmentation.reason)) {
                copy.segmentation.reason = copy.segmentation.reason ? [copy.segmentation.reason] : [];
            }
            return copy;
        });

        folioItem.model = chosenModel;
        folioItem.lines = mappedLines;
        folioItem.lineCount = mappedLines.length;
        folioItem.verifiedCount = 0;
        folioItem.document = data.document || null;
        folioItem.profiler = data.profiler || null;
        folioItem.model_info = data.model_info || null;
        folioItem.status = computeDocStatus(mappedLines, folioItem.document);

        await dbPutFolio(folioItem);
        persistQueueIndex();
        return data;
    }

    async function runActiveOCR() {
        if (!state.activeFolioId) {
            toast("Upload or select a manuscript folio first");
            return;
        }
        await runSingleFolioOCR(state.activeFolioId);
    }

    async function runSingleFolioOCR(folioId) {
        const item = state.queue.find((x) => x.id === folioId);
        if (!item) return;

        if (state.activeFolioId !== folioId) {
            await selectFolio(folioId);
        }

        const runBtn = document.getElementById("btn-run");
        const loader = document.getElementById("ocr-loading-overlay");
        const loaderTxt = document.getElementById("ocr-loading-text");
        const modelSel = document.getElementById("model-select");
        const modelLabel = modelSel ? modelSel.options[modelSel.selectedIndex].text : item.model;

        if (runBtn) runBtn.disabled = true;
        if (loaderTxt) loaderTxt.innerText = `Transcribing ${item.name} with ${modelLabel}...`;
        if (loader) loader.style.display = "flex";

        item.status = "Processing";
        renderQueue();
        updateHeaderFolioBadge(item);

        try {
            const data = await executeOcrRequestForFolio(item);
            state.lines = item.lines;
            state.documentMeta = item.document;
            state.profilerMeta = item.profiler;
            state.modelInfoMeta = data.model_info || null;
            state.undoStack = [];
            state.redoStack = [];
            state.selection = state.lines.length > 0
                ? { level: "line", lineIdx: 0, charIdx: null }
                : { level: "line", lineIdx: null, charIdx: null };

            updateWorkspaceChromeState();
            renderQueue();
            updateHeaderFolioBadge(item);
            updateCorpusFooter();
            renderOverlays();
            renderEditor();
            renderDocTelemetry();

            if (state.editorMode === "paragraphs" && state.lines.length > 0) {
                await refreshParagraphFlow();
            }

            const reasons = (item.document && item.document.reasons) || [];
            if (reasons.includes("blank_page")) {
                toast("Blank folio detected — no text lines extracted");
            } else if (reasons.includes("possible_missed_lines")) {
                toast(`Extracted ${state.lines.length} lines (flagged: possible missed ink)`);
            } else {
                toast(`Extracted ${state.lines.length} lines (${data.profiler?.backend_total_sec || 0}s)`);
            }
            appendLog(
                `[${item.name}] Extracted ${state.lines.length} lines | Status: ${item.status} | Latency: ${data.profiler?.backend_total_sec || "?"}s`,
                reasons.length ? "warn" : "success"
            );
        } catch (err) {
            item.status = "Unprocessed";
            renderQueue();
            updateHeaderFolioBadge(item);
            toast("Transcription failed: " + err.message);
            appendLog(`OCR Error on ${item.name}: ${err.message}`, "error");
        } finally {
            if (loader) loader.style.display = "none";
            if (runBtn && state.currentImageBase64) runBtn.disabled = false;
        }
    }

    async function runBatchOCR(scope = "unprocessed") {
        if (state.batchRunning) return;
        const targets = scope === "all"
            ? [...state.queue]
            : state.queue.filter((x) => statusToSlug(x.status) === "unprocessed");

        if (targets.length === 0) {
            if (state.queue.length > 0 && confirm("All queued folios have already been processed. Re-run OCR on the entire queue?")) {
                return runBatchOCR("all");
            }
            toast("No unprocessed folios in queue");
            return;
        }

        state.batchRunning = true;
        state.batchAbortRequested = false;

        const card = document.getElementById("batch-progress-card");
        const countSpan = document.getElementById("batch-progress-count");
        const subSpan = document.getElementById("batch-progress-sub");
        const fill = document.getElementById("batch-progress-fill");
        if (card) card.classList.add("active");
        renderQueue();

        let completed = 0;
        let totalExtracted = 0;

        for (let i = 0; i < targets.length; i++) {
            if (state.batchAbortRequested) break;
            const folio = targets[i];
            if (countSpan) countSpan.innerText = `Batch OCR: ${i + 1} / ${targets.length}`;
            if (subSpan) subSpan.innerText = folio.name;
            if (fill) fill.style.width = `${Math.round((i / targets.length) * 100)}%`;

            folio.status = "Processing";
            renderQueue();

            try {
                const data = await executeOcrRequestForFolio(folio);
                completed++;
                totalExtracted += folio.lines.length;
                if (folio.id === state.activeFolioId) {
                    state.lines = folio.lines;
                    state.documentMeta = folio.document;
                    state.profilerMeta = folio.profiler;
                    state.modelInfoMeta = data.model_info || null;
                    updateHeaderFolioBadge(folio);
                    renderOverlays();
                    renderEditor();
                    renderDocTelemetry();
                }
                appendLog(`Batch [${i + 1}/${targets.length}] ${folio.name}: ${folio.lines.length} lines (${folio.status})`, "success");
            } catch (err) {
                folio.status = "Unprocessed";
                appendLog(`Batch error on ${folio.name}: ${err.message}`, "error");
            }

            renderQueue();
            updateCorpusFooter();
        }

        if (fill) fill.style.width = "100%";
        setTimeout(() => {
            if (card) card.classList.remove("active");
        }, 800);

        const wasAborted = state.batchAbortRequested;
        state.batchRunning = false;
        state.batchAbortRequested = false;
        renderQueue();

        toast(
            wasAborted
                ? `Batch paused after ${completed} folio(s) (${totalExtracted} lines)`
                : `Batch completed: ${completed} folio(s), ${totalExtracted} lines extracted`
        );
    }

    function stopBatchOCR() {
        if (state.batchRunning) {
            state.batchAbortRequested = true;
            toast("Stopping batch after current folio...");
        }
    }

    // =========================================================================
    // Paleographic Document Viewer: Pan, Zoom, Rotate, Filters, Loupe & Overlays
    // =========================================================================
    function updateCanvasTransform() {
        const layer = document.getElementById("viewer-transform-layer");
        const readout = document.getElementById("zoom-readout");
        if (layer) {
            layer.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
        }
        if (readout) {
            readout.innerText = `${Math.round(state.scale * 100)}%`;
        }
    }

    function zoomToFit() {
        const stage = document.getElementById("viewer-stage");
        const imgEl = document.getElementById("document-image");
        if (!stage || !imgEl || !imgEl.naturalWidth) return;
        const cw = stage.clientWidth;
        const ch = stage.clientHeight;
        if (cw <= 0 || ch <= 0) return;
        const s = Math.min(cw / imgEl.naturalWidth, ch / imgEl.naturalHeight) * 0.94;
        state.scale = Math.max(0.08, Math.min(8, s));
        state.panX = Math.round((cw - imgEl.naturalWidth * state.scale) / 2);
        state.panY = Math.round((ch - imgEl.naturalHeight * state.scale) / 2);
        updateCanvasTransform();
    }

    function zoomToWidth() {
        const stage = document.getElementById("viewer-stage");
        const imgEl = document.getElementById("document-image");
        if (!stage || !imgEl || !imgEl.naturalWidth) return;
        const cw = stage.clientWidth;
        state.scale = Math.max(0.08, Math.min(8, (cw / imgEl.naturalWidth) * 0.96));
        state.panX = Math.round((cw - imgEl.naturalWidth * state.scale) / 2);
        state.panY = 14;
        updateCanvasTransform();
    }

    function zoomTo100() {
        const stage = document.getElementById("viewer-stage");
        if (!stage) return;
        const cx = stage.clientWidth / 2;
        const cy = stage.clientHeight / 2;
        const ix = (cx - state.panX) / state.scale;
        const iy = (cy - state.panY) / state.scale;
        state.scale = 1.0;
        state.panX = cx - ix;
        state.panY = cy - iy;
        updateCanvasTransform();
    }

    function zoomStep(factor) {
        const stage = document.getElementById("viewer-stage");
        if (!stage) return;
        const cx = stage.clientWidth / 2;
        const cy = stage.clientHeight / 2;
        const ix = (cx - state.panX) / state.scale;
        const iy = (cy - state.panY) / state.scale;
        const ns = Math.max(0.08, Math.min(10, state.scale * factor));
        state.panX = cx - ix * ns;
        state.panY = cy - iy * ns;
        state.scale = ns;
        updateCanvasTransform();
    }

    async function rotateActiveImage90() {
        if (!state.currentImageBase64) return;
        const im = new Image();
        im.onload = async () => {
            const c = document.createElement("canvas");
            c.width = im.height;
            c.height = im.width;
            const ctx = c.getContext("2d");
            ctx.translate(c.width / 2, c.height / 2);
            ctx.rotate(Math.PI / 2);
            ctx.drawImage(im, -im.width / 2, -im.height / 2);
            const rotatedB64 = c.toDataURL("image/png").split(",")[1];
            state.currentImageBase64 = rotatedB64;
            const activeItem = state.queue.find((x) => x.id === state.activeFolioId);
            if (activeItem) {
                activeItem.image = rotatedB64;
                activeItem.thumb = await generateThumbnail(rotatedB64);
                activeItem.lines = [];
                activeItem.status = "Unprocessed";
                state.lines = [];
                await dbPutFolio(activeItem);
                persistQueueIndex();
                renderQueue();
            }
            const imgEl = document.getElementById("document-image");
            imgEl.src = "data:image/png;base64," + rotatedB64;
            toast("Rotated folio 90° clockwise — re-run transcription for updated geometry");
        };
        im.src = "data:image/png;base64," + state.currentImageBase64;
    }

    function applyCanvasFilters() {
        const imgEl = document.getElementById("document-image");
        if (!imgEl) return;
        const f = state.filters;
        const parts = [
            `contrast(${f.contrast}%)`,
            `brightness(${f.brightness}%)`,
            `grayscale(${f.grayscale}%)`
        ];
        if (f.invert) parts.push("invert(1) hue-rotate(180deg)");
        imgEl.style.filter = parts.join(" ");
    }

    function updateFilterControl(key, val) {
        if (key === "invert") state.filters.invert = Boolean(val);
        else state.filters[key] = Number(val);
        const lbl = document.getElementById(`lbl-filter-${key}`);
        if (lbl) lbl.innerText = `${state.filters[key]}%`;
        applyCanvasFilters();
    }

    function resetCanvasFilters() {
        state.filters = { contrast: 100, brightness: 100, grayscale: 0, invert: false };
        ["contrast", "brightness", "grayscale"].forEach((k) => {
            const inp = document.getElementById(`inp-filter-${k}`);
            const lbl = document.getElementById(`lbl-filter-${k}`);
            if (inp) inp.value = String(state.filters[k]);
            if (lbl) lbl.innerText = `${state.filters[k]}%`;
        });
        const inv = document.getElementById("inp-filter-invert");
        if (inv) inv.checked = false;
        applyCanvasFilters();
        toast("Reset manuscript inspection filters");
    }

    function toggleMagnifier() {
        state.magnifierEnabled = !state.magnifierEnabled;
        const btn = document.getElementById("btn-magnifier");
        const loupe = document.getElementById("magnifier-loupe");
        if (btn) btn.classList.toggle("active", state.magnifierEnabled);
        if (!state.magnifierEnabled && loupe) loupe.style.display = "none";
        if (state.magnifierEnabled) toast("Magnifier Loupe active — move cursor over manuscript");
    }

    function updateMagnifier(e) {
        const loupe = document.getElementById("magnifier-loupe");
        const stage = document.getElementById("viewer-stage");
        if (!state.magnifierEnabled || !loupe || !stage || !state.currentImageBase64) return;
        const rect = stage.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const imgX = (mx - state.panX) / state.scale;
        const imgY = (my - state.panY) / state.scale;

        if (imgX < 0 || imgY < 0 || imgX > state.imageWidth || imgY > state.imageHeight) {
            loupe.style.display = "none";
            return;
        }
        const zoom = 2.6;
        const size = 176;
        loupe.style.display = "block";
        loupe.style.left = `${Math.min(rect.width - size - 8, Math.max(8, mx + 18))}px`;
        loupe.style.top = `${Math.min(rect.height - size - 8, Math.max(8, my - size / 2))}px`;
        loupe.style.backgroundImage = `url("data:image/png;base64,${state.currentImageBase64}")`;
        loupe.style.backgroundSize = `${state.imageWidth * zoom}px ${state.imageHeight * zoom}px`;
        loupe.style.backgroundPosition = `${-(imgX * zoom - size / 2)}px ${-(imgY * zoom - size / 2)}px`;
    }

    // =========================================================================
    // Bounding Box & Curved Polygon Ribbon Rendering (Strictly per AGENTS.md)
    // =========================================================================
    function syncBboxModeUI() {
        const btn = document.getElementById("btn-toggle-bbox");
        const lbl = document.getElementById("lbl-bbox-mode");
        const bboxOverlay = document.getElementById("bbox-overlay");
        const polyOverlay = document.getElementById("polygon-overlay");
        state.showBboxOverlay = state.bboxMode !== "off";
        const modeLabels = { spotlight: "Spotlight", all: "All Boxes", off: "Hidden" };
        if (lbl) lbl.innerText = modeLabels[state.bboxMode] || "Spotlight";
        if (btn) btn.classList.toggle("active", state.bboxMode !== "off");
        if (bboxOverlay) bboxOverlay.classList.toggle("spotlight-mode", state.bboxMode === "spotlight");
        if (polyOverlay) polyOverlay.classList.toggle("spotlight-mode", state.bboxMode === "spotlight");
    }

    function cycleBboxMode() {
        const order = ["spotlight", "all", "off"];
        const nextIdx = (order.indexOf(state.bboxMode) + 1) % order.length;
        state.bboxMode = order[nextIdx];
        localStorage.setItem("cham_studio_bbox_mode", state.bboxMode);
        syncBboxModeUI();
        renderOverlays();
        const names = {
            spotlight: "Spotlight mode (subtle baselines + active focus)",
            all: "All core-band bounding boxes visible",
            off: "Bounding boxes hidden"
        };
        toast(names[state.bboxMode]);
    }

    function renderOverlays() {
        syncBboxModeUI();
        renderPolygons();
        renderBboxes();
    }

    function renderPolygons() {
        const svg = document.getElementById("polygon-overlay");
        if (!svg) return;
        svg.innerHTML = "";
        if (!state.showPolygonOverlay || !state.lines.length || !state.imageWidth) {
            svg.style.display = "none";
            return;
        }
        svg.style.display = "block";
        svg.setAttribute("width", String(state.imageWidth));
        svg.setAttribute("height", String(state.imageHeight));
        svg.setAttribute("viewBox", `0 0 ${state.imageWidth} ${state.imageHeight}`);

        state.lines.forEach((item, idx) => {
            if (!Array.isArray(item.polygon) || item.polygon.length < 3) return;
            const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
            const pts = item.polygon.map((p) => `${p[0]},${p[1]}`).join(" ");
            poly.setAttribute("points", pts);
            let cls = "polygon-ribbon";
            if (item.human_verified) cls += " verified";
            if (state.selection.lineIdx === idx) cls += " active";
            poly.setAttribute("class", cls);
            poly.id = `poly-line-${idx}`;
            svg.appendChild(poly);
        });
    }

    function renderBboxes() {
        const bboxOverlay = document.getElementById("bbox-overlay");
        if (!bboxOverlay) return;
        bboxOverlay.innerHTML = "";
        bboxOverlay.style.display = state.showBboxOverlay ? "block" : "none";
        if (!state.lines || state.lines.length === 0) return;

        state.lines.forEach((item, idx) => {
            const box = item.bbox;
            if (!box || box.length < 4) return;

            const left = box[0];
            const width = Math.max(4, box[2] - box[0]);

            // AGENTS.md Vertical Rule: strictly prioritize core_box `coords` to avoid ascender/descender bleed
            const hasCoords = Array.isArray(item.coords) && item.coords.length === 2;
            const top = hasCoords ? Math.max(0, item.coords[0] - 2) : Math.max(0, box[1] - 2);
            const height = hasCoords
                ? Math.max(6, (item.coords[1] - item.coords[0]) + 4)
                : Math.max(6, (box[3] - box[1]) + 4);

            const rect = document.createElement("div");
            rect.className = "bbox-rect";
            if (item.human_verified) rect.classList.add("verified");
            else if (item.segmentation && item.segmentation.needs_review) rect.classList.add("needs-review");
            if (state.selection.lineIdx === idx) rect.classList.add("active");

            rect.id = `bbox-line-${idx}`;
            rect.style.left = `${left}px`;
            rect.style.top = `${top}px`;
            rect.style.width = `${width}px`;
            rect.style.height = `${height}px`;

            const tag = document.createElement("span");
            tag.className = "bbox-line-tag";
            tag.innerText = `#${idx + 1} · ${Math.round((item.confidence || 0) * 100)}%`;
            rect.appendChild(tag);

            // Resize handles on active bounding box
            if (state.selection.lineIdx === idx) {
                ["nw", "ne", "sw", "se"].forEach((corner) => {
                    const h = document.createElement("div");
                    h.className = `bbox-handle ${corner}`;
                    h.addEventListener("mousedown", (ev) => startResizingBbox(ev, idx, corner));
                    rect.appendChild(h);
                });
            }

            rect.addEventListener("click", (ev) => {
                ev.stopPropagation();
                selectLine(idx, true);
            });
            rect.addEventListener("dblclick", (ev) => {
                ev.stopPropagation();
                selectLine(idx, true);
                enterEditMode(idx);
            });

            bboxOverlay.appendChild(rect);
        });

        if (state.selection.level === "char" && state.selection.lineIdx !== null && state.selection.charIdx !== null) {
            drawProportionalCharBbox(state.selection.lineIdx, state.selection.charIdx, false);
        }
    }

    // AGENTS.md Horizontal Rule: proportional grapheme width via Canvas measureText + Intl.Segmenter
    function drawProportionalCharBbox(lineIdx, charIdx, panToChar = false) {
        const existing = document.getElementById("char-bbox-indicator");
        if (existing) existing.remove();

        const bboxOverlay = document.getElementById("bbox-overlay");
        const lineData = state.lines[lineIdx];
        if (!bboxOverlay || !lineData || !lineData.bbox) return;

        const [xmin, ymin, xmax, ymax] = lineData.bbox;
        const text = lineData.text || "";
        const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });
        const segments = Array.from(segmenter.segment(text));
        if (!segments.length || charIdx < 0 || charIdx >= segments.length) return;

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        ctx.font = "20px 'Noto Sans Cham', 'Google Sans Flex', sans-serif";

        let totalW = 0;
        const widths = segments.map((seg) => {
            const w = Math.max(1, ctx.measureText(seg.segment).width);
            totalW += w;
            return w;
        });

        let xOffset = 0;
        for (let i = 0; i < charIdx; i++) xOffset += widths[i];

        const boxWidth = Math.max(1, xmax - xmin);
        const cX1 = Math.round(xmin + boxWidth * (xOffset / Math.max(1, totalW)));
        const cX2 = Math.round(xmin + boxWidth * ((xOffset + widths[charIdx]) / Math.max(1, totalW)));

        const hasCoords = Array.isArray(lineData.coords) && lineData.coords.length === 2;
        const vTop = hasCoords ? Math.max(0, lineData.coords[0] - 2) : Math.max(0, ymin - 2);
        const vHeight = hasCoords ? (lineData.coords[1] - lineData.coords[0]) + 4 : (ymax - ymin) + 4;

        const ind = document.createElement("div");
        ind.className = "char-bbox-rect";
        ind.id = "char-bbox-indicator";
        ind.style.left = `${cX1}px`;
        ind.style.top = `${vTop}px`;
        ind.style.width = `${Math.max(4, cX2 - cX1)}px`;
        ind.style.height = `${vHeight}px`;
        bboxOverlay.appendChild(ind);

        if (panToChar) {
            const stage = document.getElementById("viewer-stage");
            if (stage) {
                const cx = (cX1 + cX2) / 2;
                const cy = vTop + vHeight / 2;
                const targetScale = Math.max(state.scale, 1.75);
                state.panX = stage.clientWidth / 2 - cx * targetScale;
                state.panY = stage.clientHeight / 2 - cy * targetScale;
                state.scale = targetScale;
                updateCanvasTransform();
            }
        }
    }

    function startResizingBbox(ev, lineIdx, corner) {
        ev.stopPropagation();
        ev.preventDefault();
        const item = state.lines[lineIdx];
        if (!item || !item.bbox) return;
        const hasCoords = Array.isArray(item.coords) && item.coords.length === 2;
        const origBox = [
            item.bbox[0],
            hasCoords ? item.coords[0] : item.bbox[1],
            item.bbox[2],
            hasCoords ? item.coords[1] : item.bbox[3]
        ];
        state.resizingBbox = {
            lineIdx,
            corner,
            startClientX: ev.clientX,
            startClientY: ev.clientY,
            origBox
        };
    }

    function toggleDrawBboxMode(force) {
        state.isDrawBboxMode = typeof force === "boolean" ? force : !state.isDrawBboxMode;
        const btn = document.getElementById("btn-draw-bbox");
        const stage = document.getElementById("viewer-stage");
        if (btn) btn.classList.toggle("active", state.isDrawBboxMode);
        if (stage) stage.classList.toggle("draw-mode", state.isDrawBboxMode);
        if (state.isDrawBboxMode) {
            toast("Draw Line Box mode: drag across manuscript to delineate & transcribe a region");
        }
    }

    function toggleBboxOverlay() {
        cycleBboxMode();
    }

    function togglePolygonOverlay() {
        state.showPolygonOverlay = !state.showPolygonOverlay;
        const btn = document.getElementById("btn-toggle-poly");
        if (btn) btn.classList.toggle("active", state.showPolygonOverlay);
        renderPolygons();
    }

    async function ocrRegionAndUpsert(x1, y1, x2, y2, replaceIdx = null) {
        if (!state.currentImageBase64) return;
        const modelSel = document.getElementById("model-select");
        const modelVer = modelSel ? modelSel.value : "v24_1_r4";
        toast("Transcribing selected manuscript region...");

        try {
            const res = await fetch("/ocr_crop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: state.currentImageBase64,
                    bbox: [x1, y1, x2, y2],
                    model: modelVer
                })
            });
            const data = await res.json();
            if (data.error) {
                toast("Region OCR error: " + data.error);
                return;
            }

            pushUndoSnapshot();

            const rawCropB64 = data.image || (data.crop_image ? data.crop_image.split(",")[1] : "");
            if (replaceIdx !== null && state.lines[replaceIdx]) {
                const target = state.lines[replaceIdx];
                target.text = data.text || "";
                target.confidence = typeof data.confidence === "number" ? data.confidence : 1.0;
                target.calibrated_confidence = data.calibrated_confidence ?? target.confidence;
                target.bbox = [x1, y1, x2, y2];
                target.coords = [y1, y2];
                if (rawCropB64) target.image = rawCropB64;
                target.human_verified = false;
                renderOverlays();
                renderEditor();
                selectLine(replaceIdx, false);
                syncActiveFolioToQueue();
                toast(`Re-transcribed line #${replaceIdx + 1}`);
                return;
            }

            const newLine = {
                index: state.lines.length + 1,
                text: data.text || "",
                confidence: typeof data.confidence === "number" ? data.confidence : 1.0,
                calibrated_confidence: data.calibrated_confidence ?? 1.0,
                coords: [y1, y2],
                bbox: [x1, y1, x2, y2],
                image: rawCropB64,
                segmentation: {
                    selected_crop_type: "manual_draw",
                    needs_review: false,
                    reason: ["manual_drawn"]
                },
                human_verified: true
            };

            const midY = (y1 + y2) / 2;
            let insertIdx = state.lines.length;
            for (let i = 0; i < state.lines.length; i++) {
                const b = state.lines[i].bbox;
                if (b && (b[1] + b[3]) / 2 > midY) {
                    insertIdx = i;
                    break;
                }
            }
            state.lines.splice(insertIdx, 0, newLine);
            reindexLines();
            renderOverlays();
            renderEditor();
            selectLine(insertIdx, true);
            enterEditMode(insertIdx);
            syncActiveFolioToQueue();
            saveCorrectionToBackend(insertIdx, newLine.text);
            toggleDrawBboxMode(false);
            toast(`Inserted delineated textline #${insertIdx + 1}`);
        } catch (err) {
            toast("Network error calling /ocr_crop");
        }
    }

    function initViewerEvents() {
        const stage = document.getElementById("viewer-stage");
        const previewRect = document.getElementById("drawing-bbox-rect");
        if (!stage) return;

        stage.addEventListener("wheel", (e) => {
            if (!state.currentImageBase64) return;
            e.preventDefault();
            const rect = stage.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const ix = (mx - state.panX) / state.scale;
            const iy = (my - state.panY) / state.scale;
            const factor = e.deltaY < 0 ? 1.14 : 0.87;
            const ns = Math.max(0.08, Math.min(10, state.scale * factor));
            state.panX = mx - ix * ns;
            state.panY = my - iy * ns;
            state.scale = ns;
            updateCanvasTransform();
        }, { passive: false });

        stage.addEventListener("mousedown", (e) => {
            if (!state.currentImageBase64) return;
            if (
                e.target.closest(".empty-stage-card") ||
                e.target.closest(".canvas-floating-hud") ||
                e.target.closest(".canvas-top-pill")
            ) {
                return;
            }

            if (state.isDrawBboxMode) {
                const rect = stage.getBoundingClientRect();
                const ix = (e.clientX - rect.left - state.panX) / state.scale;
                const iy = (e.clientY - rect.top - state.panY) / state.scale;
                if (ix < 0 || iy < 0 || ix > state.imageWidth || iy > state.imageHeight) return;
                state.isDrawingBbox = true;
                state.drawStartPt = { x: ix, y: iy };
                if (previewRect) {
                    previewRect.style.left = `${ix}px`;
                    previewRect.style.top = `${iy}px`;
                    previewRect.style.width = "0px";
                    previewRect.style.height = "0px";
                    previewRect.style.display = "block";
                }
                e.preventDefault();
                return;
            }

            if (e.target.classList.contains("bbox-rect") || e.target.classList.contains("bbox-handle")) return;
            state.isPanning = true;
            state.panStartX = e.clientX - state.panX;
            state.panStartY = e.clientY - state.panY;
            e.preventDefault();
        });

        window.addEventListener("mousemove", (e) => {
            if (state.magnifierEnabled) updateMagnifier(e);

            if (state.resizingBbox) {
                const rb = state.resizingBbox;
                const dx = (e.clientX - rb.startClientX) / state.scale;
                const dy = (e.clientY - rb.startClientY) / state.scale;
                let [x1, y1, x2, y2] = rb.origBox;
                if (rb.corner.includes("w")) x1 = Math.max(0, Math.min(x2 - 12, Math.round(x1 + dx)));
                if (rb.corner.includes("e")) x2 = Math.min(state.imageWidth, Math.max(x1 + 12, Math.round(x2 + dx)));
                if (rb.corner.includes("n")) y1 = Math.max(0, Math.min(y2 - 8, Math.round(y1 + dy)));
                if (rb.corner.includes("s")) y2 = Math.min(state.imageHeight, Math.max(y1 + 8, Math.round(y2 + dy)));
                const item = state.lines[rb.lineIdx];
                if (item) {
                    item.bbox = [x1, y1, x2, y2];
                    item.coords = [y1, y2];
                    renderBboxes();
                }
                return;
            }

            if (state.isDrawingBbox && state.drawStartPt) {
                const rect = stage.getBoundingClientRect();
                const cx = Math.max(0, Math.min(state.imageWidth, (e.clientX - rect.left - state.panX) / state.scale));
                const cy = Math.max(0, Math.min(state.imageHeight, (e.clientY - rect.top - state.panY) / state.scale));
                if (previewRect) {
                    previewRect.style.left = `${Math.min(state.drawStartPt.x, cx)}px`;
                    previewRect.style.top = `${Math.min(state.drawStartPt.y, cy)}px`;
                    previewRect.style.width = `${Math.abs(cx - state.drawStartPt.x)}px`;
                    previewRect.style.height = `${Math.abs(cy - state.drawStartPt.y)}px`;
                }
                return;
            }

            if (state.isPanning) {
                state.panX = e.clientX - state.panStartX;
                state.panY = e.clientY - state.panStartY;
                updateCanvasTransform();
            }
        });

        window.addEventListener("mouseup", (e) => {
            if (state.resizingBbox) {
                const idx = state.resizingBbox.lineIdx;
                state.resizingBbox = null;
                syncActiveFolioToQueue();
                toast(`Adjusted bounding box for line #${idx + 1} — press R to re-transcribe crop`);
                return;
            }

            if (state.isDrawingBbox && state.drawStartPt) {
                state.isDrawingBbox = false;
                if (previewRect) previewRect.style.display = "none";
                const rect = stage.getBoundingClientRect();
                const cx = Math.max(0, Math.min(state.imageWidth, (e.clientX - rect.left - state.panX) / state.scale));
                const cy = Math.max(0, Math.min(state.imageHeight, (e.clientY - rect.top - state.panY) / state.scale));
                const x1 = Math.round(Math.min(state.drawStartPt.x, cx));
                const y1 = Math.round(Math.min(state.drawStartPt.y, cy));
                const x2 = Math.round(Math.max(state.drawStartPt.x, cx));
                const y2 = Math.round(Math.max(state.drawStartPt.y, cy));
                state.drawStartPt = null;
                if (x2 - x1 >= 10 && y2 - y1 >= 6) {
                    ocrRegionAndUpsert(x1, y1, x2, y2, null);
                }
                return;
            }

            state.isPanning = false;
        });

        // Drag & drop anywhere on viewer stage or queue intake
        [stage, document.getElementById("intake-dropzone")].forEach((zone) => {
            if (!zone) return;
            zone.addEventListener("dragover", (e) => {
                e.preventDefault();
                zone.classList.add("drag-over");
            });
            zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
            zone.addEventListener("drop", (e) => {
                e.preventDefault();
                zone.classList.remove("drag-over");
                if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    handleIncomingFiles(e.dataTransfer.files);
                }
            });
        });
    }

    // =========================================================================
    // Scholarly Transcription Editor, Paragraph Flow & Undo/Redo
    // =========================================================================
    function pushUndoSnapshot() {
        state.undoStack.push(JSON.stringify(state.lines));
        if (state.undoStack.length > 40) state.undoStack.shift();
        state.redoStack = [];
        updateUndoRedoButtons();
    }

    function undoEdit() {
        if (!state.undoStack.length) {
            toast("Nothing to undo");
            return;
        }
        state.redoStack.push(JSON.stringify(state.lines));
        state.lines = JSON.parse(state.undoStack.pop());
        updateUndoRedoButtons();
        renderOverlays();
        renderEditor();
        syncActiveFolioToQueue();
        toast("Undid last action");
    }

    function redoEdit() {
        if (!state.redoStack.length) {
            toast("Nothing to redo");
            return;
        }
        state.undoStack.push(JSON.stringify(state.lines));
        state.lines = JSON.parse(state.redoStack.pop());
        updateUndoRedoButtons();
        renderOverlays();
        renderEditor();
        syncActiveFolioToQueue();
        toast("Redid action");
    }

    function updateUndoRedoButtons() {
        const u = document.getElementById("btn-undo");
        const r = document.getElementById("btn-redo");
        if (u) u.disabled = state.undoStack.length === 0;
        if (r) r.disabled = state.redoStack.length === 0;
    }

    function reindexLines() {
        state.lines.forEach((ln, idx) => { ln.index = idx + 1; });
    }

    function setEditorMode(mode) {
        state.editorMode = mode === "paragraphs" ? "paragraphs" : "lines";
        document.querySelectorAll(".editor-mode-btn").forEach((b) => {
            b.classList.toggle("active", b.dataset.emode === state.editorMode);
        });
        if (state.editorMode === "paragraphs" && state.lines.length > 0) {
            refreshParagraphFlow();
        } else {
            renderEditor();
        }
    }

    async function refreshParagraphFlow() {
        if (!state.lines.length) {
            state.paragraphs = [];
            renderEditor();
            return;
        }
        try {
            const res = await fetch("/reconstruct_paragraphs", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    lines: state.lines.map((l) => ({
                        text: l.text || "",
                        bbox: l.bbox || [0, 0, 0, 0],
                        confidence: l.confidence || 1.0
                    }))
                })
            });
            const data = await res.json();
            if (Array.isArray(data.paragraphs) && data.paragraphs.length > 0) {
                state.paragraphs = data.paragraphs;
                renderEditor();
                return;
            }
        } catch (err) {
            console.warn("Using client paragraph reconstruction fallback:", err);
        }

        // Client fallback reconstruction using Cham terminators (꩞, ꩝, ?, !)
        const paras = [];
        let bucket = [];
        let indices = [];
        state.lines.forEach((ln, idx) => {
            const t = (ln.text || "").trim();
            if (!t) return;
            bucket.push(t);
            indices.push(idx);
            const endsStanza = /[꩞꩝?!:]$/.test(t);
            const nextText = idx + 1 < state.lines.length ? (state.lines[idx + 1].text || "").trim() : "";
            const nextStartsStanza = /^[꩐-꩙0-9]+[꩞.:\s–-]/.test(nextText);
            if (endsStanza || nextStartsStanza || idx === state.lines.length - 1) {
                paras.push({
                    paragraph_id: paras.length,
                    text: bucket.join(" "),
                    line_indices: [...indices]
                });
                bucket = [];
                indices = [];
            }
        });
        state.paragraphs = paras;
        renderEditor();
    }

    function syncCropModeUI() {
        const btn = document.getElementById("btn-toggle-crops");
        const lbl = document.getElementById("lbl-crop-mode");
        const labels = { focus: "Crop: Focus", all: "Crop: All", off: "Crop: Off" };
        if (lbl) lbl.innerText = labels[state.cropMode] || "Crop: Focus";
        if (btn) btn.classList.toggle("active", state.cropMode !== "off");
    }

    function cycleCropMode() {
        const order = ["focus", "all", "off"];
        const nextIdx = (order.indexOf(state.cropMode) + 1) % order.length;
        state.cropMode = order[nextIdx];
        localStorage.setItem("cham_studio_crop_mode", state.cropMode);
        syncCropModeUI();
        renderEditor();
        const desc = {
            focus: "Active-line focus crop (unfolds above active textline)",
            all: "Showing cutout strips on all lines",
            off: "Inline line cutouts hidden"
        };
        toast(desc[state.cropMode]);
    }

    function toggleLineCrops() {
        cycleCropMode();
    }

    function toggleRomanized() {
        state.showRomanized = !state.showRomanized;
        localStorage.setItem("cham_studio_roman", state.showRomanized ? "1" : "0");
        const btn = document.getElementById("btn-toggle-roman");
        if (btn) btn.classList.toggle("active", state.showRomanized);
        renderEditor();
    }

    function toggleInspector(force) {
        state.showInspector = typeof force === "boolean" ? force : !state.showInspector;
        localStorage.setItem("cham_studio_inspector", state.showInspector ? "1" : "0");
        const btn = document.getElementById("btn-toggle-inspector");
        const drawer = document.getElementById("inspector-drawer");
        if (btn) btn.classList.toggle("active", state.showInspector);
        if (drawer) drawer.classList.toggle("open", state.showInspector);
        if (state.showInspector) renderInspectorContent();
    }

    function toggleFindBar(force) {
        state.findBarOpen = typeof force === "boolean" ? force : !state.findBarOpen;
        const bar = document.getElementById("find-replace-bar");
        const btn = document.getElementById("btn-toggle-find");
        if (bar) bar.classList.toggle("open", state.findBarOpen);
        if (btn) btn.classList.toggle("active", state.findBarOpen);
        if (state.findBarOpen) {
            const inp = document.getElementById("find-query-input");
            if (inp) inp.focus();
        } else if (state.searchQuery) {
            state.searchQuery = "";
            renderEditor();
        }
    }

    function handleFindQueryChange(val) {
        state.searchQuery = String(val || "");
        renderEditor();
    }

    function executeReplaceAll() {
        const findVal = document.getElementById("find-query-input")?.value || "";
        const replVal = document.getElementById("replace-query-input")?.value || "";
        if (!findVal || !state.lines.length) return;
        pushUndoSnapshot();
        let count = 0;
        state.lines.forEach((ln) => {
            if (ln.text && ln.text.includes(findVal)) {
                const occurrences = ln.text.split(findVal).length - 1;
                count += occurrences;
                ln.text = ln.text.split(findVal).join(replVal);
                ln.human_verified = true;
            }
        });
        renderEditor();
        syncActiveFolioToQueue();
        toast(`Replaced ${count} occurrence(s) across active folio`);
    }

    function renderDocTelemetry() {
        const strip = document.getElementById("doc-telemetry-strip");
        if (!strip) return;
        if (!state.lines.length && !state.documentMeta) {
            strip.className = "doc-telemetry-strip";
            strip.innerHTML = "";
            return;
        }
        const doc = state.documentMeta || {};
        const prof = state.profilerMeta || {};
        const reasons = doc.reasons || [];
        const layout = doc.layout || {};
        const cols = layout.columns && layout.columns > 1 ? `${layout.columns}-column layout` : "Single-column flow";
        const det = prof.detector ? `Detector: ${prof.detector}` : `Method: ${prof.method || "dbnet"}`;
        const sec = prof.backend_total_sec !== undefined ? `${prof.backend_total_sec}s` : "";

        strip.className = "doc-telemetry-strip visible" + (reasons.length ? " warn" : "");
        strip.innerHTML = `
            <span>${escapeHtml(det)} · ${escapeHtml(cols)}${sec ? ` · ${sec}` : ""}</span>
            <span>${reasons.length ? `Flags: ${escapeHtml(reasons.join(", "))}` : `${state.lines.length} textlines`}</span>
        `;
    }

    function buildCropStripElement(item, idx, isFocusMode) {
        const cropBox = document.createElement("div");
        cropBox.className = "line-crop-strip" + (isFocusMode ? " focus-strip" : "");
        cropBox.title = "Synced manuscript line cutout — click to center & zoom on canvas";
        cropBox.onclick = (e) => {
            e.stopPropagation();
            selectLine(idx, false);
            zoomToLineBbox(idx);
        };
        cropBox.innerHTML = `<img src="data:image/png;base64,${item.image}" alt="Line #${idx + 1}">`;
        return cropBox;
    }

    function syncFocusCropSlots() {
        if (state.editorMode !== "lines") return;
        state.lines.forEach((item, idx) => {
            const slot = document.getElementById(`crop-slot-${idx}`);
            if (!slot) return;
            const isSelected = state.selection.lineIdx === idx || state.editingLineIdx === idx;
            const shouldShow =
                Boolean(item && item.image) &&
                (state.cropMode === "all" || state.layoutMode === "focus" || (state.cropMode === "focus" && isSelected));
            if (shouldShow) {
                if (!slot.firstChild) {
                    slot.appendChild(buildCropStripElement(item, idx, state.cropMode === "focus"));
                } else {
                    slot.firstChild.classList.toggle("focus-strip", state.cropMode === "focus");
                }
            } else if (slot.firstChild) {
                slot.innerHTML = "";
            }

            const extGroup = document.getElementById(`inline-ext-${idx}`);
            if (extGroup) {
                extGroup.style.display = isSelected ? "inline-flex" : "none";
            }
        });
    }

    function renderEditor() {
        const body = document.getElementById("editor-scroll-body");
        const countBadge = document.getElementById("editor-line-count");
        if (countBadge) countBadge.innerText = `${state.lines.length} lines`;
        updateUndoRedoButtons();
        updateWorkspaceChromeState();
        syncCropModeUI();

        if (!body) return;
        body.innerHTML = "";

        if (!state.lines || state.lines.length === 0) {
            body.innerHTML = `
                <div class="editor-ready-card">
                    <div class="empty-icon-circle" style="width:40px;height:40px;border-radius:11px;">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                    </div>
                    <div class="editor-ready-title">Ready to Transcribe Folio</div>
                    <div class="editor-ready-desc">
                        Run automatic line detection &amp; recognition, or delineate a specific region directly on the manuscript canvas.
                    </div>
                    <div class="editor-ready-actions">
                        <button type="button" class="btn btn-primary btn-sm" onclick="StudioApp.runActiveOCR()">
                            <span>▶ Transcribe Folio</span>
                            <span class="kbd-hint" style="background:rgba(255,255,255,0.2);color:#fff;border-color:rgba(255,255,255,0.3);">Ctrl+Enter</span>
                        </button>
                        <button type="button" class="btn btn-sm" onclick="StudioApp.toggleDrawBboxMode()">
                            <span>✎ Draw Region</span>
                            <span class="kbd-hint">D</span>
                        </button>
                    </div>
                </div>`;
            updateActionBar();
            return;
        }

        if (state.editorMode === "paragraphs") {
            renderParagraphFlowEditor(body);
            updateActionBar();
            return;
        }

        const segmenter = new Intl.Segmenter("en", { granularity: "grapheme" });

        state.lines.forEach((item, idx) => {
            const conf = typeof item.confidence === "number" ? item.confidence : 0;
            const tier = conf >= 0.75 ? "high" : (conf >= 0.45 ? "medium" : "low");
            const isSelected = state.selection.lineIdx === idx;
            const isEditing = state.editingLineIdx === idx;

            const row = document.createElement("div");
            row.className = `editor-line-row ${isSelected ? "active" : ""}`;
            row.id = `line-row-${idx}`;
            row.addEventListener("click", () => selectLine(idx, false, true));
            row.addEventListener("dblclick", (e) => {
                if (e.target.closest("button, input, .line-hover-tools")) return;
                e.stopPropagation();
                enterEditMode(idx);
            });

            const gutter = document.createElement("div");
            gutter.className = "line-gutter";
            gutter.innerHTML = `<span>${idx + 1}</span>`;

            const confBar = document.createElement("div");
            confBar.className = `conf-bar ${tier}`;
            confBar.title = `Recognition Confidence: ${Math.round(conf * 100)}%`;

            const lineBody = document.createElement("div");
            lineBody.className = "line-body";

            // Synced Line Cutout Slot (Active-Line Focus Crop by default to eliminate Gaze Pivot)
            const cropSlot = document.createElement("div");
            cropSlot.id = `crop-slot-${idx}`;
            const showCropNow =
                Boolean(item.image) &&
                (state.cropMode === "all" || state.layoutMode === "focus" || (state.cropMode === "focus" && (isSelected || isEditing)));
            if (showCropNow) {
                cropSlot.appendChild(buildCropStripElement(item, idx, state.cropMode === "focus"));
            }
            lineBody.appendChild(cropSlot);

            // Diplomatic Cham Script Display
            const viewDiv = document.createElement("div");
            viewDiv.className = "line-cham-view" + (conf < 0.60 && !item.human_verified ? " low-conf" : "");
            viewDiv.style.display = isEditing ? "none" : "block";
            viewDiv.ondblclick = (e) => {
                e.stopPropagation();
                enterEditMode(idx);
            };

            const txt = item.text || "";
            if (!txt) {
                viewDiv.innerHTML = `<span style="color:var(--text-tertiary);font-style:italic;font-size:13px;font-family:var(--font-sans);">(Empty line — double-click to transcribe)</span>`;
            } else {
                const segs = Array.from(segmenter.segment(txt));
                segs.forEach((seg, cIdx) => {
                    const sp = document.createElement("span");
                    sp.className = "char-span";
                    if (state.selection.level === "char" && isSelected && state.selection.charIdx === cIdx) {
                        sp.classList.add("active");
                    }
                    if (state.searchQuery && seg.segment.includes(state.searchQuery)) {
                        sp.classList.add("search-hit");
                    }
                    sp.id = `char-${idx}-${cIdx}`;
                    sp.innerText = seg.segment;
                    sp.onclick = (ev) => {
                        ev.stopPropagation();
                        selectCharacter(idx, cIdx, true);
                    };
                    viewDiv.appendChild(sp);
                });
                if (state.searchQuery && txt.includes(state.searchQuery)) {
                    viewDiv.style.background = "rgba(255, 232, 163, 0.28)";
                    viewDiv.style.borderRadius = "4px";
                }
            }
            lineBody.appendChild(viewDiv);

            // Non-blur Persistent Inline Input Mode (with Typewriter Enter / Shift+Enter flow)
            const editWrap = document.createElement("div");
            editWrap.className = "line-edit-wrap" + (isEditing ? " open" : "");
            const inp = document.createElement("input");
            inp.type = "text";
            inp.className = "line-cham-input";
            inp.id = `line-input-${idx}`;
            inp.value = txt;
            inp.addEventListener("focus", () => { state.activeInputEl = inp; });
            inp.addEventListener("keydown", (ev) => {
                if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
                    ev.preventDefault();
                    commitLineEdit(idx, inp.value, true);
                } else if (ev.key === "Enter" && ev.shiftKey) {
                    ev.preventDefault();
                    commitLineEdit(idx, inp.value, false);
                    syncActiveFolioToQueue();
                    if (idx > 0) {
                        enterEditMode(idx - 1);
                        panCanvasToLineGentle(idx - 1);
                    } else {
                        renderOverlays();
                        renderEditor();
                    }
                } else if (ev.key === "Enter") {
                    ev.preventDefault();
                    commitLineEdit(idx, inp.value, false);
                    syncActiveFolioToQueue();
                    if (idx + 1 < state.lines.length) {
                        enterEditMode(idx + 1);
                        panCanvasToLineGentle(idx + 1);
                    } else {
                        renderOverlays();
                        renderEditor();
                        toast("Completed final line of folio");
                    }
                } else if (ev.key === "Escape") {
                    ev.preventDefault();
                    cancelLineEdit();
                }
            });

            const saveBtn = document.createElement("button");
            saveBtn.type = "button";
            saveBtn.className = "btn btn-primary btn-xs";
            saveBtn.innerText = "Save";
            saveBtn.title = "Save line (Ctrl+Enter) — or press Enter to advance to next line";
            saveBtn.addEventListener("mousedown", (e) => e.preventDefault());
            saveBtn.onclick = (e) => {
                e.stopPropagation();
                commitLineEdit(idx, inp.value, true);
            };

            const cancelBtn = document.createElement("button");
            cancelBtn.type = "button";
            cancelBtn.className = "btn btn-xs";
            cancelBtn.innerText = "Esc";
            cancelBtn.addEventListener("mousedown", (e) => e.preventDefault());
            cancelBtn.onclick = (e) => {
                e.stopPropagation();
                cancelLineEdit();
            };

            editWrap.appendChild(inp);
            editWrap.appendChild(saveBtn);
            editWrap.appendChild(cancelBtn);
            lineBody.appendChild(editWrap);

            // Optional Scholarly Romanization Subtitle
            if (state.showRomanized && txt && global.ChamLinguistics) {
                const romDiv = document.createElement("div");
                romDiv.className = "line-roman-sub";
                romDiv.innerText = global.ChamLinguistics.transliterateCham(txt);
                lineBody.appendChild(romDiv);
            }

            // Metadata & Inline Contextual Actions Row (Right at the Active Line!)
            const metaRow = document.createElement("div");
            metaRow.className = "line-badges-row";

            const tags = document.createElement("div");
            tags.className = "line-meta-tags";
            tags.innerHTML = `<span class="micro-tag">${Math.round(conf * 100)}%</span>`;
            if (item.human_verified) {
                tags.innerHTML += `<span class="micro-tag verified">✓ Verified</span>`;
            } else if (item.segmentation && item.segmentation.needs_review) {
                tags.innerHTML += `<span class="micro-tag review">Needs review</span>`;
            }
            if (item.segmentation && item.segmentation.selected_crop_type === "dbnet" && Array.isArray(item.polygon) && item.polygon.length > 4) {
                tags.innerHTML += `<span class="micro-tag" title="Straightened via Cham-DBNet polygon ribbon">poly</span>`;
            }

            const hoverTools = document.createElement("div");
            hoverTools.className = "line-hover-tools";
            hoverTools.innerHTML = `
                <button type="button" class="btn btn-xs" data-lact="edit" title="Edit line (Enter)">Edit</button>
                <button type="button" class="btn btn-xs" style="color:var(--confidence-high);" data-lact="approve" title="Verify line (A)">✓</button>
                <button type="button" class="btn btn-xs" data-lact="reocr" title="Re-transcribe crop (R)">↻</button>
                <span id="inline-ext-${idx}" style="display:${isSelected ? "inline-flex" : "none"};align-items:center;gap:3px;">
                    <button type="button" class="btn btn-xs" data-lact="split" title="Split line at cursor/midpoint (S)">Split</button>
                    ${idx + 1 < state.lines.length ? `<button type="button" class="btn btn-xs" data-lact="merge" title="Merge with line below (M)">Merge ↓</button>` : ""}
                    <button type="button" class="btn btn-xs" data-lact="flag" title="Flag for review (W)">Flag</button>
                    <button type="button" class="btn btn-xs" data-lact="inspect" title="Inspect Unicode clusters (Alt+U)">U+</button>
                </span>
                <button type="button" class="btn btn-xs" data-lact="del" title="Delete line (Del)">🗑</button>
            `;
            hoverTools.querySelector('[data-lact="edit"]').onclick = (e) => { e.stopPropagation(); enterEditMode(idx); };
            hoverTools.querySelector('[data-lact="approve"]').onclick = (e) => { e.stopPropagation(); selectLine(idx, false); approveSelectedLine(); };
            hoverTools.querySelector('[data-lact="reocr"]').onclick = (e) => { e.stopPropagation(); reOcrLine(idx); };
            hoverTools.querySelector('[data-lact="split"]').onclick = (e) => { e.stopPropagation(); selectLine(idx, false); splitSelectedLine(); };
            const mergeBtn = hoverTools.querySelector('[data-lact="merge"]');
            if (mergeBtn) mergeBtn.onclick = (e) => { e.stopPropagation(); selectLine(idx, false); mergeSelectedWithNext(); };
            hoverTools.querySelector('[data-lact="flag"]').onclick = (e) => { e.stopPropagation(); selectLine(idx, false); flagSelectedWrong(); };
            hoverTools.querySelector('[data-lact="inspect"]').onclick = (e) => { e.stopPropagation(); selectLine(idx, false); toggleInspector(true); };
            hoverTools.querySelector('[data-lact="del"]').onclick = (e) => { e.stopPropagation(); deleteLine(idx); };

            metaRow.appendChild(tags);
            metaRow.appendChild(hoverTools);
            lineBody.appendChild(metaRow);

            row.appendChild(gutter);
            row.appendChild(confBar);
            row.appendChild(lineBody);
            body.appendChild(row);
        });

        updateActionBar();
        if (state.showInspector) renderInspectorContent();
    }

    function renderParagraphFlowEditor(body) {
        const paras = state.paragraphs.length ? state.paragraphs : [{ paragraph_id: 0, text: state.lines.map((l) => l.text).join(" "), line_indices: state.lines.map((_, i) => i) }];
        paras.forEach((p, pIdx) => {
            const card = document.createElement("div");
            card.className = "para-card";
            const lineRange = p.line_indices && p.line_indices.length
                ? `Lines #${p.line_indices[0] + 1}–#${p.line_indices[p.line_indices.length - 1] + 1}`
                : `Stanza ${pIdx + 1}`;
            card.innerHTML = `
                <div class="para-header">
                    <span>Paragraph / Stanza ${pIdx + 1}</span>
                    <span>${lineRange}</span>
                </div>
            `;
            const chamDiv = document.createElement("div");
            chamDiv.className = "para-cham-text";
            (p.line_indices || []).forEach((lIdx, k) => {
                const ln = state.lines[lIdx];
                if (!ln) return;
                const seg = document.createElement("span");
                seg.className = "para-segment" + (state.selection.lineIdx === lIdx ? " active" : "");
                seg.innerText = (ln.text || "").trim() + (k < p.line_indices.length - 1 ? " " : "");
                seg.title = `Line #${lIdx + 1} — Click to highlight on manuscript, Double-click to edit`;
                seg.onclick = () => {
                    selectLine(lIdx, false);
                    zoomToLineBbox(lIdx);
                    renderEditor();
                };
                seg.ondblclick = () => {
                    setEditorMode("lines");
                    selectLine(lIdx, true);
                    enterEditMode(lIdx);
                };
                chamDiv.appendChild(seg);
            });
            card.appendChild(chamDiv);

            if (state.showRomanized && global.ChamLinguistics) {
                const rom = document.createElement("div");
                rom.className = "para-roman-text";
                rom.innerText = global.ChamLinguistics.transliterateCham(p.text || "");
                card.appendChild(rom);
            }
            body.appendChild(card);
        });
    }

    // =========================================================================
    // Selection, Editing & Line Operations
    // =========================================================================
    function panCanvasToLineGentle(lineIdx) {
        const item = state.lines[lineIdx];
        const stage = document.getElementById("viewer-stage");
        if (!item || !item.bbox || !stage) return;
        const cx = (item.bbox[0] + item.bbox[2]) / 2;
        const cy = (item.bbox[1] + item.bbox[3]) / 2;
        const screenY = state.panY + cy * state.scale;
        const marginY = Math.min(90, stage.clientHeight * 0.22);
        if (screenY < marginY || screenY > stage.clientHeight - marginY) {
            state.panY = Math.round(stage.clientHeight / 2 - cy * state.scale);
            updateCanvasTransform();
        }
    }

    let focusCropTimer = null;

    function selectLine(lineIdx, scrollEditor = true, deferFocusCrop = false) {
        if (lineIdx === null || lineIdx < 0 || lineIdx >= state.lines.length) return;
        const wasAlreadySelected = (state.selection.lineIdx === lineIdx);
        if (state.editingLineIdx !== null && state.editingLineIdx !== lineIdx) {
            const prevInp = document.getElementById(`line-input-${state.editingLineIdx}`);
            if (prevInp) commitLineEdit(state.editingLineIdx, prevInp.value, false);
        }
        state.selection = { level: "line", lineIdx, charIdx: null };
        renderOverlays();

        document.querySelectorAll(".editor-line-row").forEach((r, i) => r.classList.toggle("active", i === lineIdx));
        document.querySelectorAll(".char-span.active").forEach((s) => s.classList.remove("active"));

        clearTimeout(focusCropTimer);
        if (deferFocusCrop && !wasAlreadySelected && state.cropMode === "focus") {
            // Defer unfolding the crop strip by 210ms so a double-click on an unselected line never shifts text under cursor!
            focusCropTimer = setTimeout(() => {
                syncFocusCropSlots();
            }, 210);
        } else {
            syncFocusCropSlots();
        }

        if (scrollEditor) {
            const rowEl = document.getElementById(`line-row-${lineIdx}`);
            if (rowEl) rowEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
        updateActionBar();
        if (state.showInspector) renderInspectorContent();
    }

    function selectCharacter(lineIdx, charIdx, panViewer = true) {
        state.selection = { level: "char", lineIdx, charIdx };
        document.querySelectorAll(".editor-line-row").forEach((r, i) => r.classList.toggle("active", i === lineIdx));
        document.querySelectorAll(".char-span").forEach((s) => s.classList.remove("active"));
        const targetSpan = document.getElementById(`char-${lineIdx}-${charIdx}`);
        if (targetSpan) targetSpan.classList.add("active");
        syncFocusCropSlots();

        renderPolygons();
        renderBboxes();
        drawProportionalCharBbox(lineIdx, charIdx, panViewer);
        updateActionBar();
        if (state.showInspector) renderInspectorContent();
    }

    function zoomToLineBbox(lineIdx) {
        const item = state.lines[lineIdx];
        const stage = document.getElementById("viewer-stage");
        if (!item || !item.bbox || !stage) return;
        const cx = (item.bbox[0] + item.bbox[2]) / 2;
        const cy = (item.bbox[1] + item.bbox[3]) / 2;
        const ts = Math.max(state.scale, 1.35);
        state.panX = stage.clientWidth / 2 - cx * ts;
        state.panY = stage.clientHeight / 2 - cy * ts;
        state.scale = ts;
        updateCanvasTransform();
    }

    function enterEditMode(lineIdx) {
        if (lineIdx === null || !state.lines[lineIdx]) return;
        clearTimeout(focusCropTimer);
        state.selection = { level: "line", lineIdx, charIdx: null };
        state.editingLineIdx = lineIdx;
        renderOverlays();
        renderEditor();
        const rowEl = document.getElementById(`line-row-${lineIdx}`);
        if (rowEl) rowEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        const inp = document.getElementById(`line-input-${lineIdx}`);
        if (inp) {
            state.activeInputEl = inp;
            inp.focus();
            inp.setSelectionRange(inp.value.length, inp.value.length);
        }
    }

    async function commitLineEdit(lineIdx, newRawValue, reRender = true) {
        const item = state.lines[lineIdx];
        if (!item) return;
        let normalized = String(newRawValue || "");
        if (global.ChamLinguistics && /[\uAA00-\uAA5F]/.test(normalized)) {
            normalized = global.ChamLinguistics.normalizeChamLogicalClient(normalized);
        }

        if (item.text !== normalized) {
            pushUndoSnapshot();
            item.text = normalized;
            item.confidence = 1.0;
            item.human_verified = true;
            if (item.segmentation) item.segmentation.needs_review = false;
            saveCorrectionToBackend(lineIdx, normalized);
        }

        state.editingLineIdx = null;
        state.activeInputEl = null;
        if (reRender) {
            renderOverlays();
            renderEditor();
            syncActiveFolioToQueue();
        }
    }

    function cancelLineEdit() {
        state.editingLineIdx = null;
        state.activeInputEl = null;
        renderEditor();
    }

    function insertTextIntoSelectedLine(str) {
        if (!str) return;
        if (state.editingLineIdx !== null) {
            const inp = document.getElementById(`line-input-${state.editingLineIdx}`);
            if (inp) {
                const s = inp.selectionStart ?? inp.value.length;
                const e = inp.selectionEnd ?? inp.value.length;
                inp.value = inp.value.slice(0, s) + str + inp.value.slice(e);
                inp.focus();
                inp.setSelectionRange(s + str.length, s + str.length);
                return;
            }
        }
        if (state.selection.lineIdx !== null && state.lines[state.selection.lineIdx]) {
            pushUndoSnapshot();
            const idx = state.selection.lineIdx;
            state.lines[idx].text = (state.lines[idx].text || "") + str;
            state.lines[idx].human_verified = true;
            renderEditor();
            syncActiveFolioToQueue();
            saveCorrectionToBackend(idx, state.lines[idx].text);
            toast(`Inserted into line #${idx + 1}`);
        } else {
            toast("Select a transcription line first");
        }
    }

    function approveSelectedLine() {
        const idx = state.selection.lineIdx;
        if (idx === null || !state.lines[idx]) {
            toast("Select a transcription line first");
            return;
        }
        pushUndoSnapshot();
        state.lines[idx].confidence = 1.0;
        state.lines[idx].human_verified = true;
        if (state.lines[idx].segmentation) state.lines[idx].segmentation.needs_review = false;

        saveCorrectionToBackend(idx, state.lines[idx].text);
        syncActiveFolioToQueue();
        renderQueue();
        renderOverlays();
        renderEditor();

        if (idx + 1 < state.lines.length) {
            selectLine(idx + 1, true);
        }
        toast(`Verified line #${idx + 1}`);
    }

    function approveAllHighConfidence() {
        if (!state.lines.length) return;
        pushUndoSnapshot();
        let count = 0;
        state.lines.forEach((ln) => {
            if ((ln.confidence || 0) >= 0.75 && !ln.human_verified) {
                ln.confidence = 1.0;
                ln.human_verified = true;
                if (ln.segmentation) ln.segmentation.needs_review = false;
                count++;
            }
        });
        syncActiveFolioToQueue();
        renderQueue();
        renderOverlays();
        renderEditor();
        toast(`Batch verified ${count} high-confidence line(s)`);
    }

    function flagSelectedWrong() {
        const idx = state.selection.lineIdx;
        if (idx === null || !state.lines[idx]) return;
        pushUndoSnapshot();
        const ln = state.lines[idx];
        ln.confidence = 0.2;
        ln.human_verified = false;
        ln.segmentation = ln.segmentation || {};
        ln.segmentation.needs_review = true;
        syncActiveFolioToQueue();
        renderQueue();
        renderOverlays();
        renderEditor();
        toast(`Flagged line #${idx + 1} for review`);
    }

    function splitSelectedLine() {
        const idx = state.selection.lineIdx;
        if (idx === null || !state.lines[idx]) {
            toast("Select a line to split");
            return;
        }
        const orig = state.lines[idx];
        const activeInp = state.editingLineIdx === idx ? document.getElementById(`line-input-${idx}`) : null;
        const fullText = activeInp ? activeInp.value : (orig.text || "");
        if (fullText.length <= 1) {
            toast("Line is too short to split");
            return;
        }

        // Split at exact cursor position if editing, otherwise smart split near midpoint whitespace/danda
        let cutPos = Math.ceil(fullText.length / 2);
        if (activeInp && typeof activeInp.selectionStart === "number" && activeInp.selectionStart > 0 && activeInp.selectionStart < fullText.length) {
            cutPos = activeInp.selectionStart;
        } else {
            const mid = Math.floor(fullText.length / 2);
            const spaceRight = fullText.indexOf(" ", mid);
            const spaceLeft = fullText.lastIndexOf(" ", mid);
            if (spaceRight !== -1 && spaceRight - mid < 12) cutPos = spaceRight + 1;
            else if (spaceLeft > 0 && mid - spaceLeft < 12) cutPos = spaceLeft + 1;
        }

        pushUndoSnapshot();
        const box = orig.bbox || [0, 0, state.imageWidth || 400, 40];
        const yMid = Math.round((box[1] + box[3]) / 2);

        const partA = JSON.parse(JSON.stringify(orig));
        partA.text = fullText.slice(0, cutPos).trim();
        partA.bbox = [box[0], box[1], box[2], yMid];
        partA.coords = [box[1], yMid];
        partA.human_verified = true;

        const partB = JSON.parse(JSON.stringify(orig));
        partB.text = fullText.slice(cutPos).trim();
        partB.bbox = [box[0], yMid, box[2], box[3]];
        partB.coords = [yMid, box[3]];
        partB.human_verified = true;

        state.editingLineIdx = null;
        state.lines.splice(idx, 1, partA, partB);
        reindexLines();
        syncActiveFolioToQueue();
        renderOverlays();
        renderEditor();
        selectLine(idx, true);
        toast(`Split line #${idx + 1} into two segments`);
    }

    function mergeSelectedWithNext() {
        const idx = state.selection.lineIdx;
        if (idx === null || idx + 1 >= state.lines.length) {
            toast("Select a non-final line to merge downward");
            return;
        }
        pushUndoSnapshot();
        const a = state.lines[idx];
        const b = state.lines[idx + 1];
        const mergedText = [a.text || "", b.text || ""].map((s) => s.trim()).filter(Boolean).join(" ");
        const boxA = a.bbox || [0, 0, 0, 0];
        const boxB = b.bbox || [0, 0, 0, 0];
        a.text = mergedText;
        a.bbox = [
            Math.min(boxA[0], boxB[0]),
            Math.min(boxA[1], boxB[1]),
            Math.max(boxA[2], boxB[2]),
            Math.max(boxA[3], boxB[3])
        ];
        const cA = a.coords || [boxA[1], boxA[3]];
        const cB = b.coords || [boxB[1], boxB[3]];
        a.coords = [Math.min(cA[0], cB[0]), Math.max(cA[1], cB[1])];
        a.confidence = 1.0;
        a.human_verified = true;
        if (a.segmentation) a.segmentation.needs_review = false;

        state.lines.splice(idx + 1, 1);
        reindexLines();
        saveCorrectionToBackend(idx, a.text);
        syncActiveFolioToQueue();
        renderOverlays();
        renderEditor();
        selectLine(idx, true);
        toast(`Merged lines #${idx + 1} & #${idx + 2}`);
    }

    function deleteLine(lineIdx) {
        const idx = typeof lineIdx === "number" ? lineIdx : state.selection.lineIdx;
        if (idx === null || !state.lines[idx]) return;
        pushUndoSnapshot();
        state.lines.splice(idx, 1);
        reindexLines();
        if (state.selection.lineIdx >= state.lines.length) {
            state.selection.lineIdx = state.lines.length ? state.lines.length - 1 : null;
        }
        syncActiveFolioToQueue();
        renderQueue();
        renderOverlays();
        renderEditor();
        toast(`Deleted line #${idx + 1}`);
    }

    function reOcrLine(lineIdx) {
        const idx = typeof lineIdx === "number" ? lineIdx : state.selection.lineIdx;
        if (idx === null || !state.lines[idx] || !state.lines[idx].bbox) {
            toast("Select a bounded line to re-transcribe");
            return;
        }
        const [x1, y1, x2, y2] = state.lines[idx].bbox;
        ocrRegionAndUpsert(x1, y1, x2, y2, idx);
    }

    function navFlaggedLine(dir) {
        if (!state.lines.length) return;
        const flagged = [];
        state.lines.forEach((ln, i) => {
            if (!ln.human_verified && ((ln.confidence || 0) < 0.60 || (ln.segmentation && ln.segmentation.needs_review))) {
                flagged.push(i);
            }
        });
        if (!flagged.length) {
            toast("No remaining lines flagged for review on this folio");
            return;
        }
        const cur = state.selection.lineIdx ?? -1;
        let next = flagged[0];
        if (dir === "next") {
            const f = flagged.find((i) => i > cur);
            next = f !== undefined ? f : flagged[0];
        } else {
            const rev = [...flagged].reverse();
            const f = rev.find((i) => i < cur);
            next = f !== undefined ? f : flagged[flagged.length - 1];
        }
        selectLine(next, true);
        zoomToLineBbox(next);
    }

    function updateActionBar() {
        const statusEl = document.getElementById("action-bar-status");
        if (!statusEl) return;
        const total = state.lines.length;
        const verified = state.lines.filter((l) => l.human_verified).length;
        const review = state.lines.filter(
            (l) => !l.human_verified && ((l.confidence || 0) < 0.60 || (l.segmentation && l.segmentation.needs_review))
        ).length;
        const idx = state.selection.lineIdx;
        const activePrefix = (idx !== null && state.lines[idx])
            ? `<strong style="color:var(--accent);">Line #${idx + 1}</strong> · `
            : "";
        statusEl.innerHTML = `
            ${activePrefix}<span><strong>${verified}</strong>/${total} verified</span>
            ${review > 0 ? `<span class="micro-tag review">${review} flagged</span>` : `<span class="micro-tag verified">Clean</span>`}
        `;
    }

    function renderInspectorContent() {
        const container = document.getElementById("inspector-content");
        if (!container || !global.ChamLinguistics) return;

        const idx = state.selection.lineIdx;
        if (idx === null || !state.lines[idx]) {
            container.innerHTML = `<div style="font-size:11.5px;color:var(--text-tertiary);">Click any character or textline in the editor to inspect its canonical Unicode codepoints (U+AA00–U+AA5F) and Brahmic cluster structure.</div>`;
            return;
        }

        const lineText = state.lines[idx].text || "";
        let targetStr = lineText;
        let headerLabel = `Line #${idx + 1} Cluster Overview`;

        if (state.selection.level === "char" && state.selection.charIdx !== null) {
            const segs = Array.from(new Intl.Segmenter("en", { granularity: "grapheme" }).segment(lineText));
            if (segs[state.selection.charIdx]) {
                targetStr = segs[state.selection.charIdx].segment;
                headerLabel = `Grapheme #${state.selection.charIdx + 1} in Line #${idx + 1}`;
            }
        } else if (lineText.length > 18) {
            targetStr = Array.from(lineText).slice(0, 18).join("");
            headerLabel = `Line #${idx + 1} (First 18 Codepoints — click a specific character to isolate)`;
        }

        const report = global.ChamLinguistics.inspectCluster(targetStr);
        const chipsHtml = report.codepoints.map((cp) => `
            <div class="cp-chip" title="${escapeHtml(cp.category)}">
                <span class="cp-glyph">${escapeHtml(cp.display)}</span>
                <div class="cp-details">
                    <span class="cp-hex">${cp.hex} · ${escapeHtml(cp.category)}</span>
                    <span class="cp-name">${escapeHtml(cp.name)}${cp.minimalPair ? ` · Watch pair: ${escapeHtml(cp.minimalPair)}` : ""}</span>
                </div>
            </div>
        `).join("");

        const orderTag = report.isCanonicalOrder
            ? `<span class="micro-tag verified">Canonical Logical Order ✓</span>`
            : `<span class="micro-tag review">Non-canonical cluster order</span>`;

        container.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;font-size:11.5px;">
                <strong>${escapeHtml(headerLabel)}</strong>
                <div style="display:flex;gap:6px;align-items:center;">
                    ${orderTag}
                    ${report.romanized ? `<span class="micro-tag">Roman: ${escapeHtml(report.romanized)}</span>` : ""}
                </div>
            </div>
            <div class="cp-chips-grid">${chipsHtml || "<span style='font-size:11.5px;color:var(--text-tertiary);'>Empty</span>"}</div>
        `;
    }

    function saveCorrectionToBackend(lineIdx, text) {
        if (!text || !text.trim()) return;
        const activeFolio = state.queue.find((x) => x.id === state.activeFolioId);
        const ln = state.lines[lineIdx] || {};
        const modelSel = document.getElementById("model-select");
        fetch("/save_correction", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text.trim(),
                line_idx: lineIdx + 1,
                image_name: activeFolio ? activeFolio.name : "manuscript.png",
                original_text: ln.original_text !== undefined ? ln.original_text : (ln.text || ""),
                confidence: ln.original_confidence !== undefined ? ln.original_confidence : (ln.confidence ?? null),
                calibrated_confidence:
                    ln.original_calibrated_confidence !== undefined
                        ? ln.original_calibrated_confidence
                        : (ln.calibrated_confidence ?? null),
                model: modelSel ? modelSel.value : "v24_1_r4",
                bbox: ln.bbox || null,
                crop_image: ln.image || null
            })
        }).catch(() => {});
    }

    // =========================================================================
    // Multi-Format & Multi-Page Corpus Export Suite
    // =========================================================================
    async function gatherExportDocuments(scope) {
        syncActiveFolioToQueue();
        if (scope === "active") {
            const cur = state.queue.find((x) => x.id === state.activeFolioId);
            return cur ? [cur] : [];
        }
        const docs = [];
        for (const it of state.queue) {
            if (!Array.isArray(it.lines)) {
                const stored = await dbGetFolio(it.id);
                if (stored && Array.isArray(stored.lines)) it.lines = stored.lines;
            }
            if (it.lines && it.lines.length > 0) docs.push(it);
        }
        return docs;
    }

    function triggerDownloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    async function exportFormat(kind) {
        const scopeEl = document.getElementById("export-scope-select");
        const scope = scopeEl ? scopeEl.value : "active";
        const docs = await gatherExportDocuments(scope);
        if (!docs.length) {
            toast("No transcribed lines available for the selected export scope");
            return;
        }

        closeModal("export-modal");

        if (kind === "copy_txt") {
            const out = docs.map((d) =>
                docs.length > 1
                    ? `=== ${d.name} ===\n` + (d.lines || []).map((l) => l.text).join("\n")
                    : (d.lines || []).map((l) => l.text).join("\n")
            ).join("\n\n");
            await navigator.clipboard.writeText(out);
            toast("Copied plaintext transcription to clipboard");
            return;
        }

        if (kind === "txt") {
            const out = docs.map((d) =>
                docs.length > 1
                    ? `# Folio: ${d.name}\n` + (d.lines || []).map((l) => l.text).join("\n")
                    : (d.lines || []).map((l) => l.text).join("\n")
            ).join("\n\n");
            triggerDownloadBlob(new Blob([out], { type: "text/plain;charset=utf-8" }), "cham_ocr_transcription.txt");
            toast("Exported Unicode plaintext (.txt)");
            return;
        }

        if (kind === "md") {
            const parts = [
                "# Cham OCR Studio — Paleographic Transcription Report",
                `- **Exported**: ${new Date().toISOString()}`,
                `- **Folios**: ${docs.length}`,
                ""
            ];
            docs.forEach((d, i) => {
                parts.push(`## ${i + 1}. ${d.name} (${d.status || "OCR done"})`);
                parts.push("");
                (d.lines || []).forEach((ln, lIdx) => {
                    parts.push(`${lIdx + 1}. **${ln.text || ""}**`);
                    if (global.ChamLinguistics && ln.text) {
                        parts.push(`   - *Romanized*: ${global.ChamLinguistics.transliterateCham(ln.text)} (\`${Math.round((ln.confidence || 0) * 100)}%\`)`);
                    }
                });
                parts.push("");
            });
            triggerDownloadBlob(new Blob([parts.join("\n")], { type: "text/markdown;charset=utf-8" }), "cham_ocr_scholarly_report.md");
            toast("Exported scholarly Markdown report (.md)");
            return;
        }

        if (kind === "json") {
            // Preserve exact single-page compatibility with scripts/ingest_studio_to_realscan_gold.py
            const payload = docs.length === 1
                ? { image_name: docs[0].name, status: docs[0].status, model: docs[0].model, lines: docs[0].lines }
                : { exported_at: new Date().toISOString(), documents: docs.map((d) => ({ image_name: d.name, status: d.status, model: d.model, lines: d.lines })) };
            triggerDownloadBlob(
                new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" }),
                "cham_ocr_transcription.json"
            );
            toast("Exported structured JSON (.json)");
            return;
        }

        if (kind === "pagexml") {
            const d = docs[0];
            const xmlLines = (d.lines || []).map((ln, idx) => {
                const b = ln.bbox || [0, 0, 100, 30];
                const pts = Array.isArray(ln.polygon) && ln.polygon.length >= 3
                    ? ln.polygon.map((p) => `${Math.round(p[0])},${Math.round(p[1])}`).join(" ")
                    : `${b[0]},${b[1]} ${b[2]},${b[1]} ${b[2]},${b[3]} ${b[0]},${b[3]}`;
                return `      <TextLine id="tl_${idx + 1}">
        <Coords points="${pts}"/>
        <TextEquiv conf="${(ln.confidence || 0).toFixed(4)}"><Unicode>${escapeXml(ln.text || "")}</Unicode></TextEquiv>
      </TextLine>`;
            }).join("\n");
            const xml = `<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata>
    <Creator>Cham OCR Studio (https://ocr.cham.asia)</Creator>
    <Created>${new Date().toISOString()}</Created>
  </Metadata>
  <Page imageFilename="${escapeXml(d.name)}" imageWidth="${state.imageWidth || 1000}" imageHeight="${state.imageHeight || 1000}">
    <TextRegion id="r1">
${xmlLines}
    </TextRegion>
  </Page>
</PcGts>`;
            triggerDownloadBlob(new Blob([xml], { type: "application/xml;charset=utf-8" }), `${d.name.replace(/\.[^.]+$/, "")}_pagexml.xml`);
            toast("Exported archival PAGE-XML (.xml)");
            return;
        }

        if (kind === "zip") {
            toast("Building Active Learning & Corpus ZIP archive...");
            try {
                const res = await fetch("/export_corpus_zip", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        documents: docs.map((d) => ({
                            name: d.name,
                            status: d.status,
                            model: d.model,
                            lines: d.lines
                        }))
                    })
                });
                if (!res.ok) throw new Error("ZIP export failed on server");
                const zipBlob = await res.blob();
                triggerDownloadBlob(zipBlob, "cham_ocr_corpus_export.zip");
                toast(`Downloaded ZIP package (${docs.length} folio(s) + crops + manifest)`);
            } catch (err) {
                toast("ZIP export error: " + err.message);
            }
        }
    }

    // =========================================================================
    // Command Palette (Ctrl+K / Cmd+K)
    // =========================================================================
    function getCommandCatalog() {
        return [
            { group: "Intake & Batch OCR", label: "Upload Manuscript Images or Multi-Page PDF", kbd: "Ctrl+O", run: () => document.getElementById("file-input").click() },
            { group: "Intake & Batch OCR", label: "Transcribe Active Folio", kbd: "Ctrl+Enter", run: () => runActiveOCR() },
            { group: "Intake & Batch OCR", label: "Batch Transcribe All Unprocessed Folios", kbd: "Ctrl+Shift+Enter", run: () => runBatchOCR("unprocessed") },
            { group: "Intake & Batch OCR", label: "Browse Curated Demo Manuscript Specimens", kbd: "", run: () => openModal("samples-modal") },
            { group: "Intake & Batch OCR", label: "Load 4-Folio Demo Batch into Queue", kbd: "", run: () => loadDemoBatchSet() },

            { group: "Workspace Layout", label: "Toggle Corpus Queue Drawer", kbd: "Alt+Q", run: () => toggleSidebarPane() },
            { group: "Workspace Layout", label: "Switch Layout: 3-Pane Resizable Workbench", kbd: "Alt+1", run: () => setLayoutMode("workbench") },
            { group: "Workspace Layout", label: "Switch Layout: Side-by-Side 2-Pane Compare", kbd: "Alt+2", run: () => setLayoutMode("side") },
            { group: "Workspace Layout", label: "Switch Layout: Stacked Horizontal–Vertical (Agal Palm-Leaf)", kbd: "Alt+3", run: () => setLayoutMode("stacked") },
            { group: "Workspace Layout", label: "Switch Layout: Focus Scholarly Reader", kbd: "Alt+4", run: () => setLayoutMode("focus") },

            { group: "Editor & Paleography", label: "Toggle Diplomatic Lines ↔ Continuous Paragraph Flow", kbd: "Alt+P", run: () => setEditorMode(state.editorMode === "lines" ? "paragraphs" : "lines") },
            { group: "Editor & Paleography", label: "Cycle Line Cutout Strip (Focus → All → Off)", kbd: "Alt+C", run: () => cycleCropMode() },
            { group: "Editor & Paleography", label: "Toggle Romanized Transliteration Subtitle (Aa)", kbd: "Alt+T", run: () => toggleRomanized() },
            { group: "Editor & Paleography", label: "Toggle Unicode Cluster & Codepoint Inspector", kbd: "Alt+U", run: () => toggleInspector() },
            { group: "Editor & Paleography", label: "Open Cham Virtual Keyboard (Akhar Thrah)", kbd: "Alt+K", run: () => global.ChamKeyboard && global.ChamKeyboard.toggle() },
            { group: "Editor & Paleography", label: "Batch Verify All High-Confidence Lines (≥ 75%)", kbd: "", run: () => approveAllHighConfidence() },
            { group: "Editor & Paleography", label: "Jump to Next Needs-Review Line", kbd: "]", run: () => navFlaggedLine("next") },

            { group: "Canvas Inspection", label: "Cycle Bounding Box Mode (Spotlight → All → Off)", kbd: "B", run: () => cycleBboxMode() },
            { group: "Canvas Inspection", label: "Draw Manual Bounding Box Region", kbd: "D", run: () => toggleDrawBboxMode() },
            { group: "Canvas Inspection", label: "Toggle Curved Polygon Ribbon Overlay", kbd: "P", run: () => togglePolygonOverlay() },
            { group: "Canvas Inspection", label: "Toggle 2.5× Paleographic Magnifier Loupe", kbd: "L", run: () => toggleMagnifier() },
            { group: "Canvas Inspection", label: "Rotate Active Folio 90° Clockwise", kbd: "", run: () => rotateActiveImage90() },
            { group: "Canvas Inspection", label: "Reset Image Contrast & Brightness Filters", kbd: "", run: () => resetCanvasFilters() },

            { group: "Export & Documentation", label: "Open Complete User Guide & Feature Manual (/help)", kbd: "?", run: () => window.open("/help", "_blank", "noopener") },
            { group: "Export & Documentation", label: "Open Multi-Format Corpus Export Suite", kbd: "Ctrl+E", run: () => openModal("export-modal") },
            { group: "Export & Documentation", label: "Download Active Learning Corpus ZIP", kbd: "", run: () => exportFormat("zip") },
            { group: "Export & Documentation", label: "Read Full Academic Technical Report (/research)", kbd: "", run: () => window.open("/research", "_blank", "noopener") },
            { group: "Export & Documentation", label: "Open Architecture & Paleographic Guide", kbd: "", run: () => openModal("about-modal") },
            { group: "Export & Documentation", label: "Open System Runtime Logs", kbd: "", run: () => openModal("logs-modal") }
        ];
    }

    function openCommandPalette() {
        openModal("cmdk-modal");
        const inp = document.getElementById("cmdk-input");
        if (inp) {
            inp.value = "";
            inp.focus();
        }
        state.cmdkSelectedIndex = 0;
        renderCommandPaletteList("");
    }

    function renderCommandPaletteList(query) {
        const listEl = document.getElementById("cmdk-list");
        if (!listEl) return;
        const q = String(query || "").trim().toLowerCase();
        const all = getCommandCatalog();
        const filtered = q
            ? all.filter((c) => c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q))
            : all;
        state.cmdkFilteredItems = filtered;
        state.cmdkSelectedIndex = Math.min(state.cmdkSelectedIndex, Math.max(0, filtered.length - 1));

        listEl.innerHTML = "";
        if (!filtered.length) {
            listEl.innerHTML = `<div style="padding:18px;text-align:center;color:var(--text-tertiary);font-size:12.5px;">No matching Studio commands</div>`;
            return;
        }

        let currentGroup = "";
        filtered.forEach((cmd, i) => {
            if (cmd.group !== currentGroup) {
                currentGroup = cmd.group;
                const gh = document.createElement("div");
                gh.className = "cmdk-group-title";
                gh.innerText = currentGroup;
                listEl.appendChild(gh);
            }
            const row = document.createElement("div");
            row.className = "cmdk-item" + (i === state.cmdkSelectedIndex ? " selected" : "");
            row.innerHTML = `
                <span>${escapeHtml(cmd.label)}</span>
                ${cmd.kbd ? `<span class="kbd-hint">${escapeHtml(cmd.kbd)}</span>` : ""}
            `;
            row.onclick = () => {
                closeModal("cmdk-modal");
                cmd.run();
            };
            listEl.appendChild(row);
        });
    }

    // =========================================================================
    // Popovers, Modals, Logs & Keyboard Shortcuts
    // =========================================================================
    function togglePopover(id, ev) {
        if (ev) ev.stopPropagation();
        const target = document.getElementById(id);
        if (!target) return;
        const wasOpen = target.classList.contains("open");
        closeAllPopovers();
        if (!wasOpen) target.classList.add("open");
    }

    function closeAllPopovers() {
        document.querySelectorAll(".popover-menu.open").forEach((m) => m.classList.remove("open"));
    }

    function openModal(id) {
        closeAllPopovers();
        const m = document.getElementById(id);
        if (m) m.classList.add("open");
    }

    function closeModal(id) {
        const m = document.getElementById(id);
        if (m) m.classList.remove("open");
    }

    function toast(msg) {
        const stack = document.getElementById("toast-stack");
        if (!stack) return;
        const el = document.createElement("div");
        el.className = "studio-toast";
        el.innerText = msg;
        stack.appendChild(el);
        setTimeout(() => {
            el.style.opacity = "0";
            setTimeout(() => el.remove(), 260);
        }, 2600);
    }

    function appendLog(message, level = "info") {
        const pre = document.getElementById("system-logs-container");
        if (!pre) return;
        const ts = new Date().toLocaleTimeString();
        const span = document.createElement("span");
        if (level === "error") span.style.color = "var(--confidence-low)";
        else if (level === "warn") span.style.color = "var(--confidence-medium)";
        else if (level === "success") span.style.color = "var(--confidence-high)";
        span.innerText = `[${ts}] ${message}\n`;
        pre.appendChild(span);
        pre.scrollTop = pre.scrollHeight;
    }

    function clearLogs() {
        const pre = document.getElementById("system-logs-container");
        if (pre) pre.innerHTML = "";
        appendLog("Runtime log buffer cleared.", "info");
    }

    function escapeHtml(str) {
        return String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
    }

    function escapeXml(str) {
        return escapeHtml(str).replace(/'/g, "&apos;");
    }

    function initGlobalShortcuts() {
        // AGENTS.md Safe Clipboard Blob Handling: pass getAsFile() directly without new File()
        document.addEventListener("paste", (e) => {
            const tag = document.activeElement ? document.activeElement.tagName : "";
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            const cd = e.clipboardData || window.clipboardData;
            if (!cd || !cd.items) return;
            for (let i = 0; i < cd.items.length; i++) {
                if (cd.items[i].type && cd.items[i].type.indexOf("image") !== -1) {
                    const blobFile = cd.items[i].getAsFile();
                    if (blobFile) {
                        e.preventDefault();
                        toast("Importing manuscript image from clipboard...");
                        handleIncomingFiles([blobFile]);
                        break;
                    }
                }
            }
        });

        document.addEventListener("click", (e) => {
            if (!e.target.closest(".popover-anchor")) {
                closeAllPopovers();
            }
            if (e.target.classList.contains("modal-backdrop")) {
                e.target.classList.remove("open");
            }
        });

        document.addEventListener("keydown", (e) => {
            const cmdkOpen = document.getElementById("cmdk-modal")?.classList.contains("open");
            if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) {
                e.preventDefault();
                if (cmdkOpen) closeModal("cmdk-modal");
                else openCommandPalette();
                return;
            }

            if (cmdkOpen) {
                if (e.key === "Escape") {
                    e.preventDefault();
                    closeModal("cmdk-modal");
                } else if (e.key === "ArrowDown") {
                    e.preventDefault();
                    state.cmdkSelectedIndex = Math.min(state.cmdkFilteredItems.length - 1, state.cmdkSelectedIndex + 1);
                    renderCommandPaletteList(document.getElementById("cmdk-input")?.value || "");
                } else if (e.key === "ArrowUp") {
                    e.preventDefault();
                    state.cmdkSelectedIndex = Math.max(0, state.cmdkSelectedIndex - 1);
                    renderCommandPaletteList(document.getElementById("cmdk-input")?.value || "");
                } else if (e.key === "Enter" && state.cmdkFilteredItems[state.cmdkSelectedIndex]) {
                    e.preventDefault();
                    const chosen = state.cmdkFilteredItems[state.cmdkSelectedIndex];
                    closeModal("cmdk-modal");
                    chosen.run();
                }
                return;
            }

            if (e.altKey && (e.key === "q" || e.key === "Q")) {
                e.preventDefault();
                toggleSidebarPane();
                return;
            }
            if (e.altKey && (e.key === "c" || e.key === "C")) {
                e.preventDefault();
                cycleCropMode();
                return;
            }
            if (e.altKey && (e.key === "k" || e.key === "K")) {
                e.preventDefault();
                if (global.ChamKeyboard) global.ChamKeyboard.toggle();
                return;
            }
            if (e.altKey && ["1", "2", "3", "4"].includes(e.key)) {
                e.preventDefault();
                const map = { "1": "workbench", "2": "side", "3": "stacked", "4": "focus" };
                setLayoutMode(map[e.key]);
                return;
            }
            if (e.altKey && (e.key === "p" || e.key === "P")) {
                e.preventDefault();
                setEditorMode(state.editorMode === "lines" ? "paragraphs" : "lines");
                return;
            }
            if (e.altKey && (e.key === "t" || e.key === "T")) {
                e.preventDefault();
                toggleRomanized();
                return;
            }
            if (e.altKey && (e.key === "u" || e.key === "U")) {
                e.preventDefault();
                toggleInspector();
                return;
            }

            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                if (e.shiftKey) runBatchOCR("unprocessed");
                else runActiveOCR();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === "o" || e.key === "O")) {
                e.preventDefault();
                document.getElementById("file-input")?.click();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === "e" || e.key === "E")) {
                e.preventDefault();
                openModal("export-modal");
                return;
            }

            const inInput = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName || "");
            if (inInput) return;

            if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
                e.preventDefault();
                if (e.shiftKey) redoEdit();
                else undoEdit();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
                e.preventDefault();
                redoEdit();
                return;
            }
            if ((e.ctrlKey || e.metaKey) && (e.key === "f" || e.key === "F")) {
                e.preventDefault();
                toggleFindBar(true);
                return;
            }

            if (e.key === "Escape") {
                if (state.isDrawBboxMode) toggleDrawBboxMode(false);
                else if (state.magnifierEnabled) toggleMagnifier();
                else if (global.ChamKeyboard && global.ChamKeyboard.isOpen()) global.ChamKeyboard.toggle(false);
                return;
            }

            if (e.key === "a" || e.key === "A") approveSelectedLine();
            else if (e.key === "b" || e.key === "B") cycleBboxMode();
            else if (e.key === "w" || e.key === "W") flagSelectedWrong();
            else if (e.key === "s" || e.key === "S") splitSelectedLine();
            else if (e.key === "m" || e.key === "M") mergeSelectedWithNext();
            else if (e.key === "r" || e.key === "R") reOcrLine();
            else if (e.key === "d" || e.key === "D") toggleDrawBboxMode();
            else if (e.key === "p" || e.key === "P") togglePolygonOverlay();
            else if (e.key === "l" || e.key === "L") toggleMagnifier();
            else if (e.key === "]") navFlaggedLine("next");
            else if (e.key === "[") navFlaggedLine("prev");
            else if (e.key === "?") window.open("/help", "_blank", "noopener");
            else if (e.key === "Delete" || e.key === "Backspace") {
                if (state.selection.lineIdx !== null) {
                    e.preventDefault();
                    deleteLine(state.selection.lineIdx);
                }
            } else if (e.key === "Enter" && state.selection.lineIdx !== null) {
                e.preventDefault();
                enterEditMode(state.selection.lineIdx);
            } else if (e.key === "ArrowDown" && state.selection.lineIdx !== null && state.selection.lineIdx + 1 < state.lines.length) {
                e.preventDefault();
                selectLine(state.selection.lineIdx + 1, true);
                panCanvasToLineGentle(state.selection.lineIdx);
            } else if (e.key === "ArrowUp" && state.selection.lineIdx !== null && state.selection.lineIdx - 1 >= 0) {
                e.preventDefault();
                selectLine(state.selection.lineIdx - 1, true);
                panCanvasToLineGentle(state.selection.lineIdx);
            }
        });
    }

    async function hydrateModelCatalog() {
        try {
            const res = await fetch("/api/models");
            if (!res.ok) return;
            const data = await res.json();
            const sel = document.getElementById("model-select");
            if (!sel || !Array.isArray(data.models)) return;
            const currentVal = sel.value || data.default_model || "v24_1_r4";
            const groups = {};
            data.models.forEach((m) => {
                if (!m.installed) return;
                if (!groups[m.group]) groups[m.group] = [];
                groups[m.group].push(m);
            });
            sel.innerHTML = "";
            Object.keys(groups).forEach((gName) => {
                const og = document.createElement("optgroup");
                og.label = gName;
                groups[gName].forEach((m) => {
                    const opt = document.createElement("option");
                    opt.value = m.id;
                    opt.textContent = m.short_label || m.label;
                    opt.title = `${m.label} — ${m.summary || ""}`;
                    if (m.id === currentVal) opt.selected = true;
                    og.appendChild(opt);
                });
                sel.appendChild(og);
            });
        } catch (err) {
            console.warn("Model catalog hydration skipped:", err);
        }
    }

    // =========================================================================
    // Bootstrap
    // =========================================================================
    document.addEventListener("DOMContentLoaded", () => {
        applyLayoutVars();
        setLayoutMode(state.layoutMode);
        initSplitResizers();
        initViewerEvents();
        initGlobalShortcuts();

        // Sync persisted UI toggles
        const autoChk = document.getElementById("chk-auto-ocr");
        if (autoChk) autoChk.checked = state.autoTranscribeOnUpload;
        syncBboxModeUI();
        syncCropModeUI();
        if (state.showRomanized) document.getElementById("btn-toggle-roman")?.classList.add("active");
        updateWorkspaceChromeState();

        const cmdkInput = document.getElementById("cmdk-input");
        if (cmdkInput) {
            cmdkInput.addEventListener("input", (e) => {
                state.cmdkSelectedIndex = 0;
                renderCommandPaletteList(e.target.value);
            });
        }

        appendLog("Cham OCR Studio Workbench initialized (https://ocr.cham.asia).", "info");
        hydrateModelCatalog();
        fetchCuratedSamples();
        restoreQueueFromStorage();
    });

    // Public API for HTML handlers & Virtual Keyboard integration
    global.StudioApp = {
        setLayoutMode,
        toggleSidebarPane,
        handleIncomingFiles,
        confirmPdfImport,
        loadSampleById,
        loadDemoBatchSet,
        setQueueFilter,
        setQueueSearch,
        setAutoOcr: (val) => {
            state.autoTranscribeOnUpload = Boolean(val);
            localStorage.setItem("cham_studio_auto_ocr", val ? "1" : "0");
        },
        setSegMethod: (val) => { state.segMethod = val; },
        setReadingOrder: (val) => { state.readingOrder = val; },
        setValleyGap: (val) => { state.valleyGap = parseInt(val, 10) || 12; },
        setValleyWindow: (val) => { state.valleyWindow = parseInt(val, 10) || 25; },
        clearEntireQueue,
        runActiveOCR,
        runBatchOCR,
        stopBatchOCR,
        zoomToFit,
        zoomToWidth,
        zoomTo100,
        zoomStep,
        rotateActiveImage90,
        updateFilterControl,
        resetCanvasFilters,
        toggleMagnifier,
        toggleDrawBboxMode,
        cycleBboxMode,
        toggleBboxOverlay,
        togglePolygonOverlay,
        setEditorMode,
        cycleCropMode,
        toggleLineCrops,
        toggleRomanized,
        toggleInspector,
        toggleFindBar,
        handleFindQueryChange,
        executeReplaceAll,
        undoEdit,
        redoEdit,
        approveSelectedLine,
        approveAllHighConfidence,
        flagSelectedWrong,
        splitSelectedLine,
        mergeSelectedWithNext,
        deleteLine,
        reOcrLine,
        navFlaggedLine,
        exportFormat,
        openCommandPalette,
        togglePopover,
        openModal,
        closeModal,
        clearLogs,
        toast,
        getActiveLineInput: () => state.activeInputEl,
        commitActiveEdit: () => {
            if (state.editingLineIdx !== null) {
                const inp = document.getElementById(`line-input-${state.editingLineIdx}`);
                if (inp) commitLineEdit(state.editingLineIdx, inp.value);
            }
        },
        insertTextIntoSelectedLine
    };
})(window);
