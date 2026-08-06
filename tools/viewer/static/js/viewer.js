/* Shared runtime for the config viewer (fresh implementation, vanilla JS,
   no build step). Exposes a single `V` namespace used by both pages. */
"use strict";

const V = (() => {
    // ---------- DOM builder ----------

    function el(tag, attrs = {}, ...children) {
        const node = document.createElement(tag);
        for (const [key, value] of Object.entries(attrs)) {
            if (value === undefined || value === null) continue;
            if (key === "class") node.className = value;
            else if (key === "dataset") Object.assign(node.dataset, value);
            else if (key.startsWith("on") && typeof value === "function") {
                node.addEventListener(key.slice(2), value);
            } else if (key === "value" && "value" in node) node.value = value;
            else if (typeof value === "boolean") { if (value) node.setAttribute(key, ""); }
            else node.setAttribute(key, value);
        }
        for (const child of children.flat(Infinity)) {
            if (child === undefined || child === null || child === false) continue;
            node.append(child instanceof Node ? child : String(child));
        }
        return node;
    }

    // ---------- data helpers ----------

    const clone = (v) => (v === undefined ? undefined : structuredClone(v));

    const isDict = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

    function getPath(obj, path) {
        let cur = obj;
        for (const key of path) {
            if (cur === null || cur === undefined) return undefined;
            cur = cur[key];
        }
        return cur;
    }

    function setPath(obj, path, value) {
        let cur = obj;
        for (let i = 0; i < path.length - 1; i++) {
            const key = path[i];
            if (cur[key] === undefined || cur[key] === null) {
                cur[key] = /^\d+$/.test(String(path[i + 1])) ? [] : {};
            }
            cur = cur[key];
        }
        cur[path[path.length - 1]] = value;
    }

    // Returns true when the delete changed an ARRAY's shape: splicing shifts
    // every later element down one, so any row path holding an index past the
    // removed one now points at a different value. Callers must re-render
    // their row list on true — and note infra.config.deep_merge merges lists
    // by index, so a stale index would silently re-target a child's override.
    function deletePath(obj, path) {
        const parents = [];
        let cur = obj;
        for (let i = 0; i < path.length - 1; i++) {
            if (!isDict(cur) && !Array.isArray(cur)) return false;
            parents.push(cur);
            cur = cur[path[i]];
        }
        if (cur === null || typeof cur !== "object") return false;
        const spliced = Array.isArray(cur);
        if (spliced) cur.splice(Number(path[path.length - 1]), 1);
        else delete cur[path[path.length - 1]];
        // prune now-empty dicts so removed overrides don't leave `key: {}` husks
        for (let i = path.length - 2; i >= 0; i--) {
            const holder = parents[i];
            const child = holder[path[i]];
            if (isDict(child) && Object.keys(child).length === 0) delete holder[path[i]];
            else break;
        }
        return spliced;
    }

    // ---------- persistence ----------

    function loadStored(key, fallback) {
        try {
            const raw = localStorage.getItem(key);
            return raw === null ? fallback : JSON.parse(raw);
        } catch {
            return fallback;
        }
    }

    const store = (key, value) => localStorage.setItem(key, JSON.stringify(value));

    // ---------- status bar / dirty state ----------

    let dirty = false;

    function status(message, kind = "") {
        const node = document.getElementById("status");
        node.textContent = message;
        node.className = `status ${kind}`;
    }

    function setDirty(value) {
        dirty = value;
        document.getElementById("dirty").hidden = !value;
    }

    const isDirty = () => dirty;

    window.addEventListener("beforeunload", (e) => {
        if (dirty) {
            e.preventDefault();
            e.returnValue = "";
        }
    });

    // ---------- fetch ----------

    async function api(url, body) {
        const options = body === undefined ? {} : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        };
        const response = await fetch(url, options);
        if (!response.ok) {
            let detail = response.statusText;
            try { detail = (await response.json()).detail || detail; } catch { /* keep */ }
            throw new Error(detail);
        }
        return response.json();
    }

    // Saves rewrite whole files through safe_load -> dump; the user must be
    // told comments/formatting are lost (spec-mandated wording).
    const SAVE_WARNING =
        "Saving rewrites the whole file: YAML comments are dropped and formatting is normalized. Continue?";

    // ---------- multiselect popover ----------

    function multiSelect({ mount, emptyLabel, selected, onChange }) {
        const caret = el("span", { class: "caret" }, "▼");
        const summary = el("span", {}, emptyLabel);
        const button = el("button", { class: "btn", type: "button" }, summary, caret);
        const panel = el("div", { class: "panel", hidden: true });
        const root = el("div", { class: "msel" }, button, panel);
        mount.append(root);

        let options = [];

        const refreshSummary = () => {
            if (selected.size === 0) summary.textContent = emptyLabel;
            else if (selected.size === 1) summary.textContent = [...selected][0];
            else summary.textContent = `${selected.size} selected`;
        };

        button.addEventListener("click", (e) => {
            e.stopPropagation();
            panel.hidden = !panel.hidden;
        });
        document.addEventListener("click", (e) => {
            if (!root.contains(e.target)) panel.hidden = true;
        });

        function rebuildPanel() {
            panel.replaceChildren(...options.map(({ value, sub }) => {
                const checkbox = el("input", { type: "checkbox" });
                checkbox.checked = selected.has(value);
                return el("div", {
                    class: "opt",
                    onclick: () => {
                        checkbox.checked = !checkbox.checked;
                        if (checkbox.checked) selected.add(value);
                        else selected.delete(value);
                        refreshSummary();
                        onChange();
                    },
                }, checkbox, el("span", {}, value, sub ? el("span", { class: "sub" }, sub) : null));
            }));
        }

        return {
            // options: [{value, sub?}]; validValues (optional): keep selections
            // in this set — defaults to the option values themselves. Pass a
            // wider set when the options are merely search-filtered.
            setOptions(next, validValues) {
                options = next;
                const known = new Set(validValues ?? options.map(o => o.value));
                for (const v of [...selected]) if (!known.has(v)) selected.delete(v);
                rebuildPanel();
                refreshSummary();
            },
            refresh() { rebuildPanel(); refreshSummary(); },
        };
    }

    // ---------- textarea autogrow ----------

    function autogrow(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = `${textarea.scrollHeight}px`;
    }

    // ---------- diff (fresh LCS implementation) ----------
    //
    // Suffix-table LCS over a flat Int32Array, walked forwards to emit
    // eq/del/ins ops; adjacent del/ins runs are paired into changed lines,
    // which get a character-level second pass when small enough.

    function lcsOps(a, b) {
        const n = a.length, m = b.length, width = m + 1;
        const table = new Int32Array((n + 1) * width);
        for (let i = n - 1; i >= 0; i--) {
            for (let j = m - 1; j >= 0; j--) {
                table[i * width + j] = a[i] === b[j]
                    ? table[(i + 1) * width + j + 1] + 1
                    : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
            }
        }
        const ops = [];
        let i = 0, j = 0;
        while (i < n && j < m) {
            if (a[i] === b[j]) { ops.push(["eq", a[i], b[j]]); i++; j++; }
            else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) { ops.push(["del", a[i], null]); i++; }
            else { ops.push(["ins", null, b[j]]); j++; }
        }
        while (i < n) ops.push(["del", a[i++], null]);
        while (j < m) ops.push(["ins", null, b[j++]]);
        return ops;
    }

    function charSpans(lineA, lineB) {
        const spansA = [], spansB = [];
        for (const [op, ca, cb] of lcsOps([...lineA], [...lineB])) {
            if (op === "eq") { spansA.push(["", ca]); spansB.push(["", cb]); }
            else if (op === "del") spansA.push(["del", ca]);
            else spansB.push(["ins", cb]);
        }
        return [spansA, spansB];
    }

    function mergeSpans(spans) {
        // coalesce adjacent chars with the same class into single spans
        const out = [];
        for (const [cls, ch] of spans) {
            if (out.length && out[out.length - 1][0] === cls) out[out.length - 1][1] += ch;
            else out.push([cls, ch]);
        }
        return out.map(([cls, text]) =>
            cls ? el("span", { class: cls }, text) : document.createTextNode(text));
    }

    function diffLine(nodes) { return el("div", { class: "dl" }, nodes); }

    // Returns {paneA, paneB, identical}: two aligned DOM panes.
    function lineDiff(textA, textB) {
        if (textA === textB) {
            return { identical: true, paneA: null, paneB: null };
        }
        // Per-PAIR cap: the char-level LCS table is O(lenA * lenB) ints, so it
        // must be bounded by the two lines' own lengths — a single pair of
        // very long lines would otherwise allocate hundreds of MB.
        const CHAR_DIFF_MAX_LINE = 1000;
        const paneA = el("div", { class: "diff-pane" });
        const paneB = el("div", { class: "diff-pane" });
        const flushPair = (dels, inss) => {
            const pairs = Math.min(dels.length, inss.length);
            for (let k = 0; k < pairs; k++) {
                if (dels[k].length <= CHAR_DIFF_MAX_LINE && inss[k].length <= CHAR_DIFF_MAX_LINE) {
                    const [sa, sb] = charSpans(dels[k], inss[k]);
                    paneA.append(diffLine(mergeSpans(sa)));
                    paneB.append(diffLine(mergeSpans(sb)));
                } else {
                    paneA.append(diffLine([el("span", { class: "del" }, dels[k])]));
                    paneB.append(diffLine([el("span", { class: "ins" }, inss[k])]));
                }
            }
            for (let k = pairs; k < dels.length; k++) {
                paneA.append(diffLine([el("span", { class: "del" }, dels[k])]));
                paneB.append(diffLine([]));
            }
            for (let k = pairs; k < inss.length; k++) {
                paneA.append(diffLine([]));
                paneB.append(diffLine([el("span", { class: "ins" }, inss[k])]));
            }
        };
        const linesA = textA.split("\n"), linesB = textB.split("\n");
        // Line-level LCS table is O(linesA * linesB) ints too: cap the PRODUCT
        // (two ~5k-line texts would otherwise allocate ~100MB) and fall back
        // to whole-text del/ins panes.
        if (linesA.length * linesB.length > 250000) {
            for (const la of linesA) paneA.append(diffLine([el("span", { class: "del" }, la)]));
            for (const lb of linesB) paneB.append(diffLine([el("span", { class: "ins" }, lb)]));
            return { identical: false, paneA, paneB };
        }
        let dels = [], inss = [];
        for (const [op, la, lb] of lcsOps(linesA, linesB)) {
            if (op === "eq") {
                flushPair(dels, inss);
                dels = []; inss = [];
                paneA.append(diffLine([document.createTextNode(la)]));
                paneB.append(diffLine([document.createTextNode(lb)]));
            } else if (op === "del") dels.push(la);
            else inss.push(lb);
        }
        flushPair(dels, inss);
        return { identical: false, paneA, paneB };
    }

    return {
        el, clone, isDict, getPath, setPath, deletePath,
        loadStored, store, status, setDirty, isDirty, api,
        SAVE_WARNING, multiSelect, autogrow, lineDiff,
    };
})();
