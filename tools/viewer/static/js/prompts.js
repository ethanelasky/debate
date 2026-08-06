/* Prompts page: prompt-library entries as columns, flattened components as
   rows. raw = each source file's own content (override detection + saving);
   resolved = _includes/_extends applied (display). The page never authors
   prompt text — textareas round-trip what the user types, and promoting an
   inherited component copies the existing resolved value verbatim. */
"use strict";

(() => {
    // The entry schema (infra/envs/debate/prompts.py): the fixed keys, plus
    // "cues" — every OTHER key is a per-turn cue keyed by protocol slot name,
    // and the page has no protocol, so it groups them all together.
    const GROUPS = [
        "vars",
        "overall_system", "debater_system", "debater_system_proposer",
        "debater_system_critic", "judge_system",
        "pre_debate", "pre_debate_proposer", "pre_debate_critic", "pre_debate_judge",
        "attribution",
        "cues",
    ];
    const CUES = "cues";
    // Stage keys: one row each, value a string or a block list.
    const STAGES = new Set(GROUPS.filter(g => g !== "vars" && g !== "attribution" && g !== CUES));
    const KEY_ENTRIES = "cv.prompts.entries";
    const KEY_COMPONENTS = "cv.prompts.components";

    // Metadata editors whose text does not currently parse as JSON, keyed by
    // entry+row. State keeps the last VALID value; saving is blocked until
    // these clear (the single rolling .backup.yaml cannot undo a bad save).
    const invalidJson = new Map();
    const editorKey = (name, row) => `${name} ${row.id}`;

    const state = {
        raw: null,
        resolved: null,
        sourceFiles: {},
        origSourceFiles: {},
        files: [],
        mtimes: {},
        invalid: {},
        cues: {},
        entries: new Set(V.loadStored(KEY_ENTRIES, [])),
        components: new Set(V.loadStored(KEY_COMPONENTS, [])),
    };

    const persist = () => {
        V.store(KEY_ENTRIES, [...state.entries]);
        V.store(KEY_COMPONENTS, [...state.components]);
    };

    const entrySelect = V.multiSelect({
        mount: document.getElementById("entrySelect"),
        emptyLabel: "All entries",
        selected: state.entries,
        onChange: () => { persist(); render(); },
    });
    const componentSelect = V.multiSelect({
        mount: document.getElementById("componentSelect"),
        emptyLabel: "All components",
        selected: state.components,
        onChange: () => { persist(); render(); },
    });

    // ---------- component flattening (new schema) ----------

    const isBlockList = (v) => Array.isArray(v) && v.every(x => typeof x === "string");

    function componentsOf(name) {
        const entry = state.resolved[name];
        if (!V.isDict(entry)) return [];
        const rows = [];
        for (const [key, value] of Object.entries(entry)) {
            if (key === "_extends") continue;
            if (key === "vars" && V.isDict(value)) {
                for (const varName of Object.keys(value)) {
                    rows.push({ path: [key, varName], group: key });
                }
            } else if (key === "attribution" && V.isDict(value)) {
                // reader -> author -> template: one row per (reader, author)
                // pair, since as a single row this renders as a JSON blob.
                for (const [reader, authors] of Object.entries(value)) {
                    if (V.isDict(authors)) {
                        for (const author of Object.keys(authors)) {
                            rows.push({ path: [key, reader, author], group: key });
                        }
                    } else {
                        rows.push({ path: [key, reader], group: key });
                    }
                }
            } else if (STAGES.has(key) || key === "vars" || key === "attribution") {
                // a stage (or a malformed vars/attribution) is one row; block
                // lists become one textarea per block inside that row.
                rows.push({ path: [key], group: key });
            } else {
                // any other key is a per-turn cue keyed by protocol slot name.
                rows.push({ path: [key], group: CUES });
            }
        }
        return rows.map(r => ({ ...r, id: r.path.join(".") }));
    }

    function collectRows(names) {
        const seen = new Set();
        const rows = [];
        for (const name of names) {
            for (const row of componentsOf(name)) {
                if (!seen.has(row.id)) { seen.add(row.id); rows.push(row); }
            }
        }
        const rank = (r) => {
            const i = GROUPS.indexOf(r.group);
            return i === -1 ? GROUPS.length : i;
        };
        return rows
            .map((row, i) => ({ row, i }))
            .sort((x, y) => rank(x.row) - rank(y.row) || x.i - y.i)
            .map(x => x.row);
    }

    // Joined text as prompts.py's _join renders block lists; metadata as JSON.
    function cellText(name, row) {
        const v = V.getPath(state.resolved[name], row.path);
        if (v === undefined) return null;
        if (typeof v === "string") return v;
        if (isBlockList(v)) return v.map(b => b.trim()).filter(Boolean).join("\n\n");
        return JSON.stringify(v, null, 2);
    }

    // ---------- inheritance (leaf-path granularity) ----------

    const parentOf = (name) => (V.isDict(state.raw[name]) ? state.raw[name]._extends : undefined);

    const inRaw = (name, row) => V.getPath(state.raw[name], row.path) !== undefined;

    const isInherited = (name, row) =>
        parentOf(name) !== undefined && !inRaw(name, row)
        && V.getPath(state.resolved[name], row.path) !== undefined;

    const canReset = (name, row) =>
        parentOf(name) !== undefined && inRaw(name, row)
        && V.getPath(state.resolved[parentOf(name)], row.path) !== undefined;

    function resetToInherited(name, row) {
        V.deletePath(state.raw[name], row.path);
        const fromParent = V.getPath(state.resolved[parentOf(name)], row.path);
        if (fromParent !== undefined) V.setPath(state.resolved[name], row.path, V.clone(fromParent));
        else V.deletePath(state.resolved[name], row.path);
        V.setDirty(true);
        render();
    }

    // ---------- editing ----------

    // blockIdx === null edits a plain string / metadata leaf; a number edits
    // one block of a list template. Editing an inherited component first
    // copies the resolved value into the child's raw entry (verbatim), then
    // replaces only the edited piece — matching by-index _extends merging.
    // Row paths address dict keys only (stage / vars name / reader+author /
    // cue slot name), so no path passes through a list index.
    function applyEdit(name, row, blockIdx, value) {
        if (!V.isDict(state.raw[name])) state.raw[name] = {};
        if (blockIdx === null) {
            V.setPath(state.raw[name], row.path, value);
        } else {
            let leaf = V.getPath(state.raw[name], row.path);
            const resolvedLeaf = V.getPath(state.resolved[name], row.path);
            if (!Array.isArray(leaf)) {
                leaf = isBlockList(resolvedLeaf) ? V.clone(resolvedLeaf) : [];
                V.setPath(state.raw[name], row.path, leaf);
            } else if (leaf.length <= blockIdx && isBlockList(resolvedLeaf)) {
                for (let i = leaf.length; i < resolvedLeaf.length; i++) leaf.push(resolvedLeaf[i]);
            }
            while (leaf.length <= blockIdx) leaf.push("");
            leaf[blockIdx] = value;
        }
        if (!V.isDict(state.resolved[name])) state.resolved[name] = {};
        V.setPath(state.resolved[name], row.path, V.clone(V.getPath(state.raw[name], row.path)));
        V.setDirty(true);
        // Children's resolved views are NOT recomputed in-session (that would
        // duplicate infra's resolution client-side); save + reload refreshes.
        const kids = Object.keys(state.raw).filter(n => parentOf(n) === name);
        if (kids.length) V.status(
            `"${name}" is extended by ${kids.join(", ")} — their resolved views refresh after save + reload`);
    }

    // ---------- load / save ----------

    async function load() {
        try {
            const payload = await V.api("/api/prompts");
            state.raw = payload.raw;
            state.resolved = payload.resolved;
            state.sourceFiles = payload.source_files || {};
            state.origSourceFiles = V.clone(state.sourceFiles);
            state.files = payload.files || [];
            state.mtimes = payload.mtimes || {};
            state.invalid = payload.invalid || {};
            state.cues = payload.cues || {};
            renderFileErrors(payload.errors || {});
            initFilters();
            render();
            V.setDirty(false);
            V.status("Prompts loaded", "ok");
        } catch (err) {
            V.status(`Error loading prompts: ${err.message}`, "err");
        }
    }

    async function save() {
        if (invalidJson.size) {
            // Those editors' values were never applied; saving now would write
            // the last valid value while the screen shows something else.
            V.status(
                `Fix the invalid JSON first: ${[...invalidJson.values()].join(", ")}`, "err");
            return;
        }
        if (!V.isDirty()) { V.status("No changes to save"); return; }
        if (!confirm(V.SAVE_WARNING)) return;
        const saveBtn = document.getElementById("saveBtn");
        try {
            saveBtn.disabled = true;
            V.status("Saving…");
            const overrides = {};
            for (const [name, file] of Object.entries(state.sourceFiles)) {
                if (!state.origSourceFiles[name] && file) overrides[name] = file;
            }
            const result = await V.api("/api/prompts", {
                raw: state.raw,
                source_file_overrides: overrides,
                mtimes: state.mtimes,
            });
            if (result.status === "no_changes") {
                V.status(result.message);
            } else {
                Object.assign(state.mtimes, result.mtimes || {});
                state.origSourceFiles = V.clone(state.sourceFiles);
                V.setDirty(false);
                V.status(result.message, "ok");
            }
        } catch (err) {
            V.status(`Error saving: ${err.message}`, "err");
        } finally {
            saveBtn.disabled = false;
        }
    }

    // ---------- filters ----------

    function renderFileErrors(errors) {
        const banner = document.getElementById("fileErrors");
        const broken = Object.keys(errors);
        banner.hidden = broken.length === 0;
        banner.replaceChildren(...broken.map(f =>
            V.el("div", {}, V.el("strong", {}, f), `: ${errors[f]}`)));
    }

    function initFilters() {
        const names = Object.keys(state.resolved);
        // setOptions prunes stale stored selections first; only THEN decide
        // whether the selection is empty and needs the first-entry default —
        // a fully-stale localStorage set must not render an empty page.
        entrySelect.setOptions(names.map(value => ({
            value,
            sub: state.sourceFiles[value] || "new",
        })));
        if (names.length && state.entries.size === 0) {
            state.entries.add(names[0]);
            entrySelect.refresh();
        }
        componentSelect.setOptions(collectRows(names).map(r => ({ value: r.id })));
        persist();
    }

    function visibleRows(names) {
        const term = document.getElementById("search").value.trim().toLowerCase();
        return collectRows(names).filter(row => {
            if (state.components.size > 0 && !state.components.has(row.id)) return false;
            if (!term) return true;
            if (row.id.toLowerCase().includes(term)) return true;
            return names.some(name => (cellText(name, row) || "").toLowerCase().includes(term));
        });
    }

    // ---------- rendering ----------

    function buildEditor(name, row) {
        const editor = V.el("div", { class: "cell-editor" });
        const resolvedLeaf = V.getPath(state.resolved[name], row.path);
        const rawLeaf = V.getPath(state.raw[name], row.path);
        const hasParent = parentOf(name) !== undefined;

        const area = (text, blockIdx, meta) => {
            const node = V.el("textarea", { spellcheck: "false" });
            node.value = text;
            let errorNote = null;
            node.addEventListener("input", () => {
                let value = node.value;
                if (meta) {
                    try {
                        value = JSON.parse(node.value);
                    } catch (err) {
                        // Mid-edit text lives in the DOM only: writing it into
                        // state would replace a structured YAML value with one
                        // string, and saving that is not recoverable from the
                        // single rolling backup. Block the save instead.
                        invalidJson.set(editorKey(name, row), `${name} / ${row.id}`);
                        node.classList.add("json-invalid");
                        if (!errorNote) {
                            errorNote = V.el("div", { class: "json-error" });
                            node.after(errorNote);
                        }
                        errorNote.textContent = `Invalid JSON — not applied: ${err.message}`;
                        V.autogrow(node);
                        V.status(`Invalid JSON in ${name} / ${row.id} — saving is blocked`, "err");
                        return;
                    }
                    invalidJson.delete(editorKey(name, row));
                    node.classList.remove("json-invalid");
                    if (errorNote) { errorNote.remove(); errorNote = null; }
                }
                applyEdit(name, row, blockIdx, value);
                V.autogrow(node);
                const td = node.closest("td");
                if (td) td.classList.remove("cell-inh");
                const chip = td && td.querySelector(".chip.inherited");
                if (chip) { chip.className = "chip override"; chip.textContent = "override"; }
            });
            return node;
        };

        if (isBlockList(resolvedLeaf) && resolvedLeaf.length > 1) {
            const rawLen = Array.isArray(rawLeaf) ? rawLeaf.length : 0;
            resolvedLeaf.forEach((block, i) => {
                const blockInherited = hasParent && (rawLeaf === undefined || i >= rawLen);
                editor.append(
                    V.el("div", { class: "block-tag" }, `block ${i + 1}${blockInherited ? " · inherited" : ""}`),
                    area(block, i, false),
                );
            });
        } else if (isBlockList(resolvedLeaf)) {
            editor.append(area(resolvedLeaf[0] ?? "", 0, false));
        } else if (resolvedLeaf === undefined || typeof resolvedLeaf === "string") {
            editor.append(area(resolvedLeaf ?? "", null, false));
        } else {
            editor.append(area(JSON.stringify(resolvedLeaf, null, 2), null, true));
        }
        return editor;
    }

    function buildCellFoot(name, row) {
        if (parentOf(name) === undefined) return null;
        const foot = V.el("div", { class: "cell-foot" });
        if (isInherited(name, row)) {
            foot.append(V.el("span", { class: "chip inherited" }, "inherited"));
        } else if (inRaw(name, row)) {
            foot.append(V.el("span", { class: "chip override" }, "override"));
            if (canReset(name, row)) {
                foot.append(V.el("button", {
                    class: "reset-btn",
                    title: "Remove the override and restore the inherited value",
                    onclick: () => resetToInherited(name, row),
                }, "↺"));
            }
        }
        return foot.childNodes.length ? foot : null;
    }

    function buildHeader(names) {
        const headRow = V.el("tr", {},
            V.el("th", { class: "rowhead" }, "Component"),
            names.map(name => {
                const sub = V.el("div", { class: "col-sub" });
                const parent = parentOf(name);
                if (parent !== undefined) {
                    sub.append(V.el("span", {
                        class: "chip extends",
                        title: "Show the parent entry",
                        onclick: () => {
                            state.entries.clear();
                            state.entries.add(parent);
                            persist();
                            entrySelect.refresh();
                            render();
                        },
                    }, `extends ${parent}`));
                }
                sub.append(V.el("span", { class: "chip file" },
                    state.sourceFiles[name] || "new entry"));
                if (state.invalid[name]) {
                    sub.append(V.el("span", {
                        class: "chip error",
                        title: "load_prompt_library would refuse this entry",
                    }, `invalid: ${state.invalid[name].join(", ")}`));
                }
                const cues = state.cues[name] || [];
                if (cues.length) {
                    // Informational: without the protocol the page cannot say
                    // whether these are real slot names or typo'd stages.
                    sub.append(V.el("span", {
                        class: "chip",
                        title: "read as per-turn cues, keyed by protocol slot name",
                    }, `cues: ${cues.join(", ")}`));
                }
                return V.el("th", {}, V.el("div", { class: "col-title" }, name), sub);
            }),
        );
        return V.el("thead", {}, headRow);
    }

    function render() {
        // every textarea is rebuilt from state (i.e. from last-valid values),
        // so no unparsed mid-edit text survives this
        invalidJson.clear();
        const mount = document.getElementById("tableMount");
        const names = Object.keys(state.resolved || {}).filter(
            n => state.entries.size === 0 || state.entries.has(n));
        const diffBar = document.getElementById("diffBar");

        if (!names.length) {
            mount.replaceChildren(V.el("div", { class: "empty" }, "No entries selected"));
            diffBar.hidden = true;
            return;
        }

        const rows = visibleRows(names);
        // Diff mode requires exactly two EXPLICITLY SELECTED entries — an
        // empty selection falling back to "all entries" must never trigger it.
        const diffMode = state.entries.size === 2 && names.length === 2;
        let sameCount = 0, diffCount = 0;

        const tbody = V.el("tbody");
        let lastGroup = null;

        for (const row of rows) {
            if (row.group !== lastGroup) {
                tbody.append(V.el("tr", { class: "group-row" },
                    V.el("td", { colspan: names.length + 1 }, row.group)));
                lastGroup = row.group;
            }

            let diff = null;
            if (diffMode) {
                const a = cellText(names[0], row);
                const b = cellText(names[1], row);
                diff = {
                    a: a ?? "", b: b ?? "",
                    aMissing: a === null, bMissing: b === null,
                    ...V.lineDiff(a ?? "", b ?? ""),
                };
                diff.identical = diff.identical && diff.aMissing === diff.bMissing;
            }

            const tr = V.el("tr");
            const label = V.el("td", { class: "rowhead" });
            if (diffMode) {
                if (diff.identical) {
                    sameCount++;
                    tr.className = "row-same collapsed";
                    label.append(V.el("span", { class: "twist" }, "▾"), row.id, " ",
                        V.el("span", { class: "chip same" }, "same"));
                    label.addEventListener("click", () => tr.classList.toggle("collapsed"));
                } else {
                    diffCount++;
                    tr.className = "row-diff";
                    label.append(row.id, " ", V.el("span", { class: "chip differs" }, "differs"));
                }
            } else {
                label.append(row.id);
            }
            tr.append(label);

            names.forEach((name, idx) => {
                const td = V.el("td", { class: `cell${isInherited(name, row) ? " cell-inh" : ""}` });
                const body = V.el("div", { class: "cell-body" });

                if (diffMode && !diff.identical) {
                    const missing = idx === 0 ? diff.aMissing : diff.bMissing;
                    // paneA/paneB are null when the texts are equal but one
                    // side is missing — fall back to a plain pane of the text.
                    let pane;
                    if (missing) {
                        pane = V.el("div", { class: "diff-pane" },
                            V.el("span", { class: "missing" }, "(not defined)"));
                    } else {
                        pane = (idx === 0 ? diff.paneA : diff.paneB)
                            ?? V.el("div", { class: "diff-pane" },
                                V.el("div", { class: "dl" }, idx === 0 ? diff.a : diff.b));
                    }
                    pane.title = "Click to edit";
                    const editor = buildEditor(name, row);
                    editor.style.display = "none";
                    pane.addEventListener("click", () => {
                        pane.style.display = "none";
                        editor.style.display = "";
                        editor.querySelectorAll("textarea").forEach(V.autogrow);
                        const first = editor.querySelector("textarea");
                        if (first) first.focus();
                    });
                    editor.addEventListener("focusout", () => {
                        setTimeout(() => {
                            if (editor.contains(document.activeElement)) return;
                            render();   // re-derive both panes from current state
                        }, 0);
                    });
                    body.append(pane, editor);
                } else {
                    body.append(buildEditor(name, row));
                }
                const foot = buildCellFoot(name, row);
                if (foot) body.append(foot);
                td.append(body);
                tr.append(td);
            });
            tbody.append(tr);
        }

        if (!rows.length) {
            tbody.append(V.el("tr", {}, V.el("td", { colspan: names.length + 1, class: "empty" },
                "No components match the current filters")));
        }

        const table = V.el("table", { class: "cmp" }, buildHeader(names), tbody);
        mount.replaceChildren(table);
        mount.querySelectorAll("textarea").forEach(V.autogrow);

        diffBar.hidden = !diffMode;
        if (diffMode) {
            document.getElementById("diffSummary").textContent =
                `Diff mode — ${sameCount} identical, ${diffCount} different`;
        }
    }

    // ---------- new entry modal ----------

    const modal = document.getElementById("newEntryModal");

    function openModal() {
        const inherit = document.getElementById("newEntryParent");
        inherit.replaceChildren(V.el("option", { value: "" }, "— start empty —"),
            ...Object.keys(state.resolved).map(n => V.el("option", { value: n }, n)));
        const target = document.getElementById("newEntryFile");
        target.replaceChildren(...state.files.map(f => V.el("option", { value: f }, f)));
        document.getElementById("newEntryName").value = "";
        modal.hidden = false;
        document.getElementById("newEntryName").focus();
    }

    function createEntry() {
        const name = document.getElementById("newEntryName").value.trim();
        const parent = document.getElementById("newEntryParent").value;
        const file = document.getElementById("newEntryFile").value;
        if (!name) { alert("Enter an entry name."); return; }
        if (state.raw[name] !== undefined || state.resolved[name] !== undefined) {
            alert(`Entry "${name}" already exists.`);
            return;
        }
        state.raw[name] = parent ? { _extends: parent } : {};
        state.resolved[name] = parent ? V.clone(state.resolved[parent] ?? {}) : {};
        state.sourceFiles[name] = file || (parent ? state.sourceFiles[parent] : state.files[0]) || "";
        state.entries.clear();
        state.entries.add(name);
        if (parent) state.entries.add(parent);
        initFilters();
        render();
        V.setDirty(true);
        V.status(parent ? `Created "${name}" extending "${parent}"` : `Created empty entry "${name}"`, "ok");
        modal.hidden = true;
    }

    document.getElementById("newEntryBtn").addEventListener("click", openModal);
    document.getElementById("newEntryCancel").addEventListener("click", () => { modal.hidden = true; });
    document.getElementById("newEntryCreate").addEventListener("click", createEntry);
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.hidden = true; });

    // ---------- toolbar wiring ----------

    document.getElementById("search").addEventListener("input", render);
    document.getElementById("saveBtn").addEventListener("click", save);
    document.getElementById("reloadBtn").addEventListener("click", () => {
        if (V.isDirty() && !confirm("You have unsaved changes. Reload anyway?")) return;
        load();
    });
    document.getElementById("expandAll").addEventListener("click", () => {
        document.querySelectorAll("tr.row-same.collapsed").forEach(tr => tr.classList.remove("collapsed"));
    });
    document.getElementById("collapseAll").addEventListener("click", () => {
        document.querySelectorAll("tr.row-same").forEach(tr => tr.classList.add("collapsed"));
    });
    document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); save(); }
        if (e.key === "Escape") modal.hidden = true;
    });

    load();
})();
