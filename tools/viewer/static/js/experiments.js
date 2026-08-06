/* Experiments page: experiments as columns, flattened leaf fields as rows —
   built for spotting single-field divergence between near-identical configs.
   raw = the file's own yaml (explicit overrides + what saves round-trip);
   resolved = _includes/_extends applied and _models/_preset expanded. */
"use strict";

(() => {
    const KEY_SELECTED = "cv.experiments.selected";
    const KEY_FILE = "cv.experiments.file";

    const state = {
        raw: null,
        resolved: null,
        models: {},
        file: null,
        mtime: null,
        selected: new Set(V.loadStored(KEY_SELECTED, [])),
        rows: [],
    };

    const persist = () => {
        V.store(KEY_SELECTED, [...state.selected]);
        V.store(KEY_FILE, state.file);
    };

    const experimentSelect = V.multiSelect({
        mount: document.getElementById("experimentSelect"),
        emptyLabel: "Select experiments…",
        selected: state.selected,
        onChange: () => { persist(); render(); },
    });

    const parentOf = (name) => (V.isDict(state.raw[name]) ? state.raw[name]._extends : undefined);

    // ---------- row derivation ----------
    //
    // Union of leaf paths over the selected experiments' resolved AND raw
    // views (raw contributes `_preset`, resolved the expanded preset fields),
    // in first-seen key order so the table mirrors the YAML's own layout.

    function buildRows(names) {
        const rows = [];
        const walk = (values, path) => {
            const dicts = values.filter(V.isDict);
            const arrays = values.filter(Array.isArray);
            const scalars = values.filter(v => v !== undefined && !V.isDict(v) && !Array.isArray(v));
            if (dicts.length || arrays.length) {
                // Mixed shapes at this path (container in some experiments,
                // scalar in others): the scalar side must still get a row —
                // divergence-spotting is the whole point. Container-valued
                // columns render "(nested)" in that row; recursion into the
                // container keys continues as normal below.
                const mixed = scalars.length > 0 || (dicts.length > 0 && arrays.length > 0);
                if (mixed && path.length) rows.push({ path, mixed: true });
                if (dicts.length) {
                    const keys = [];
                    for (const d of dicts) for (const k of Object.keys(d)) if (!keys.includes(k)) keys.push(k);
                    for (const k of keys) {
                        if (path.length === 0 && k === "_extends") continue;  // column chip instead
                        walk(values.map(v => (V.isDict(v) ? v[k] : undefined)), [...path, k]);
                    }
                } else {
                    const len = Math.max(...arrays.map(a => a.length));
                    for (let i = 0; i < len; i++) {
                        walk(values.map(v => (Array.isArray(v) ? v[i] : undefined)), [...path, String(i)]);
                    }
                }
            } else if (path.length) {
                rows.push({ path });
            }
        };
        walk(names.map(n => state.resolved[n]).concat(names.map(n => state.raw[n])), []);
        return rows;
    }

    const isPresetRow = (row) => row.path[row.path.length - 1] === "_preset";

    function displayValue(v) {
        if (v === undefined) return "";
        if (v === null) return "null";
        if (typeof v === "object") return JSON.stringify(v);
        return String(v);
    }

    // An emptied text field is the EMPTY STRING, never a key deletion: the
    // only way to remove a key is the cell's ✕ button (clearCell) or a
    // select's blank option, so a value can be cleared without the row
    // disappearing under the cursor.
    function parseValue(text, reference) {
        if (text === "") return "";
        // A string-typed field stays a string: typing "true"/"1" into it must
        // not coerce (the reference is the field's current resolved value).
        if (typeof reference === "string") return text;
        if (text === "true" || text === "false") return text === "true";
        if (text === "null") return null;
        const numeric = text.trim() !== "" && !Number.isNaN(Number(text));
        if (numeric && (typeof reference === "number" || reference === undefined)) return Number(text);
        return text;
    }

    // ---------- load / save ----------

    async function loadFiles() {
        try {
            const payload = await V.api("/api/experiments/files");
            const picker = document.getElementById("filePicker");
            picker.replaceChildren(...payload.files.map(f => V.el("option", { value: f }, f)));
            if (!payload.files.length) {
                V.status("No config files found", "err");
                return;
            }
            const remembered = V.loadStored(KEY_FILE, null);
            picker.value = payload.files.includes(remembered) ? remembered : payload.default;
            await loadFile(picker.value);
        } catch (err) {
            V.status(`Error listing files: ${err.message}`, "err");
        }
    }

    async function loadFile(filename) {
        try {
            const payload = await V.api(`/api/experiments/file/${encodeURIComponent(filename)}`);
            state.raw = payload.raw;
            state.resolved = payload.resolved;
            state.models = payload.models || {};
            state.mtime = payload.mtime;
            state.file = filename;
            persist();
            refreshExperimentOptions();
            render();
            renderModels();
            V.setDirty(false);
            V.status(`Loaded ${filename}`, "ok");
        } catch (err) {
            V.status(`Error loading ${filename}: ${err.message}`, "err");
        }
    }

    async function save() {
        if (!state.file) return;
        if (!V.isDirty()) { V.status("No changes to save"); return; }
        if (!confirm(V.SAVE_WARNING)) return;
        const saveBtn = document.getElementById("saveBtn");
        try {
            saveBtn.disabled = true;
            V.status("Saving…");
            const result = await V.api(`/api/experiments/file/${encodeURIComponent(state.file)}`,
                { raw: state.raw, mtime: state.mtime });
            if (result.status === "no_changes") {
                V.status(result.message);
            } else {
                state.mtime = result.mtime;
                V.setDirty(false);
                V.status(`${result.message} (backup: ${result.backup})`, "ok");
            }
        } catch (err) {
            V.status(`Error saving: ${err.message}`, "err");
        } finally {
            saveBtn.disabled = false;
        }
    }

    // ---------- experiment picker ----------

    function refreshExperimentOptions() {
        const term = document.getElementById("search").value.trim().toLowerCase();
        const all = Object.keys(state.resolved || {}).filter(n => n !== "_models");
        const names = all.filter(n => !term || n.toLowerCase().includes(term));
        // valid set = ALL experiments: a search filter must not drop selections
        experimentSelect.setOptions(names.map(value => {
            const parent = parentOf(value);
            return { value, sub: parent !== undefined ? `extends ${parent}` : undefined };
        }), new Set(all));
    }

    // ---------- editing ----------

    function onEdit(target, name, row) {
        const reference = V.getPath(state.resolved[name], row.path);
        // A select's blank option ("—" / "— no preset —") IS its clear
        // affordance; text fields clear via the cell's ✕ button instead.
        const cleared = target.tagName === "SELECT" && target.value === "";
        const value = cleared ? undefined : parseValue(target.value, reference);
        const parent = parentOf(name);
        const inheritedValue = parent !== undefined ? V.getPath(state.resolved[parent], row.path) : undefined;
        const preset = isPresetRow(row);
        let structural = false;

        if (!V.isDict(state.raw[name])) state.raw[name] = {};
        if (value === undefined || (!preset && value === inheritedValue)) {
            structural = V.deletePath(state.raw[name], row.path);
            if (inheritedValue !== undefined) V.setPath(state.resolved[name], row.path, inheritedValue);
            // no parent value: the resolved leaf must go too — a stale value
            // styled as "inherited" would misreport the config.
            else structural = V.deletePath(state.resolved[name], row.path) || structural;
        } else {
            V.setPath(state.raw[name], row.path, value);
            if (!preset) V.setPath(state.resolved[name], row.path, value);
        }
        if (preset) V.status("Preset changed — save, then reload to see the expanded fields");
        else {
            const kids = Object.keys(state.raw).filter(n => parentOf(n) === name);
            if (kids.length) V.status(
                `"${name}" is extended by ${kids.join(", ")} — their resolved views refresh after save + reload`);
        }

        V.setDirty(true);
        if (structural) {
            // An array element was spliced out: every later index shifted, so
            // the row paths are stale — rebuild them before the next edit
            // writes to the wrong element (deep_merge merges lists by index).
            render();
            return;
        }

        const nowInherited = V.getPath(state.raw[name], row.path) === undefined
            && V.getPath(state.resolved[name], row.path) !== undefined;
        target.classList.toggle("inh", nowInherited);
        const td = target.closest("td");
        if (td) td.classList.toggle("cell-inh", nowInherited);
    }

    // Explicit key removal (the ✕ in a cell). Always re-renders: removing an
    // array element renumbers its siblings, and removing a dict key can prune
    // empty parents, so the row list must be rebuilt either way.
    function clearCell(name, row) {
        const parent = parentOf(name);
        const inheritedValue = parent !== undefined
            ? V.getPath(state.resolved[parent], row.path) : undefined;
        V.deletePath(state.raw[name], row.path);
        if (inheritedValue !== undefined) V.setPath(state.resolved[name], row.path, inheritedValue);
        else V.deletePath(state.resolved[name], row.path);
        V.setDirty(true);
        render();
    }

    // Mixed-shape row (R5): a column whose value at this path is a container
    // shows a read-only "(nested)" placeholder — its real rows are the child
    // paths — while scalar-valued columns keep their editable cell.
    function isNestedInCell(name, row) {
        if (!row.mixed) return false;
        const v = V.getPath(state.resolved[name], row.path) !== undefined
            ? V.getPath(state.resolved[name], row.path)
            : V.getPath(state.raw[name], row.path);
        return V.isDict(v) || Array.isArray(v);
    }

    function buildInput(name, row) {
        if (isNestedInCell(name, row)) {
            return V.el("span", { class: "nested-note", title: "Container value — edit its child rows" }, "(nested)");
        }
        const resolvedValue = V.getPath(state.resolved[name], row.path);
        const rawValue = V.getPath(state.raw[name], row.path);
        const value = resolvedValue !== undefined ? resolvedValue : rawValue;
        const inherited = rawValue === undefined;
        const cls = `cell-input${inherited ? " inh" : ""}`;

        let node;
        if (isPresetRow(row)) {
            node = V.el("select", { class: cls },
                V.el("option", { value: "" }, "— no preset —"),
                Object.keys(state.models).map(m => V.el("option", { value: m, selected: m === value }, m)));
            node.value = typeof value === "string" ? value : "";
        } else if (typeof value === "boolean") {
            node = V.el("select", { class: cls },
                V.el("option", { value: "" }, "—"),
                V.el("option", { value: "true" }, "true"),
                V.el("option", { value: "false" }, "false"));
            node.value = String(value);
        } else {
            node = V.el("input", { type: "text", class: cls, spellcheck: "false" });
            node.value = displayValue(value);
        }
        node.addEventListener(node.tagName === "SELECT" ? "change" : "input", () => onEdit(node, name, row));
        return node;
    }

    // ---------- table ----------

    function duplicateExperiment(name) {
        const newName = prompt("Name for the duplicate:", `${name}_copy`);
        if (!newName) return;
        if (state.raw[newName] !== undefined) { alert("That name already exists."); return; }
        state.raw[newName] = V.clone(state.raw[name] ?? {});
        state.resolved[newName] = V.clone(state.resolved[name] ?? {});
        state.selected.add(newName);
        persist();
        refreshExperimentOptions();
        render();
        V.setDirty(true);
    }

    function deleteExperiment(name) {
        if (!(name in state.raw)) {
            alert(`"${name}" is defined in an include — cannot delete it here. Edit its own file.`);
            return;
        }
        if (!confirm(`Delete experiment "${name}" from ${state.file}?`)) return;
        delete state.raw[name];
        delete state.resolved[name];
        state.selected.delete(name);
        persist();
        refreshExperimentOptions();
        render();
        V.setDirty(true);
    }

    function render() {
        const mount = document.getElementById("tableMount");
        const names = [...state.selected].filter(
            n => state.resolved[n] !== undefined || state.raw[n] !== undefined);

        if (!names.length) {
            mount.replaceChildren(V.el("div", { class: "empty" }, "Select experiments to compare"));
            return;
        }

        state.rows = buildRows(names);

        const head = V.el("thead", {}, V.el("tr", {},
            V.el("th", { class: "rowhead" }, "Field"),
            names.map(name => {
                const sub = V.el("div", { class: "col-sub" });
                const parent = parentOf(name);
                if (parent !== undefined) sub.append(V.el("span", { class: "chip extends" }, `extends ${parent}`));
                sub.append(
                    V.el("button", { class: "btn tiny", onclick: () => duplicateExperiment(name) }, "Duplicate"),
                    V.el("button", { class: "btn tiny danger", onclick: () => deleteExperiment(name) }, "Delete"),
                );
                return V.el("th", {}, V.el("div", { class: "col-title" }, name), sub);
            })));

        const tbody = V.el("tbody");
        let lastSection = null;
        for (const row of state.rows) {
            const top = row.path[0];
            if (row.path.length > 1 && top !== lastSection) {
                tbody.append(V.el("tr", { class: "group-row" },
                    V.el("td", { colspan: names.length + 1 }, top)));
                lastSection = top;
            } else if (row.path.length === 1) {
                lastSection = null;
            }

            const depth = Math.min(row.path.length - 1, 3);
            const label = row.path.length === 1 ? row.path[0] : row.path.slice(1).join(".");
            const tr = V.el("tr", {},
                V.el("td", { class: "rowhead" },
                    V.el("span", { class: depth ? `indent-${depth}` : "" }, label)));
            for (const name of names) {
                const nested = isNestedInCell(name, row);
                const inherited = !nested && V.getPath(state.raw[name], row.path) === undefined;
                const td = V.el("td", { class: `cell${inherited ? " cell-inh" : ""}` },
                    buildInput(name, row));
                // Only an explicit ✕ deletes a key — emptying the text field
                // enters "". Shown when this file actually holds the key.
                if (!nested && V.getPath(state.raw[name], row.path) !== undefined) {
                    td.append(V.el("button", {
                        class: "clear-btn",
                        type: "button",
                        title: "Remove this key from the file",
                        onclick: () => clearCell(name, row),
                    }, "✕"));
                }
                tr.append(td);
            }
            tbody.append(tr);
        }

        mount.replaceChildren(V.el("table", { class: "cmp" }, head, tbody));
    }

    // ---------- model presets panel ----------

    function presetLeafPaths(config) {
        const out = [];
        const walk = (v, path) => {
            if (V.isDict(v)) for (const k of Object.keys(v)) walk(v[k], [...path, k]);
            else if (Array.isArray(v)) v.forEach((item, i) => walk(item, [...path, String(i)]));
            else out.push(path);
        };
        walk(config, []);
        return out;
    }

    function renderModels() {
        const body = document.getElementById("modelsBody");
        // Editability is PER PRESET: only presets present in THIS file's raw
        // _models are editable. Included-only presets render read-only but
        // stay visible (and stay in the _preset dropdown, which reads
        // state.models — never overwrite state.models with raw._models).
        const rawModels = V.isDict(state.raw) && V.isDict(state.raw._models) ? state.raw._models : null;
        const isEditable = (presetName) => !!(rawModels && V.isDict(rawModels[presetName]));
        const presetNames = Object.keys(state.models);
        const children = [];

        if (!presetNames.length) {
            children.push(V.el("p", { class: "models-note" }, "No model presets in this file."));
        } else if (presetNames.some(n => !isEditable(n))) {
            children.push(V.el("p", { class: "models-note" },
                "Presets from an included file are shown read-only; edit them there."));
        }

        for (const [presetName, config] of Object.entries(state.models)) {
            const editable = isEditable(presetName);
            const card = V.el("div", { class: "preset-card" },
                V.el("h4", {}, presetName,
                    editable ? null : V.el("span", { class: "chip file" }, "included")));
            for (const path of presetLeafPaths(config)) {
                const input = V.el("input", { type: "text", spellcheck: "false", disabled: !editable });
                input.value = displayValue(V.getPath(config, path));
                if (editable) input.addEventListener("input", () => {
                    const reference = V.getPath(state.models[presetName], path);
                    const value = parseValue(input.value, reference);
                    V.setPath(rawModels[presetName], path, value);
                    V.setPath(state.models[presetName], path, value);
                    V.setDirty(true);
                });
                card.append(V.el("div", { class: "preset-field" },
                    V.el("label", {}, path.join(".")), input));
            }
            children.push(card);
        }
        body.replaceChildren(...children);
    }

    // ---------- toolbar wiring ----------

    document.getElementById("filePicker").addEventListener("change", (e) => {
        if (V.isDirty() && !confirm("You have unsaved changes. Switch files anyway?")) {
            e.target.value = state.file;
            return;
        }
        state.selected.clear();
        loadFile(e.target.value);
    });
    document.getElementById("search").addEventListener("input", refreshExperimentOptions);
    document.getElementById("saveBtn").addEventListener("click", save);
    document.getElementById("reloadBtn").addEventListener("click", () => {
        if (V.isDirty() && !confirm("You have unsaved changes. Reload anyway?")) return;
        loadFile(state.file);
    });
    document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); save(); }
    });

    loadFiles();
})();
