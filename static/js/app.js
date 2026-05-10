// ===== State =====
let currentJobId = null;
let currentFile = null;
let pollTimer = null;
let selectedNotationTypes = new Set();

// ===== DOM refs =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const uploadArea = $("#upload-area");
const fileInput = $("#file-input");
const fileInfo = $("#file-info");
const fileName = $("#file-name");
const fileRemove = $("#file-remove");
const btnUpload = $("#btn-upload");
const progressContainer = $("#progress-container");
const progressFill = $("#progress-fill");
const progressText = $("#progress-text");
const settingsSection = $("#settings-section");
const analysisSummary = $("#analysis-summary");
const stemContainer = $("#stem-options-container");
const btnGenerate = $("#btn-generate");
const toast = $("#toast");

// ===== File upload UI =====
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("active");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("active");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("active");
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file || !file.name.toLowerCase().endsWith(".mp3")) {
        showToast("请选择 MP3 格式文件", "error");
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showToast("文件大小超过 50MB 限制", "error");
        return;
    }
    currentFile = file;
    fileInfo.style.display = "flex";
    fileName.textContent = file.name;
    btnUpload.disabled = false;
}

fileRemove.addEventListener("click", (e) => {
    e.stopPropagation();
    clearFile();
});

function clearFile() {
    currentFile = null;
    fileInfo.style.display = "none";
    btnUpload.disabled = true;
    fileInput.value = "";
}

// ===== Upload & analyze =====
btnUpload.addEventListener("click", async () => {
    if (!currentFile) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        if (!res.ok) throw new Error((await res.json()).detail || "上传失败");
        const data = await res.json();
        currentJobId = data.job_id;
        startPolling();
    } catch (err) {
        showToast(err.message, "error");
        setUploading(false);
    }
});

function setUploading(active) {
    if (active) {
        btnUpload.disabled = true;
        btnUpload.querySelector(".btn-text").style.display = "none";
        btnUpload.querySelector(".btn-loading").style.display = "flex";
        progressContainer.style.display = "block";
    } else {
        btnUpload.disabled = !currentFile;
        btnUpload.querySelector(".btn-text").style.display = "";
        btnUpload.querySelector(".btn-loading").style.display = "none";
    }
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 2000);
    pollStatus();
}

async function pollStatus() {
    if (!currentJobId) return;
    try {
        const res = await fetch(`/api/status/${currentJobId}`);
        if (!res.ok) return;
        const data = await res.json();

        progressFill.style.width = data.progress + "%";
        progressText.textContent = data.message;

        if (data.status === "done") {
            clearInterval(pollTimer);
            pollTimer = null;
            setUploading(false);
            showSettings(data);
            showToast("分析完成！请选择乐谱类型并生成", "success");
        } else if (data.status === "error") {
            clearInterval(pollTimer);
            pollTimer = null;
            setUploading(false);
            showToast(data.message, "error");
        }
    } catch (err) {
        // Silently retry
    }
}

// ===== Settings panel =====
function showSettings(data) {
    settingsSection.style.display = "block";

    // Fill analysis summary
    const keyDisplay = data.key + (data.key_mode === "major" ? " 大调" : " 小调");
    $("#sum-key").textContent = keyDisplay;
    $("#sum-bpm").textContent = data.bpm;
    $("#sum-ts").textContent = (data.time_signature || [4, 4]).join("/");

    const chordStr = (data.chords || []).slice(0, 8)
        .map((c) => `${c.degree}(${c.chord})`)
        .join(" → ");
    $("#sum-chords").textContent = chordStr || "—";

    analysisSummary.style.display = "flex";
    stemContainer.style.display = "block";

    // Show available stems
    updateStemOptions(data.stems || {});

    settingsSection.scrollIntoView({ behavior: "smooth" });
}

function updateStemOptions(stems) {
    $$("#stem-options .stem-option").forEach((opt) => {
        const val = opt.querySelector("input").value;
        if (stems[val]) {
            opt.style.display = "flex";
        } else {
            opt.style.display = "none";
            if (opt.classList.contains("selected")) {
                opt.classList.remove("selected");
                opt.querySelector("input").checked = false;
            }
        }
    });

    // Ensure at least one is selected
    const visible = [...$$("#stem-options .stem-option")].filter((o) => o.style.display !== "none");
    const hasSelected = visible.some((o) => o.classList.contains("selected"));
    if (!hasSelected && visible.length > 0) {
        visible[0].classList.add("selected");
        visible[0].querySelector("input").checked = true;
    }
}

// ===== Notation type selection =====
$$(".notation-option").forEach((opt) => {
    opt.addEventListener("click", () => {
        const cb = opt.querySelector("input");
        cb.checked = !cb.checked;
        opt.classList.toggle("selected", cb.checked);
        if (cb.checked) {
            selectedNotationTypes.add(opt.dataset.type);
        } else {
            selectedNotationTypes.delete(opt.dataset.type);
        }
        btnGenerate.disabled = selectedNotationTypes.size === 0;
    });
});

// ===== Stem selection =====
$$("#stem-options .stem-option").forEach((opt) => {
    opt.addEventListener("click", () => {
        $$("#stem-options .stem-option").forEach((o) => o.classList.remove("selected"));
        opt.classList.add("selected");
        opt.querySelector("input").checked = true;
    });
});

// ===== Generate score =====
const NOTATION_NAMES = { piano: "钢琴谱", guitar: "吉他谱", bass: "贝斯谱", drums: "架子鼓谱" };

btnGenerate.addEventListener("click", async () => {
    const types = [...selectedNotationTypes];
    if (types.length === 0) {
        showToast("请先选择乐谱类型", "error");
        return;
    }

    const stem = document.querySelector('input[name="stem"]:checked');
    const audioStem = stem ? stem.value : "other";

    setGenerating(true);

    const scoreLinks = $("#score-links");
    scoreLinks.style.display = "block";
    scoreLinks.innerHTML = "";

    let done = 0;
    for (const ntype of types) {
        const name = NOTATION_NAMES[ntype] || ntype;
        btnGenerate.querySelector(".btn-loading").innerHTML = `<span class="spinner"></span> 生成中 (${done + 1}/${types.length} ${name})…`;
        scoreLinks.insertAdjacentHTML("beforeend", `<span class="score-link loading" data-type="${ntype}">${name} 生成中…</span>`);

        const formData = new FormData();
        formData.append("job_id", currentJobId);
        formData.append("notation_type", ntype);
        formData.append("audio_stem", audioStem);

        try {
            const res = await fetch("/api/generate", {
                method: "POST",
                body: formData,
            });
            if (!res.ok) {
                let errMsg = "生成失败";
                try { errMsg = (await res.json()).detail || errMsg; } catch {}
                throw new Error(errMsg);
            }
            const data = await res.json();
            const linkEl = scoreLinks.querySelector(`[data-type="${ntype}"]`);
            linkEl.className = "score-link ready";
            linkEl.textContent = `📄 ${name}`;
            linkEl.addEventListener("click", () => window.open(data.score_url, "_blank"));
            done++;
        } catch (err) {
            const linkEl = scoreLinks.querySelector(`[data-type="${ntype}"]`);
            linkEl.className = "score-link error";
            linkEl.textContent = `${name}: 失败`;
            linkEl.title = err.message;
        }
    }

    setGenerating(false);
    if (done > 0) showToast(`已生成 ${done} 种乐谱，点击下方按钮查看`, "success");
});

function setGenerating(active) {
    if (active) {
        btnGenerate.disabled = true;
        btnGenerate.querySelector(".btn-text").style.display = "none";
        btnGenerate.querySelector(".btn-loading").style.display = "flex";
    } else {
        btnGenerate.disabled = selectedNotationTypes.size === 0;
        btnGenerate.querySelector(".btn-text").style.display = "";
        btnGenerate.querySelector(".btn-loading").style.display = "none";
        btnGenerate.querySelector(".btn-loading").innerHTML = '<span class="spinner"></span> 生成中...';
    }
}

// ===== Toast =====
function showToast(message, type = "info") {
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = "block";
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => {
        toast.style.display = "none";
    }, 3500);
}
