const form = document.getElementById("analysis-form");
const textInput = document.getElementById("text-input");
const fileInput = document.getElementById("file-input");
const fileName = document.getElementById("file-name");
const sampleButton = document.querySelector(".sample-button");
const resultsPanel = document.getElementById("results-panel");
const emptyState = document.getElementById("empty-state");

const scoreValue = document.getElementById("score-value");
const riskValue = document.getElementById("risk-value");
const modelName = document.getElementById("model-name");
const bestModel = document.getElementById("best-model");
const wordCount = document.getElementById("word-count");
const flaggedCount = document.getElementById("flagged-count");
const suggestionsList = document.getElementById("suggestions-list");
const sourcesList = document.getElementById("sources-list");
const passagesList = document.getElementById("passages-list");

fileInput.addEventListener("change", () => {
    fileName.textContent = fileInput.files.length ? fileInput.files[0].name : "No file selected";
});

sampleButton.addEventListener("click", () => {
    textInput.value = sampleButton.dataset.sample;
    fileInput.value = "";
    fileName.textContent = "Sample loaded";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = new FormData(form);
    const submitButton = form.querySelector(".primary-button");
    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: payload,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        renderResults(data.results);
        resultsPanel.classList.remove("hidden");
        emptyState.classList.add("hidden");
    } catch (error) {
        alert(error.message);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Analyze document";
    }
});

function renderResults(results) {
    scoreValue.textContent = `${results.plagiarism_score}%`;
    riskValue.textContent = results.risk_level;
    modelName.textContent = results.selected_model.toUpperCase();
    bestModel.textContent = `Best model: ${results.best_model.toUpperCase()}`;
    wordCount.textContent = results.metrics.word_count;
    flaggedCount.textContent = results.flagged_count;

    suggestionsList.innerHTML = results.suggestions
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join("");

    sourcesList.innerHTML = results.top_sources.length
        ? results.top_sources.map(renderSourceItem).join("")
        : "<p>No strong source matches found.</p>";

    passagesList.innerHTML = results.flagged_passages.length
        ? results.flagged_passages.map(renderPassageItem).join("")
        : "<p>No suspicious passages crossed the alert threshold.</p>";
}

function renderSourceItem(item) {
    return `
        <div class="source-item">
            <div class="match-meta">
                <span>ML ${item.ml_probability}%</span>
                <span>Cosine ${item.cosine_score}%</span>
                <span>Overlap ${item.token_overlap}%</span>
            </div>
            <p class="source-snippet">${escapeHtml(item.source_text)}</p>
        </div>
    `;
}

function renderPassageItem(item) {
    return `
        <div class="passage-item">
            <div class="match-meta">
                <span>ML ${item.ml_probability}%</span>
                <span>Cosine ${item.cosine_score}%</span>
                <span>Overlap ${item.token_overlap}%</span>
                <span>Combined ${item.combined_score}%</span>
            </div>
            <p><strong>Document passage:</strong> ${escapeHtml(item.passage)}</p>
            <p class="match-snippet"><strong>Closest source:</strong> ${escapeHtml(item.source_text)}</p>
        </div>
    `;
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}
