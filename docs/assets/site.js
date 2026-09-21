"use strict";

// Progressive enhancement: all result tables remain visible without JavaScript.
const datasetTabs = [...document.querySelectorAll("[data-dataset]")];
const datasetTablist = document.querySelector(".dataset-tabs");
function activateDataset(tab) {
  datasetTabs.forEach((button) => {
    const selected = button === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(button.getAttribute("aria-controls"));
    panel.hidden = !selected;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", button.id);
  });
}
if (datasetTablist && datasetTabs.length) {
  datasetTablist.hidden = false;
  activateDataset(datasetTabs[0]);
  datasetTabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateDataset(tab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
        return;
      event.preventDefault();
      const next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? datasetTabs.length - 1
            : (index +
                (event.key === "ArrowRight" ? 1 : -1) +
                datasetTabs.length) %
              datasetTabs.length;
      activateDataset(datasetTabs[next]);
      datasetTabs[next].focus();
    });
  });
}

// A deliberately synthetic illustration of the decoding formula, not a model
// demo, answer scorer, or benchmark evaluation. No requests or data uploads.
const original = [0.4, 0.35, 0.15, 0.1];
const perturbed = [0.55, 0.2, 0.15, 0.1];
const labels = ["A", "B", "C", "D"];
const slider = document.getElementById("alpha");
const chart = document.getElementById("token-chart");

for (let i = 0; i < labels.length; i += 1) {
  const row = document.createElement("div");
  row.className = "token-row";
  row.innerHTML = `<span class="token-name">Token ${labels[i]}</span><div class="bar-stack"><div class="bar-track"><div class="prob-bar original-bar" style="width:${original[i] * 100}%"></div></div><div class="bar-track"><div class="prob-bar vgs-bar" id="bar-${i}"></div></div></div><span class="token-value" id="value-${i}"></span>`;
  chart.appendChild(row);
}

function updateIllustration() {
  const alpha = Number(slider.value);
  const weights = original.map(
    (p, i) =>
      p *
      Math.max(
        1 + alpha * ((p - perturbed[i]) / (p + perturbed[i] + 1e-8)),
        0.01,
      ),
  );
  const total = weights.reduce((sum, value) => sum + value, 0);
  const probabilities = weights.map((value) => value / total);
  const chosen = probabilities.indexOf(Math.max(...probabilities));
  document.getElementById("alpha-value").value = alpha.toFixed(1);
  probabilities.forEach((p, i) => {
    document.getElementById(`bar-${i}`).style.width = `${p * 100}%`;
    document.getElementById(`value-${i}`).textContent =
      `${(p * 100).toFixed(1)}%`;
  });
  document.getElementById("selected-token").textContent = labels[chosen];
  document.getElementById("selected-note").textContent =
    alpha === 0 ? "original argmax" : "after reweighting";
  chart.setAttribute(
    "aria-label",
    `At guidance ${alpha.toFixed(1)}, reweighted probabilities are ${probabilities.map((p, i) => `token ${labels[i]} ${(p * 100).toFixed(1)} percent`).join(", ")}.`,
  );
}
slider.addEventListener("input", updateIllustration);
updateIllustration();

const tabs = [...document.querySelectorAll("[data-model]")];
function activateTab(tab) {
  const model = tab.dataset.model;
  tabs.forEach((button) => {
    const selected = button === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  document
    .getElementById("command-panel")
    .setAttribute("aria-labelledby", tab.id);
  document.getElementById("run-command").textContent =
    `python vgs_${model === "llava-med" ? "llavamed" : "medgemma"}_vqarad.py \\\n  --device cuda:0 --limit 3 \\\n  --output-dir outputs/${model}-smoke`;
  if (model === "medgemma") {
    const command = document.getElementById("run-command");
    command.textContent =
      '# Replace HF_Token locally, or use Hugging Face login.\nexport HF_TOKEN="HF_Token"\n\n' +
      command.textContent;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
            tabs.length;
    activateTab(tabs[next]);
    tabs[next].focus();
  });
});

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Clipboard unavailable");
}
document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const label = button.textContent;
    try {
      await copyText(document.getElementById(button.dataset.copy).textContent);
      button.textContent = "Copied!";
      document.getElementById("copy-status").textContent =
        "Copied to clipboard.";
    } catch (_) {
      button.textContent = "Select to copy";
      document.getElementById("copy-status").textContent =
        "Clipboard is unavailable. Select and copy the displayed text.";
    }
    setTimeout(() => {
      button.textContent = label;
    }, 2000);
  });
});
