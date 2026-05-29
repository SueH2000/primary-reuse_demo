const state = {
  currentResult: null,
  batchResults: [],
  currentBatchSelectedIndex: -1,
  batchReviewStatus: {},
  currentInputText: "",
  currentInputTitle: "",
  currentInputIdentifier: "",
  currentBatchJobId: "",
  currentImportJobId: "",
  currentGenericFetchJobId: "",
  autoActivateImport: false,
  autoActivateGenericFetch: false,
  serverJobs: { active_jobs: [], recent_jobs: [] },
};


const MODE_LABELS = {
  linear_only: "Linear only",
  hybrid_baseline: "Hybrid baseline",
  rag_vote_only: "RAG vote only",
  llm_sentence_judge_routed: "Sentence reviewer, routed",
  llm_sentence_judge_force: "Sentence reviewer, force",
  llm_final_classify_routed: "LLM classifier, routed",
  llm_final_classify_force: "LLM classifier, force",
  pipeline_final: "Pipeline final",
};

const REVIEWER_LABELS = {
  none: "No LLM reviewer",
  llm_sentence_judge_routed: "Sentence reviewer, routed",
  llm_sentence_judge_force: "Sentence reviewer, force",
  llm_final_classify_routed: "LLM classifier, routed",
  llm_final_classify_force: "LLM classifier, force",
};

function modeLabel(mode) {
  const value = String(mode || "");
  if (value.includes("__")) {
    const [base, reviewer] = value.split("__");
    return `${modeLabel(base)} + ${modeLabel(reviewer)}`;
  }
  return MODE_LABELS[value] || REVIEWER_LABELS[value] || value || "-";
}

const ACTIVE_JOB_STORAGE_KEY = "primary-reuse-active-jobs";

function byId(id) {
  return document.getElementById(id);
}

function setStatus(kind, text) {
  const pill = byId("statusPill");
  const msg = byId("statusText");
  pill.className = "status-pill";
  if (kind === "success") pill.classList.add("success");
  if (kind === "error") pill.classList.add("error");
  pill.textContent = kind;
  msg.textContent = text;
}

function loadActiveJobsFromStorage() {
  try {
    return JSON.parse(window.localStorage.getItem(ACTIVE_JOB_STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveActiveJobsToStorage(payload) {
  window.localStorage.setItem(ACTIVE_JOB_STORAGE_KEY, JSON.stringify(payload));
}

function markActiveJob(kind, jobId) {
  const payload = loadActiveJobsFromStorage();
  payload[kind] = jobId || "";
  saveActiveJobsToStorage(payload);
}

function clearActiveJob(kind) {
  const payload = loadActiveJobsFromStorage();
  delete payload[kind];
  saveActiveJobsToStorage(payload);
}

function clearJobProgress() {
  byId("jobProgressShell")?.classList.add("hidden");
  if (byId("jobProgressLabel")) byId("jobProgressLabel").textContent = "No active job";
  if (byId("jobProgressValue")) byId("jobProgressValue").textContent = "0%";
  if (byId("jobProgressFill")) byId("jobProgressFill").style.width = "0%";
}

function updateJobProgress(label, percent, detail = "") {
  const shell = byId("jobProgressShell");
  const labelEl = byId("jobProgressLabel");
  const valueEl = byId("jobProgressValue");
  const fillEl = byId("jobProgressFill");
  if (!shell || !labelEl || !valueEl || !fillEl) return;
  const pct = Math.max(0, Math.min(100, Number(percent || 0)));
  shell.classList.remove("hidden");
  labelEl.textContent = detail ? `${label} | ${detail}` : label;
  valueEl.textContent = `${pct.toFixed(1)}%`;
  fillEl.style.width = `${pct}%`;
}

function jobKindLabel(kind) {
  if (kind === "batch") return "Batch classification";
  if (kind === "mohammad") return "Mohammad import";
  if (kind === "fetch") return "Article fetch";
  return "Job";
}

function clearAllActiveJobs() {
  saveActiveJobsToStorage({});
}

function reconnectServerJob(jobKind, jobId) {
  clearAllActiveJobs();
  if (jobKind === "batch") {
    state.currentBatchJobId = jobId;
    markActiveJob("batch", jobId);
    switchTab("upload");
    byId("batchSummary").textContent = `Reconnected to server batch job ${jobId}.`;
    pollBatchJob(jobId).catch((error) => {
      setStatus("error", `Batch job reconnect failed: ${error.message}`);
    });
    return;
  }
  if (jobKind === "mohammad") {
    state.currentImportJobId = jobId;
    markActiveJob("mohammad", jobId);
    switchTab("mohammad");
    byId("mohammadSummary").textContent = `Reconnected to server Mohammad import job ${jobId}.`;
    pollMohammadImportJob(jobId).catch((error) => {
      setStatus("error", `Mohammad import reconnect failed: ${error.message}`);
    });
    return;
  }
  if (jobKind === "fetch") {
    state.currentGenericFetchJobId = jobId;
    markActiveJob("fetch", jobId);
    switchTab("mohammad");
    byId("genericFetchSummary").textContent = `Reconnected to server article fetch job ${jobId}.`;
    pollGenericFetchJob(jobId).catch((error) => {
      setStatus("error", `Article fetch reconnect failed: ${error.message}`);
    });
  }
}

async function cancelServerJob(jobKind, jobId) {
  setStatus("running", `Cancelling ${jobKind} job ${jobId}.`);
  try {
    await fetch(`/server_jobs/${jobKind}/${jobId}/cancel`, {
      method: "POST",
    }).then(parseJsonResponse);
    if (jobKind === "batch" && state.currentBatchJobId === jobId) {
      state.currentBatchJobId = "";
      clearActiveJob("batch");
    }
    if (jobKind === "mohammad" && state.currentImportJobId === jobId) {
      state.currentImportJobId = "";
      clearActiveJob("mohammad");
    }
    if (jobKind === "fetch" && state.currentGenericFetchJobId === jobId) {
      state.currentGenericFetchJobId = "";
      clearActiveJob("fetch");
    }
    clearJobProgress();
    await loadServerJobs();
    setStatus("success", `Cancelled ${jobKind} job ${jobId}.`);
  } catch (error) {
    setStatus("error", `Cancel failed: ${error.message}`);
  }
}

function serverJobCard(job) {
  const item = document.createElement("div");
  item.className = "server-job-item";
  const displayStatus = String(job.display_status || job.status || "-");
  const isActive = ["queued", "running", "cancelling"].includes(String(job.status || "")) && Boolean(job.live_managed);
  const canReconnect = isActive;
  const canCancel = !["completed", "failed", "cancelled"].includes(String(job.status || ""));
  const processed = Number(job.processed ?? 0);
  const total = Number(job.total ?? job.count ?? job.requested_identifiers ?? 0);
  const subtitle = total
    ? `${displayStatus} | ${processed}/${total} | ${formatJobProgress(job)}`
    : `${displayStatus} | updated ${job.updated_at || "-"}`;
  item.innerHTML = `
    <div class="server-job-top">
      <div>
        <p class="server-job-title">${escapeHtml(jobKindLabel(job.job_kind))} <code>${escapeHtml(job.job_id || "-")}</code></p>
        <div class="server-job-subtitle">${escapeHtml(subtitle)}</div>
      </div>
    </div>
    <div class="server-job-badges">
      <span class="server-job-badge">${escapeHtml(job.job_kind || "-")}</span>
      <span class="server-job-badge ${job.live_managed ? "live" : "stale"}">${job.live_managed ? "live" : "stale"}</span>
    </div>
    <div class="server-job-actions"></div>
  `;
  const actions = item.querySelector(".server-job-actions");
  if (canReconnect) {
    const reconnectBtn = document.createElement("button");
    reconnectBtn.className = "ghost ghost-small";
    reconnectBtn.type = "button";
    reconnectBtn.textContent = "Reconnect";
    reconnectBtn.addEventListener("click", () => reconnectServerJob(String(job.job_kind || ""), String(job.job_id || "")));
    actions.appendChild(reconnectBtn);
  }
  if (canCancel) {
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost ghost-small";
    cancelBtn.type = "button";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", () => cancelServerJob(String(job.job_kind || ""), String(job.job_id || "")));
    actions.appendChild(cancelBtn);
  }
  return item;
}

function renderServerJobs(payload) {
  state.serverJobs = payload || { active_jobs: [], recent_jobs: [] };
  const activeHost = byId("serverJobsActive");
  const recentHost = byId("serverJobsRecent");
  activeHost.innerHTML = "";
  recentHost.innerHTML = "";
  const activeJobs = Array.isArray(payload?.active_jobs) ? payload.active_jobs : [];
  const recentJobs = Array.isArray(payload?.recent_jobs) ? payload.recent_jobs : [];
  if (!activeJobs.length) {
    const empty = document.createElement("div");
    empty.className = "server-job-empty";
    empty.textContent = "No live server-side jobs.";
    activeHost.appendChild(empty);
  } else {
    activeJobs.forEach((job) => activeHost.appendChild(serverJobCard(job)));
  }
  if (!recentJobs.length) {
    const empty = document.createElement("div");
    empty.className = "server-job-empty";
    empty.textContent = "No recent jobs.";
    recentHost.appendChild(empty);
  } else {
    recentJobs.forEach((job) => recentHost.appendChild(serverJobCard(job)));
  }
}

async function loadServerJobs() {
  const payload = await fetch("/server_jobs").then(parseJsonResponse);
  renderServerJobs(payload);
  return payload;
}

function formatConfidence(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatList(items) {
  return Array.isArray(items) && items.length ? items : ["None"];
}

function renderChips(targetId, items, secondary = false) {
  const target = byId(targetId);
  target.innerHTML = "";
  for (const item of formatList(items)) {
    const span = document.createElement("span");
    span.className = secondary ? "chip secondary" : "chip";
    span.textContent = item;
    target.appendChild(span);
  }
}

function renderGseLinks(urls) {
  const target = byId("gseLinks");
  target.innerHTML = "";
  if (!Array.isArray(urls) || !urls.length) {
    target.textContent = "";
    return;
  }
  for (const url of urls) {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = url;
    target.appendChild(a);
  }
}

function renderMainDecisionGse(result) {
  const ids = result?.evidence?.main_decision_gse_ids || [];
  const urls = result?.evidence?.main_decision_gse_urls || [];
  byId("mainDecisionGseLabel").textContent = ids.length
    ? `Main decision GSE: ${ids.join(", ")}`
    : "Main decision GSE: -";
  const target = byId("mainDecisionGseLinks");
  target.innerHTML = "";
  for (const url of urls) {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = url;
    target.appendChild(a);
  }
}

function resultPaperLabel(result) {
  return result.paper_id || result.lookup?.paper_id || result.lookup?.identifier || result.title || "Unspecified";
}

function routeLookupStatus(result) {
  if (result.found === false) return "Not found in local index";
  if (result.lookup?.identifier) return `Resolved from ${result.lookup.identifier}`;
  return "Direct text classification";
}

function renderDecisionPolicy(result) {
  const banner = byId("decisionPolicyBanner");
  if (!banner) return;
  const audit = result?.decision_audit || {};
  const finalSource = result?.final?.source || "-";
  const finalLabel = result?.final?.label || "-";
  banner.classList.remove("policy-locked", "policy-llm", "policy-baseline");

  if (audit.llm_advisory_only || audit.llm_override_lock_applied) {
    banner.classList.add("policy-locked");
    banner.textContent = `High-confidence baseline lock active. Final label stays ${finalLabel} from ${finalSource}; LLM output is advisory only.`;
    return;
  }
  if (audit.llm_used_for_final) {
    banner.classList.add("policy-llm");
    banner.textContent = `LLM review contributed to the final label (${finalLabel}). Check the LLM audit before trusting the change.`;
    return;
  }
  banner.classList.add("policy-baseline");
  banner.textContent = `Stable decision source: ${finalSource}. LLM output is optional reviewer evidence, not the default authority.`;
}

function updateFeedbackDefaults(result) {
  const predicted = result?.final?.label || "";
  byId("feedbackPredictedLabel").value = predicted;
  if (predicted) {
    byId("feedbackCorrectedLabel").value = predicted;
  }
  byId("feedbackResult").textContent = "";
  if (byId("feedbackEvidenceSentence")) {
    byId("feedbackEvidenceSentence").value = "";
  }
  const feedbackCard = document.querySelector(".feedback-card");
  feedbackCard.classList.remove("feedback-confirmed", "feedback-corrected");
}

function renderNeighbors(neighbors) {
  const table = byId("neighborsTable");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  for (const neighbor of neighbors || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(neighbor.paper_id || "-")}</td>
      <td>${escapeHtml(neighbor.label || "-")}</td>
      <td>${Number(neighbor.score || 0).toFixed(3)}</td>
      <td>${escapeHtml((neighbor.snippet || "").slice(0, 220) || "-")}</td>
    `;
    tbody.appendChild(tr);
  }
}

function updateRagGroundingVisibility(result) {
  const ragEnabled = Boolean(result?.rag?.llm_rag_context_enabled);
  const neighborCard = byId("neighborDetailCard");
  const ragContextCard = byId("ragContextDetailCard");
  const neighborTitle = byId("neighborDetailTitle");
  const ragContextTitle = byId("ragContextDetailTitle");

  if (neighborTitle) neighborTitle.textContent = ragEnabled ? "LLM grounding neighbors" : "LLM grounding neighbors not sent";
  if (ragContextTitle) ragContextTitle.textContent = ragEnabled ? "LLM RAG prompt context" : "LLM RAG prompt context disabled";

  // Important UI rule:
  // Retrieved neighbors can still exist internally for hybrid/RAG baselines, but when
  // the user disables "Send retrieved labeled examples to the LLM reviewer prompt",
  // the right inspection panel should not show them as LLM grounding context.
  if (neighborCard) neighborCard.classList.toggle("hidden", !ragEnabled);
  if (ragContextCard) ragContextCard.classList.toggle("hidden", !ragEnabled);
}

function displayedModeLabel(result) {
  const mode = result?.displayed_mode || result?.selected_mode || "";
  if (mode) return MODE_LABELS_SHORT[mode] || mode;
  const modes = result?.selected_modes || selectedModelModes();
  if (Array.isArray(modes) && modes.length === 1) return MODE_LABELS_SHORT[modes[0]] || modes[0];
  if (result?.final?.source === "linear_model_plus_rag") return "Hybrid baseline";
  if (result?.final?.source === "linear_only") return "Linear only";
  if (result?.final?.source === "rag_vote") return "RAG vote only";
  if (result?.decision_audit?.llm_strategy === "sentence_judge") return "Sentence reviewer";
  if (result?.decision_audit?.llm_strategy === "classify" && result?.decision_audit?.llm_requested) return "LLM classifier";
  return "Current result";
}

function updateMultiModeNotice(result) {
  const notice = byId("multiModeNotice");
  if (!notice) return;
  const modes = result?.selected_modes || [];
  if (Array.isArray(modes) && modes.length > 1) {
    const label = displayedModeLabel(result);
    notice.classList.remove("hidden");
    notice.textContent = `Multiple modes selected. Inspecting: ${label}. Click any comparison card below to switch the full audit view.`;
  } else {
    notice.classList.add("hidden");
  }
}

function renderResult(result) {
  state.currentResult = result;
  updateFeedbackDefaults(result);

  if (result?.comparison_results && result?.selected_modes) {
    renderStrategyComparison({
      requested_modes: result.selected_modes,
      results: result.comparison_results,
    });
  }

  byId("finalLabel").textContent = result?.final?.label || "-";
  byId("finalSource").textContent = result?.final?.source || "-";
  byId("hybridConfidence").textContent = formatConfidence(result?.predictions?.linear_model_plus_rag_conf);
  byId("recommendedRoute").textContent = result?.recommended_route || "-";
  renderDecisionPolicy(result);
  if (byId("displayedMode")) byId("displayedMode").textContent = displayedModeLabel(result);
  updateMultiModeNotice(result);
  byId("detailPaperId").textContent = result.paper_id || result.lookup?.paper_id || "-";
  byId("detailTitle").textContent = result.title || "-";
  byId("detailRouteReason").textContent = result.recommended_route_reason || "-";
  byId("detailLookupStatus").textContent = routeLookupStatus(result);
  byId("mainDecisionSentence").textContent = result?.evidence?.main_decision_sentence || "No main decision sentence found.";
  byId("mainDecisionRole").textContent = result?.evidence?.main_decision_role || "-";
  byId("predLinear").textContent = result?.predictions?.linear_model || "-";
  byId("predHybrid").textContent = result?.predictions?.linear_model_plus_rag || "-";
  byId("predRagVote").textContent = result?.predictions?.rag_vote || "-";
  byId("predRagMargin").textContent = result?.predictions?.rag_vote_margin?.toFixed?.(3) || Number(result?.predictions?.rag_vote_margin || 0).toFixed(3);
  const audit = result?.decision_audit || {};
  const llm = result?.llm || {};
  byId("llmRequested").textContent = audit.llm_requested ? "Yes" : "No";
  byId("llmCalled").textContent = audit.llm_called ? "Yes" : "No";
  byId("llmValid").textContent = audit.llm_valid ? "Yes" : "No";
  byId("llmStrategy").textContent = audit.llm_strategy || llm.strategy || "-";
  byId("llmProposedLabel").textContent = audit.llm_proposed_label || "-";
  byId("llmConfidence").textContent = formatConfidence(audit.llm_confidence);
  byId("llmUsedForFinal").textContent = audit.llm_used_for_final ? "Yes" : "No";
  byId("llmOverrideStatus").textContent = audit.override_status || "-";
  if (byId("llmAdvisoryOnly")) byId("llmAdvisoryOnly").textContent = audit.llm_advisory_only ? "Yes" : "No";
  if (byId("llmOverrideLockApplied")) byId("llmOverrideLockApplied").textContent = audit.llm_override_lock_applied ? "Yes" : "No";
  if (byId("llmOverrideLockThreshold")) byId("llmOverrideLockThreshold").textContent = audit.llm_override_lock_threshold ?? "-";
  if (byId("llmRagContextSent")) byId("llmRagContextSent").textContent = result?.rag?.llm_rag_context_enabled ? "Yes" : "No";
  if (byId("llmRagContextTopK")) byId("llmRagContextTopK").textContent = result?.rag?.llm_rag_top_k ?? "-";
  const llmMessage = !audit.llm_requested
    ? "This run did not request the LLM."
    : !audit.llm_called
      ? "The LLM was requested but routing did not call it."
      : audit.override_status === "error" || llm.error
        ? "The LLM call itself failed, so there was no usable model output. The system fell back to the baseline."
      : !audit.llm_valid
        ? "The LLM responded, but the output did not pass the strict JSON parser, so the system fell back to the baseline."
        : audit.override_status === "blocked_high_confidence_baseline"
          ? "The LLM produced a valid changed label, but it was blocked because the baseline was high-confidence auto-accept."
          : audit.override_status === "kept_high_confidence_baseline"
            ? "The LLM produced a valid output, but the high-confidence baseline stayed locked as final."
            : audit.override_applied
              ? "The LLM produced a valid output and changed the baseline decision."
              : "The LLM produced a valid output but did not change the final decision.";
  byId("llmAuditMessage").textContent = llmMessage;
  const rawLlmOutput = llm?.output?.raw || llm?.error || "-";
  byId("llmRawOutput").textContent = rawLlmOutput || "-";
  byId("structuredEvidence").textContent = result?.evidence?.structured_evidence_summary || "No structured evidence.";
  byId("evidenceText").textContent = result?.evidence?.evidence_text || "No evidence text.";
  updateRagGroundingVisibility(result);
  const ragEnabled = Boolean(result?.rag?.llm_rag_context_enabled);
  byId("standardRagContext").textContent = ragEnabled
    ? (result?.rag?.standard_rag_context || "No RAG context.")
    : "RAG grounding was disabled for the LLM reviewer, so retrieved neighbors were not sent to the LLM prompt.";
  renderChips("gseChips", result?.evidence?.gse_ids || []);
  renderGseLinks(result?.evidence?.gse_urls || []);
  renderMainDecisionGse(result);
  renderChips("accessionChips", result?.evidence?.accession_list || [], true);
  renderNeighbors(ragEnabled ? (result?.rag?.neighbors || []) : []);
}


function renderPipelineCards(result) {
  const comparisonResults = result?.comparison_results || {};
  const requested = result?.selected_modes || Object.keys(comparisonResults);
  renderStrategyComparison({ requested_modes: requested, results: comparisonResults });
}

function renderStrategyComparison(payload) {
  const panel = byId("strategyComparePanel");
  panel.classList.remove("hidden");
  const results = payload?.results || {};
  const requestedModes = payload?.requested_modes || [];
  const grid = byId("strategyCompareGrid");
  grid.innerHTML = "";

  const modeLabels = {
    linear_only: "Linear only",
    hybrid_baseline: "Hybrid baseline",
    rag_vote_only: "RAG vote only",
    llm_sentence_judge_routed: "Sentence reviewer, routed",
    llm_sentence_judge_force: "Sentence reviewer, force",
    llm_final_classify_routed: "LLM classifier, routed",
    llm_final_classify_force: "LLM classifier, force",
  };

  const llmStatus = (result, requestedStrategy) => {
    const modeValue = String(requestedStrategy || "");
    const reviewerFromComposite = modeValue.includes("__") ? modeValue.split("__").pop() : modeValue;
    const llm = result?.llm || {};
    const audit = result?.decision_audit || {};
    const finalSource = result?.final?.source || "-";
    if (requestedStrategy === "pipeline_final") {
      const policy = result?.decision_audit?.pipeline_policy || result?.pipeline?.policy || "pipeline";
      return `Composed pipeline final. Policy: ${policy}.`;
    }
    if (["hybrid_baseline", "linear_only", "rag_vote_only"].includes(requestedStrategy)) {
      const baselineText = {
        hybrid_baseline: "No LLM requested. Uses linear_model_plus_rag as the final decision.",
        linear_only: "No LLM requested. Uses the plain linear model only.",
        rag_vote_only: "No LLM requested. Uses retrieval voting only.",
      };
      return baselineText[requestedStrategy];
    }
    const forceText = reviewerFromComposite.endsWith("_force") ? "forced on every row" : "routed only on review rows";
    if (audit.override_status === "blocked_high_confidence_baseline") {
      return `LLM ran (${forceText}) but its changed label was blocked because the baseline was high-confidence auto-accept.`;
    }
    if (audit.override_status === "kept_high_confidence_baseline") {
      return `LLM ran (${forceText}) as advisory output. The high-confidence baseline remained the final decision.`;
    }
    if (!audit.llm_requested) {
      return `LLM was not requested. Final source: ${finalSource}`;
    }
    if (!llm.called) {
      return `LLM was requested (${forceText}) but not called. Final source: ${finalSource}`;
    }
    if (audit.override_status === "error" || llm.error) {
      return `LLM call failed (${forceText}). Fell back to ${finalSource}.`;
    }
    if (!llm.valid) {
      return `LLM ran (${forceText}) but output was invalid. Fell back to ${finalSource}`;
    }
    if (audit.override_applied) {
      return `LLM ran (${forceText}) and overrode the baseline.`;
    }
    if (audit.llm_used_for_final) {
      return `LLM ran (${forceText}) and set the final source to ${finalSource}.`;
    }
    return `LLM ran (${forceText}) but final source stayed ${finalSource}.`;
  };

  const strategyType = (requestedStrategy) => {
    if (requestedStrategy === "pipeline_final") return "Composed final";
    if (["linear_only", "hybrid_baseline", "rag_vote_only"].includes(requestedStrategy)) return "Baseline";
    if (requestedStrategy.endsWith("_routed")) return "Routed LLM";
    if (requestedStrategy.endsWith("_force")) return "Forced LLM";
    return "Baseline";
  };

  for (const mode of requestedModes) {
    const result = results[mode] || {};
    const card = document.createElement("button");
    card.type = "button";
    card.className = "compare-card compare-card-clickable";
    card.dataset.mode = mode;
    const currentDisplayed = state.currentResult?.displayed_mode || state.currentResult?.selected_mode || "";
    if (currentDisplayed === mode) card.classList.add("active");
    card.innerHTML = `
      <span class="summary-label">${escapeHtml(modeLabel(mode))}</span>
      <span class="compare-meta-label">Mode type</span>
      <span class="compare-meta-value">${escapeHtml(strategyType(mode))}</span>
      <span class="compare-meta-label">RAG context for LLM</span>
      <span class="compare-meta-value">${result?.rag?.llm_rag_context_enabled ? `on (${escapeHtml(result?.rag?.llm_rag_top_k ?? "-")} neighbors)` : "off"}</span>
      <span class="compare-meta-label">Final source</span>
      <span class="compare-meta-value">${escapeHtml(result?.final?.source || "-")}</span>
      <strong>${escapeHtml(result?.final?.label || "-")}</strong>
      <span class="compare-meta-label">LLM proposed</span>
      <span class="compare-meta-value">${escapeHtml(result?.decision_audit?.llm_proposed_label || "-")} ${escapeHtml(formatConfidence(result?.decision_audit?.llm_confidence))}</span>
      <span class="compare-status">${escapeHtml(llmStatus(result, mode))}</span>
      <p class="compare-sentence">${escapeHtml(result?.evidence?.main_decision_sentence || "-")}</p>
      <span class="compare-inspect-hint">Click to inspect this mode</span>
    `;
    card.addEventListener("click", () => {
      const inspected = {
        ...result,
        displayed_mode: mode,
        selected_modes: requestedModes,
        comparison_results: results,
      };
      renderResult(inspected);
    });
    grid.appendChild(card);
  }
}


const MODE_LABELS_SHORT = {
  linear_only: "Linear only",
  hybrid_baseline: "Hybrid baseline",
  rag_vote_only: "RAG vote only",
  llm_sentence_judge_routed: "Sentence reviewer, routed",
  llm_sentence_judge_force: "Sentence reviewer, force",
  llm_final_classify_routed: "LLM classifier, routed",
  llm_final_classify_force: "LLM classifier, force",
  pipeline_final: "Pipeline final",
};

function selectedBaseModes() {
  const selected = Array.from(document.querySelectorAll("[data-base-mode].is-selected"))
    .map((node) => String(node.getAttribute("data-base-mode") || "").trim())
    .filter(Boolean);
  return selected.length ? selected : ["hybrid_baseline"];
}

function selectedBaseMode() {
  return selectedBaseModes()[0] || "hybrid_baseline";
}

function selectedReviewerMode() {
  const selected = document.querySelector("[data-reviewer-mode].is-selected");
  return String(selected?.getAttribute("data-reviewer-mode") || "none").trim();
}

function includeRagContextForLlm() {
  const checkbox = byId("includeRagContextForLlm");
  return checkbox ? Boolean(checkbox.checked) : true;
}

function llmRagTopK() {
  const raw = Number(byId("llmRagTopK")?.value ?? 3);
  if (Number.isNaN(raw)) return 3;
  return Math.max(0, Math.min(10, Math.floor(raw)));
}

function selectedModelModes() {
  const bases = selectedBaseModes();
  const reviewer = selectedReviewerMode();
  return reviewer && reviewer !== "none" ? bases.map((base) => `${base}__${reviewer}`) : bases;
}

function updateSelectedModeBar() {
  const bases = selectedBaseModes();
  const reviewer = selectedReviewerMode();
  const baseText = bases.map(modeLabel).join(" + ");
  const hasReviewer = reviewer && reviewer !== "none";
  const ragText = includeRagContextForLlm() ? `LLM grounding: on, top-k=${llmRagTopK()}` : "LLM grounding: off; neighbors hidden from LLM audit";
  const message = hasReviewer
    ? `Base models: ${baseText} | LLM reviewer: ${modeLabel(reviewer)} | ${ragText} | High-confidence locks still apply.`
    : `Base models: ${baseText} | Reviewer: No LLM reviewer | ${ragText} is ready if reviewer is enabled`;
  const bar = byId("selectedModeBar");
  if (bar) bar.textContent = message;
}

function syncProcessMapSelection() {
  document.querySelectorAll("[data-base-mode], [data-reviewer-mode]").forEach((node) => {
    const checked = node.classList.contains("is-selected");
    node.setAttribute("aria-pressed", checked ? "true" : "false");
  });
  updateSelectedModeBar();
}

function resetModeSelection() {
  document.querySelectorAll("[data-base-mode]").forEach((node) => {
    node.classList.toggle("is-selected", node.getAttribute("data-base-mode") === "hybrid_baseline");
  });
  document.querySelectorAll("[data-reviewer-mode]").forEach((node) => {
    node.classList.toggle("is-selected", node.getAttribute("data-reviewer-mode") === "none");
  });
  if (byId("includeRagContextForLlm")) byId("includeRagContextForLlm").checked = true;
  if (byId("llmRagTopK")) byId("llmRagTopK").value = "3";
  syncProcessMapSelection();
}

function selectBaseMode(mode) {
  const node = document.querySelector(`[data-base-mode="${mode}"]`);
  if (!node) return;
  node.classList.toggle("is-selected");
  if (!document.querySelectorAll("[data-base-mode].is-selected").length) {
    node.classList.add("is-selected");
    setStatus("error", "Keep at least one base decision model selected.");
  }
  syncProcessMapSelection();
}

function selectReviewerMode(mode) {
  document.querySelectorAll("[data-reviewer-mode]").forEach((node) => {
    node.classList.toggle("is-selected", node.getAttribute("data-reviewer-mode") === mode);
  });
  syncProcessMapSelection();
}

function toggleModeSelection(mode) {
  // Backward-compatible wrapper for older mode buttons, if any remain.
  if (["linear_only", "hybrid_baseline", "rag_vote_only"].includes(mode)) {
    selectBaseMode(mode);
  } else {
    selectReviewerMode(mode || "none");
  }
}

function summarizeBatch(results) {
  const counts = {};
  for (const row of results) {
    const label = row?.final?.label || "No result";
    counts[label] = (counts[label] || 0) + 1;
  }
  return Object.entries(counts)
    .map(([label, count]) => `${label}: ${count}`)
    .join(" | ");
}

function formatJobProgress(job) {
  const processed = Number(job?.processed ?? 0);
  const total = Number(job?.total ?? job?.count ?? job?.requested_identifiers ?? 0);
  const percent = Number(job?.progress_percent ?? (total ? (processed / total) * 100 : 0));
  if (!total) {
    return `${percent.toFixed(1)}%`;
  }
  return `${percent.toFixed(1)}% (${processed}/${total})`;
}

function formatJobDetail(job) {
  const currentRow = Number(job?.current_row ?? 0);
  const totalRows = Number(job?.total_rows ?? job?.count ?? 0);
  const currentMode = String(job?.current_mode || "").trim();
  const identifier = String(job?.current_identifier || "").trim();
  const parts = [];
  if (currentRow && totalRows) parts.push(`row ${currentRow}/${totalRows}`);
  if (currentMode) parts.push(`mode ${currentMode}`);
  if (identifier) parts.push(identifier);
  return parts.join(" | ");
}

const POLL_INTERVAL_MS = 1500;
const MAX_CONSECUTIVE_POLL_ERRORS = 5;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBatch(results) {
  state.batchResults = results || [];
  if (!results.length) {
    state.batchReviewStatus = {};
  }
  state.currentBatchSelectedIndex = results.length ? 0 : -1;
  const shell = byId("batchTableShell");
  const tbody = byId("batchTable").querySelector("tbody");
  const summary = byId("batchSummary");
  tbody.innerHTML = "";
  if (!results.length) {
    shell.classList.add("hidden");
    summary.textContent = "No rows returned.";
    byId("downloadBatchJson").disabled = true;
    byId("downloadBatchCsv").disabled = true;
    return;
  }
  summary.textContent = `Rows: ${results.length} | ${summarizeBatch(results)}`;
  shell.classList.remove("hidden");
  results.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.dataset.rowIndex = String(index);
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${escapeHtml(resultPaperLabel(row))}</td>
      <td>
        <div class="batch-label-cell">
          <span>${escapeHtml(row?.final?.label || row?.message || "-")}</span>
          <span class="review-badge hidden"></span>
        </div>
      </td>
      <td>${escapeHtml(formatConfidence(row?.predictions?.linear_model_plus_rag_conf))}</td>
      <td>${escapeHtml(row?.recommended_route || "-")}</td>
      <td>${escapeHtml((row?.evidence?.gse_ids || []).join(", ") || "-")}</td>
    `;
    tr.addEventListener("click", () => {
      state.currentBatchSelectedIndex = index;
      highlightSelectedBatchRow();
      renderResult(row);
    });
    tbody.appendChild(tr);
  });
  highlightSelectedBatchRow();
  renderResult(results[0]);
  byId("downloadBatchJson").disabled = false;
  byId("downloadBatchCsv").disabled = false;
}

function highlightSelectedBatchRow() {
  const rows = document.querySelectorAll("#batchTable tbody tr");
  rows.forEach((row, idx) => {
    const status = state.batchReviewStatus[idx] || "";
    row.classList.toggle("reviewed-confirmed", status === "confirmed");
    row.classList.toggle("reviewed-corrected", status === "corrected");
    row.classList.toggle("active", idx === state.currentBatchSelectedIndex);
    const badge = row.querySelector(".review-badge");
    if (badge) {
      if (status === "confirmed") {
        badge.textContent = "Confirmed";
        badge.classList.remove("hidden", "corrected");
      } else if (status === "corrected") {
        badge.textContent = "Corrected";
        badge.classList.remove("hidden");
        badge.classList.add("corrected");
      } else {
        badge.textContent = "";
        badge.classList.add("hidden");
        badge.classList.remove("corrected");
      }
    }
  });
}

function objectToCsv(results) {
  const headers = [
    "paper_id",
    "title",
    "final_label",
    "final_source",
    "recommended_route",
    "recommended_route_reason",
    "linear_model_plus_rag",
    "linear_model_plus_rag_conf",
    "rag_vote",
    "rag_vote_margin",
    "llm_proposed_label",
    "llm_confidence",
    "llm_used_for_final",
    "override_status",
    "gse_ids",
    "gse_urls",
    "main_decision_gse_ids",
    "main_decision_gse_urls",
    "accession_list",
    "main_decision_sentence",
    "main_decision_role",
  ];
  const lines = [headers.join(",")];
  for (const row of results) {
    const values = [
      row.paper_id || row.lookup?.paper_id || "",
      row.title || "",
      row.final?.label || "",
      row.final?.source || "",
      row.recommended_route || "",
      row.recommended_route_reason || "",
      row.predictions?.linear_model_plus_rag || "",
      row.predictions?.linear_model_plus_rag_conf || "",
      row.predictions?.rag_vote || "",
      row.predictions?.rag_vote_margin || "",
      row.decision_audit?.llm_proposed_label || "",
      row.decision_audit?.llm_confidence || "",
      row.decision_audit?.llm_used_for_final || "",
      row.decision_audit?.override_status || "",
      (row.evidence?.gse_ids || []).join(";"),
      (row.evidence?.gse_urls || []).join(";"),
      (row.evidence?.main_decision_gse_ids || []).join(";"),
      (row.evidence?.main_decision_gse_urls || []).join(";"),
      (row.evidence?.accession_list || []).join(";"),
      row.evidence?.main_decision_sentence || "",
      row.evidence?.main_decision_role || "",
    ].map(csvCell);
    lines.push(values.join(","));
  }
  return lines.join("\n");
}

function csvCell(value) {
  const text = String(value ?? "");
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll("\"", "\"\"")}"`;
  }
  return text;
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

async function parseJsonResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = body.detail || body.message || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body;
}

async function loadMetadata() {
  try {
    const [health, capabilities, mohammadInfo, articleSource, serverJobs] = await Promise.all([
      fetch("/health").then(parseJsonResponse),
      fetch("/capabilities").then(parseJsonResponse),
      fetch("/mohammad_subset_info").then(parseJsonResponse),
      fetch("/article_source").then(parseJsonResponse),
      fetch("/server_jobs").then(parseJsonResponse),
    ]);
    const meta = health.metadata || {};
    byId("metaTrainingRows").textContent = meta.training_rows ?? "-";
    byId("metaIndexedArticles").textContent = meta.indexed_articles ?? "-";
    byId("metaExtractionMode").textContent = meta.extraction_mode ?? "-";
    byId("metaOllamaModel").textContent = meta.ollama_model ?? "-";
    byId("metaLabeledBank").textContent = meta.labeled_csv_path || "-";
    if (byId("mohammadSummary")) {
      if (mohammadInfo.exists) {
        const activePart = articleSource.active_lookup_jsonl_path
          ? ` | Active lookup JSONL: ${articleSource.active_lookup_jsonl_path}`
          : "";
        const labeledPart = articleSource.active_labeled_bank_path
          ? ` | Active labeled bank: ${articleSource.active_labeled_bank_path}`
          : "";
        byId("mohammadSummary").textContent = `Default mapping: ${mohammadInfo.mapping_csv_path} | GEO/GSE PMCIDs: ${mohammadInfo.geo_pmcid_count}${activePart}${labeledPart}`;
      } else {
        byId("mohammadSummary").textContent = mohammadInfo.message || "Mohammad mapping CSV not found.";
      }
    }
    renderServerJobs(serverJobs);
    setStatus("ready", `Backend healthy. ${capabilities.input_modes?.length || 0} input modes available.`);
  } catch (error) {
    setStatus("error", `Backend metadata load failed: ${error.message}`);
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
}

function startMohammadFlow() {
  switchTab("mohammad");
  byId("mohammadSampleSize")?.focus();
}

async function submitPaste(event) {
  event.preventDefault();
  const bases = selectedBaseModes();
  const reviewer = selectedReviewerMode();
  const baseText = bases.map(modeLabel).join(" + ");
  setStatus("running", reviewer === "none" ? `Running selected baseline model(s): ${baseText}.` : `Running selected baseline model(s): ${baseText}, each followed by ${modeLabel(reviewer)}.`);
  try {
    const payload = {
      title: byId("pasteTitle").value.trim(),
      paper_id: byId("pastePaperId").value.trim() || null,
      text: byId("pasteText").value.trim(),
    };
    state.currentInputText = payload.text;
    state.currentInputTitle = payload.title;
    state.currentInputIdentifier = payload.paper_id || "";
    const result = await runReviewPipeline(payload, false);
    renderResult(result);
    if (result?.comparison_results) {
      renderPipelineCards(result);
    }
    setStatus("success", `Pipeline finished. Final label: ${result?.final?.label || "-"}.`);
  } catch (error) {
    setStatus("error", `Classification failed: ${error.message}`);
  }
}

async function submitIdentifier(event) {
  event.preventDefault();
  const bases = selectedBaseModes();
  const reviewer = selectedReviewerMode();
  const baseText = bases.map(modeLabel).join(" + ");
  setStatus("running", reviewer === "none" ? `Looking up identifier and running selected baseline model(s): ${baseText}.` : `Looking up identifier and running selected baseline model(s): ${baseText}, each followed by ${modeLabel(reviewer)}.`);
  try {
    state.currentInputText = "";
    state.currentInputTitle = "";
    state.currentInputIdentifier = byId("identifierValue").value.trim();
    const payload = { identifier: state.currentInputIdentifier };
    const result = await runReviewPipeline(payload, true);
    renderResult(result);
    if (result?.comparison_results) {
      renderPipelineCards(result);
    }
    if (result.found === false) {
      setStatus("error", result.message || "Identifier not found in local index.");
    } else {
      setStatus("success", `Pipeline finished. Final label: ${result?.final?.label || "-"}.`);
    }
  } catch (error) {
    setStatus("error", `Identifier classification failed: ${error.message}`);
  }
}

async function submitUpload(event) {
  event.preventDefault();
  const file = byId("uploadFile").files[0];
  if (!file) {
    setStatus("error", "Select a CSV, JSONL, or JSON file before running batch classification.");
    return;
  }
  const modes = selectedModelModes();
  if (!modes.length) {
    setStatus("error", "Select at least one model first.");
    return;
  }
  setStatus("running", `Uploading ${file.name} for batch classification.`);
  try {
    state.currentInputText = "";
    state.currentInputTitle = "";
    const formData = new FormData();
    formData.append("file", file);
    const batchMode = batchRequestMode(modes);
    formData.append("use_llm", String(batchMode.use_llm));
    formData.append("force_llm", String(batchMode.force_llm));
    formData.append("llm_strategy", batchMode.llm_strategy);
    formData.append("selected_models", JSON.stringify(modes));
    const job = await fetch("/classify_upload_async", {
      method: "POST",
      body: formData,
    }).then(parseJsonResponse);
    state.currentBatchJobId = job.job_id;
    markActiveJob("batch", job.job_id);
    byId("batchSummary").textContent = `Job ${job.job_id} queued. The server will keep running it even if the page refreshes.`;
    setStatus("running", `Batch job ${job.job_id} started. Waiting for completion.`);
    await pollBatchJob(job.job_id);
  } catch (error) {
    setStatus("error", `Batch classification failed: ${error.message}`);
  }
}

async function pollBatchJob(jobId) {
  let consecutiveErrors = 0;
  while (true) {
    try {
      const job = await fetch(`/jobs/${jobId}`).then(parseJsonResponse);
      consecutiveErrors = 0;
      const detail = formatJobDetail(job);
      byId("batchSummary").textContent = `Job ${job.job_id} | status=${job.status} | rows=${job.count}${detail ? ` | ${detail}` : ""} | Refresh is safe.`;
      setStatus("running", `Batch job ${job.job_id} is running.`);
      updateJobProgress(
        `Batch job ${job.job_id}`,
        job.progress_percent,
        `${job.processed || 0}/${job.total || job.count || 0}${detail ? ` | ${detail}` : ""}`
      );
      if (job.status === "completed") {
        const results = await fetch(`/jobs/${jobId}/download?format=json`).then(parseJsonResponse);
        renderBatch(results || []);
        clearActiveJob("batch");
        clearJobProgress();
        byId("batchSummary").textContent = `Job ${job.job_id} completed. Rows=${job.count}. Results are ready to download.`;
        await loadServerJobs();
        setStatus("success", `Batch job ${job.job_id} completed.`);
        return;
      }
      if (job.status === "cancelled") {
        clearActiveJob("batch");
        clearJobProgress();
        byId("batchSummary").textContent = `Job ${job.job_id} was cancelled.`;
        await loadServerJobs();
        setStatus("ready", `Batch job ${job.job_id} was cancelled.`);
        return;
      }
      if (job.status === "failed") {
        clearActiveJob("batch");
        clearJobProgress();
        await loadServerJobs();
        throw new Error(job.error || "Batch job failed.");
      }
    } catch (error) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        throw new Error(`Lost connection while polling the batch job. The job may still be running on the server. ${error.message}`);
      }
      setStatus("running", `Batch job ${jobId} still running. Retrying status check (${consecutiveErrors}/${MAX_CONSECUTIVE_POLL_ERRORS}).`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

async function beginMohammadImport(autoActivate) {
  state.autoActivateImport = autoActivate;
  setStatus("running", autoActivate ? "Starting Mohammad GEO/GSE import and activation." : "Starting Mohammad GEO/GSE subset import.");
  try {
    const payload = {
      sample_size: Number(byId("mohammadSampleSize").value || 0),
      batch_start: Number(byId("mohammadBatchStart").value || 0),
      skip_gold_standard: Boolean(byId("mohammadSkipGoldStandard").checked),
      output_jsonl: byId("mohammadOutputJsonl").value.trim() || "mohammad_geo_articles.jsonl",
    };
    const job = await fetch("/import_mohammad_subset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(parseJsonResponse);
    state.currentImportJobId = job.job_id;
    markActiveJob("mohammad", job.job_id);
    byId("mohammadSummary").textContent = `Import job ${job.job_id} queued. Polling for completion.`;
    setStatus("running", `Import job ${job.job_id} started. Waiting for completion.`);
    await pollMohammadImportJob(job.job_id);
  } catch (error) {
    setStatus("error", `Mohammad subset import failed: ${error.message}`);
  } finally {
    state.autoActivateImport = false;
  }
}

async function beginGenericFetch(autoActivate) {
  state.autoActivateGenericFetch = autoActivate;
  setStatus("running", autoActivate ? "Starting DOI / PMCID fetch and activation." : "Starting DOI / PMCID article fetch.");
  try {
    const formData = new FormData();
    formData.append("identifiers_text", byId("genericFetchText").value.trim());
    formData.append("output_jsonl", byId("genericFetchOutputJsonl").value.trim() || "fetched_articles.jsonl");
    const file = byId("genericFetchFile").files[0];
    if (file) {
      formData.append("file", file);
    }
    const job = await fetch("/fetch_articles_async", {
      method: "POST",
      body: formData,
    }).then(parseJsonResponse);
    state.currentGenericFetchJobId = job.job_id;
    markActiveJob("fetch", job.job_id);
    byId("genericFetchSummary").textContent = `Fetch job ${job.job_id} queued. Polling for completion.`;
    setStatus("running", `Fetch job ${job.job_id} started. Waiting for completion.`);
    await pollGenericFetchJob(job.job_id);
  } catch (error) {
    setStatus("error", `Article fetch failed: ${error.message}`);
  } finally {
    state.autoActivateGenericFetch = false;
  }
}

async function submitMohammadImport(event) {
  event.preventDefault();
  await beginMohammadImport(false);
}

async function importAndActivateMohammad() {
  await beginMohammadImport(true);
}

async function submitGenericFetch(event) {
  event.preventDefault();
  await beginGenericFetch(false);
}

async function fetchAndActivateGenericSource() {
  await beginGenericFetch(true);
}

async function pollMohammadImportJob(jobId) {
  let consecutiveErrors = 0;
  while (true) {
    try {
      const job = await fetch(`/import_jobs/${jobId}`).then(parseJsonResponse);
      consecutiveErrors = 0;
      const summary = job.summary || {};
      const pieces = [
        `Job ${job.job_id}`,
        `status=${job.status}`,
        `progress=${formatJobProgress(job)}`,
      ];
      if (summary.total_geo_pmcids !== undefined) pieces.push(`total=${summary.total_geo_pmcids}`);
      if (summary.skipped_gold_standard !== undefined) pieces.push(`skipped_gs=${summary.skipped_gold_standard}`);
      if (summary.remaining_after_exclusion !== undefined) pieces.push(`remaining=${summary.remaining_after_exclusion}`);
      if (summary.batch_start !== undefined) pieces.push(`start=${summary.batch_start}`);
      if (summary.batch_size !== undefined) pieces.push(`batch=${summary.batch_size || "all"}`);
      if (summary.requested_pmcids !== undefined) pieces.push(`requested=${summary.requested_pmcids}`);
      if (summary.written !== undefined) pieces.push(`written=${summary.written}`);
      if (summary.local_hits !== undefined) pieces.push(`local=${summary.local_hits}`);
      if (summary.remote_hits !== undefined) pieces.push(`remote=${summary.remote_hits}`);
      if (summary.failed !== undefined) pieces.push(`failed=${summary.failed}`);
      byId("mohammadSummary").textContent = pieces.join(" | ");
      setStatus("running", `Mohammad import ${job.job_id} is running.`);
      updateJobProgress(`Mohammad import ${job.job_id}`, job.progress_percent, `${job.processed || 0}/${job.total || 0}`);
      if (job.status === "completed") {
        byId("downloadMohammadJsonl").disabled = false;
        byId("activateMohammadJsonl").disabled = false;
        clearActiveJob("mohammad");
        if (state.autoActivateImport) {
          await activateCurrentImportSource(true);
          clearJobProgress();
          await loadServerJobs();
          setStatus("success", `Import job ${job.job_id} completed and activated as the current lookup source.`);
          return;
        }
        clearJobProgress();
        await loadServerJobs();
        setStatus("success", `Import job ${job.job_id} completed.`);
        return;
      }
      if (job.status === "cancelled") {
        clearActiveJob("mohammad");
        clearJobProgress();
        byId("mohammadSummary").textContent = `Job ${job.job_id} was cancelled.`;
        await loadServerJobs();
        setStatus("ready", `Mohammad import ${job.job_id} was cancelled.`);
        return;
      }
      if (job.status === "failed") {
        clearActiveJob("mohammad");
        clearJobProgress();
        await loadServerJobs();
        throw new Error(job.error || "Import job failed.");
      }
    } catch (error) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        throw new Error(`Lost connection while polling the Mohammad import job. The job may still be running on the server. ${error.message}`);
      }
      setStatus("running", `Mohammad import ${jobId} still running. Retrying status check (${consecutiveErrors}/${MAX_CONSECUTIVE_POLL_ERRORS}).`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

async function activateCurrentImportSource(silent = false) {
  if (!state.currentImportJobId) {
    setStatus("error", "No completed Mohammad import job is available to activate.");
    return;
  }
  if (!silent) {
    setStatus("running", "Activating imported JSONL as the current extra lookup source.");
  }
  try {
    const job = await fetch(`/import_jobs/${state.currentImportJobId}`).then(parseJsonResponse);
    const outputPath = job.output_jsonl;
    const resp = await fetch("/article_source/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonl_path: outputPath }),
    }).then(parseJsonResponse);
    await loadMetadata();
    if (!silent) {
      setStatus("success", `Activated lookup source: ${resp.active_lookup_jsonl_path}`);
    }
  } catch (error) {
    setStatus("error", `Lookup source activation failed: ${error.message}`);
  }
}

async function pollGenericFetchJob(jobId) {
  let consecutiveErrors = 0;
  while (true) {
    try {
      const job = await fetch(`/fetch_jobs/${jobId}`).then(parseJsonResponse);
      consecutiveErrors = 0;
      const summary = job.summary || {};
      const pieces = [
        `Job ${job.job_id}`,
        `status=${job.status}`,
        `progress=${formatJobProgress(job)}`,
      ];
      if (summary.requested_identifiers !== undefined) pieces.push(`requested=${summary.requested_identifiers}`);
      if (summary.written !== undefined) pieces.push(`written=${summary.written}`);
      if (summary.gse_hit_articles !== undefined) pieces.push(`gse_hits=${summary.gse_hit_articles}`);
      if (summary.failed !== undefined) pieces.push(`failed=${summary.failed}`);
      byId("genericFetchSummary").textContent = pieces.join(" | ");
      setStatus("running", `Article fetch ${job.job_id} is running.`);
      updateJobProgress(`Article fetch ${job.job_id}`, job.progress_percent, `${job.processed || 0}/${job.total || job.requested_identifiers || 0}`);
      if (job.status === "completed") {
        byId("downloadGenericFetchJsonl").disabled = false;
        byId("activateGenericFetchJsonl").disabled = false;
        clearActiveJob("fetch");
        if (state.autoActivateGenericFetch) {
          await activateCurrentGenericFetchSource(true);
          clearJobProgress();
          await loadServerJobs();
          setStatus("success", `Fetch job ${job.job_id} completed and activated as the current lookup source.`);
          return;
        }
        clearJobProgress();
        await loadServerJobs();
        setStatus("success", `Fetch job ${job.job_id} completed.`);
        return;
      }
      if (job.status === "cancelled") {
        clearActiveJob("fetch");
        clearJobProgress();
        byId("genericFetchSummary").textContent = `Job ${job.job_id} was cancelled.`;
        await loadServerJobs();
        setStatus("ready", `Article fetch ${job.job_id} was cancelled.`);
        return;
      }
      if (job.status === "failed") {
        clearActiveJob("fetch");
        clearJobProgress();
        await loadServerJobs();
        throw new Error(job.error || "Fetch job failed.");
      }
    } catch (error) {
      consecutiveErrors += 1;
      if (consecutiveErrors >= MAX_CONSECUTIVE_POLL_ERRORS) {
        throw new Error(`Lost connection while polling the article fetch job. The job may still be running on the server. ${error.message}`);
      }
      setStatus("running", `Article fetch ${jobId} still running. Retrying status check (${consecutiveErrors}/${MAX_CONSECUTIVE_POLL_ERRORS}).`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

async function activateCurrentGenericFetchSource(silent = false) {
  if (!state.currentGenericFetchJobId) {
    setStatus("error", "No completed DOI / PMCID fetch job is available to activate.");
    return;
  }
  if (!silent) {
    setStatus("running", "Activating fetched JSONL as the current extra lookup source.");
  }
  try {
    const job = await fetch(`/fetch_jobs/${state.currentGenericFetchJobId}`).then(parseJsonResponse);
    const outputPath = job.output_jsonl;
    const resp = await fetch("/article_source/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jsonl_path: outputPath }),
    }).then(parseJsonResponse);
    await loadMetadata();
    if (!silent) {
      setStatus("success", `Activated lookup source: ${resp.active_lookup_jsonl_path}`);
    }
  } catch (error) {
    setStatus("error", `Lookup source activation failed: ${error.message}`);
  }
}

function compactResultForFeedback(result) {
  if (!result) return {};
  return {
    paper_id: result.paper_id || result.lookup?.paper_id || null,
    identifier: result.lookup?.identifier || null,
    title: result.title || result.input_record?.title || "",
    final: result.final || {},
    predictions: result.predictions || {},
    recommended_route: result.recommended_route || "",
    recommended_route_reason: result.recommended_route_reason || "",
    evidence: {
      gse_ids: result.evidence?.gse_ids || [],
      accession_list: result.evidence?.accession_list || [],
      main_decision_sentence: result.evidence?.main_decision_sentence || "",
      main_decision_role: result.evidence?.main_decision_role || "",
      main_decision_gse_ids: result.evidence?.main_decision_gse_ids || [],
    },
    decision_audit: result.decision_audit || {},
  };
}

async function submitFeedback(event) {
  event.preventDefault();
  await saveFeedback(false);
}

async function saveFeedback(markCorrect) {
  const result = state.currentResult;
  if (!result) {
    setStatus("error", "Run a classification first, then submit reviewer feedback.");
    return;
  }
  const predictedLabel = byId("feedbackPredictedLabel").value;
  const correctedLabel = markCorrect ? predictedLabel : byId("feedbackCorrectedLabel").value;
  setStatus("running", markCorrect ? "Saving reviewer confirmation that the prediction is correct." : "Saving reviewer correction to pending feedback storage.");
  try {
    const payload = {
      paper_id: result.paper_id || result.lookup?.paper_id || null,
      identifier: result.lookup?.identifier || null,
      title: result.title || result.input_record?.title || state.currentInputTitle || "",
      text:
        result.input_record?.text ||
        result.input_record?.full_text ||
        result.input_record?.article_text ||
        state.currentInputText ||
        "",
      predicted_label: predictedLabel,
      corrected_label: correctedLabel,
      reviewer: byId("feedbackReviewer").value.trim(),
      reviewer_email: byId("feedbackReviewerEmail")?.value.trim() || "",
      note: markCorrect ? (byId("feedbackNote").value.trim() || "Reviewer confirmed model prediction is correct.") : byId("feedbackNote").value.trim(),
      evidence_sentence: byId("feedbackEvidenceSentence")?.value.trim() || result.evidence?.main_decision_sentence || "",
      consent_to_store_input_text: Boolean(byId("feedbackConsentStoreText")?.checked),
      result_json: compactResultForFeedback(result),
    };
    const response = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(parseJsonResponse);
    if (state.currentBatchSelectedIndex >= 0) {
      state.batchReviewStatus[state.currentBatchSelectedIndex] = markCorrect ? "confirmed" : "corrected";
      highlightSelectedBatchRow();
    }
    const feedbackCard = document.querySelector(".feedback-card");
    feedbackCard.classList.remove("feedback-confirmed", "feedback-corrected");
    feedbackCard.classList.add(markCorrect ? "feedback-confirmed" : "feedback-corrected");
    const storageLabel = response.storage === "supabase_pending_feedback" ? "pending Supabase feedback" : (response.feedback_store_path || "local feedback storage");
    byId("feedbackResult").textContent = markCorrect
      ? `Confirmed and saved to ${storageLabel}. Review status: ${response.review_status || "pending"}.`
      : `Correction saved to ${storageLabel}. Review status: ${response.review_status || "pending"}.`;
    setStatus("success", markCorrect ? "Reviewer confirmation saved." : "Reviewer correction saved.");
  } catch (error) {
    setStatus("error", `Feedback save failed: ${error.message}`);
  }
}

async function refreshRagBank() {
  setStatus("running", "Refreshing the reviewed labeled/RAG bank from human feedback.");
  try {
    const response = await fetch("/refresh_rag_bank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        output_csv: "rag_bank_refreshed.csv",
        report_json: "rag_bank_refresh_report.json",
        activate: true,
      }),
    }).then(parseJsonResponse);
    byId("ragBankResult").textContent = `Refreshed ${response.output_csv} | updated=${response.updated_rows} | appended=${response.appended_rows} | activated=${response.activated}`;
    await loadMetadata();
    setStatus("success", "Reviewed labeled bank refreshed and activated.");
  } catch (error) {
    setStatus("error", `RAG bank refresh failed: ${error.message}`);
  }
}

function selectedModelModes() {
  const bases = selectedBaseModes();
  const reviewer = selectedReviewerMode();
  return reviewer && reviewer !== "none" ? bases.map((base) => `${base}__${reviewer}`) : bases;
}

function batchRequestMode(modes) {
  const normalized = (modes || []).map((mode) => String(mode || "").split("__").pop());
  if (normalized.includes("llm_sentence_judge_force")) {
    return { use_llm: true, force_llm: true, llm_strategy: "sentence_judge" };
  }
  if (normalized.includes("llm_final_classify_force")) {
    return { use_llm: true, force_llm: true, llm_strategy: "classify" };
  }
  if (normalized.includes("llm_sentence_judge_routed")) {
    return { use_llm: true, force_llm: false, llm_strategy: "sentence_judge" };
  }
  if (normalized.includes("llm_final_classify_routed")) {
    return { use_llm: true, force_llm: false, llm_strategy: "classify" };
  }
  return { use_llm: false, force_llm: false, llm_strategy: "classify" };
}

async function runReviewPipelineCompare(payload, identifierOnly) {
  const requestPayload = {
    ...payload,
    base_modes: selectedBaseModes(),
    reviewer_mode: selectedReviewerMode(),
    include_rag_context_for_llm: includeRagContextForLlm(),
    llm_rag_top_k: llmRagTopK(),
  };
  if (identifierOnly) {
    requestPayload.text = "";
  }
  const result = await fetch("/review_pipeline_compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  }).then(parseJsonResponse);
  result.selected_modes = selectedModelModes();
  return result;
}

async function runReviewPipeline(payload, identifierOnly) {
  // Backward-compatible wrapper: current UI uses multi-base compare.
  return runReviewPipelineCompare(payload, identifierOnly);
}

async function runSingleSelectedModel(payload, mode, identifierOnly) {
  const requestPayload = { ...payload };
  if (["linear_only", "hybrid_baseline", "rag_vote_only"].includes(mode)) {
    requestPayload.use_llm = false;
    requestPayload.force_llm = false;
    requestPayload.llm_strategy = "classify";
  } else if (mode === "llm_sentence_judge_force") {
    requestPayload.use_llm = true;
    requestPayload.force_llm = true;
    requestPayload.llm_strategy = "sentence_judge";
  } else if (mode === "llm_sentence_judge_routed") {
    requestPayload.use_llm = true;
    requestPayload.force_llm = false;
    requestPayload.llm_strategy = "sentence_judge";
  } else if (mode === "llm_final_classify_routed") {
    requestPayload.use_llm = true;
    requestPayload.force_llm = false;
    requestPayload.llm_strategy = "classify";
  } else {
    requestPayload.use_llm = true;
    requestPayload.force_llm = true;
    requestPayload.llm_strategy = "classify";
  }
  const url = identifierOnly ? "/classify_identifier" : "/classify";
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload),
  }).then(parseJsonResponse);
}

function preferredResultFromComparison(comparison) {
  const requested = comparison?.requested_modes || [];
  const results = comparison?.results || {};
  const reversed = [...requested].reverse();
  for (const mode of reversed) {
    if (results[mode]) {
      const preferred = { ...results[mode] };
      preferred.displayed_mode = mode;
      preferred.selected_modes = requested;
      preferred.comparison_results = results;
      return preferred;
    }
  }
  return null;
}

function initEvents() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  byId("heroStartMohammad").addEventListener("click", startMohammadFlow);
  byId("pasteForm").addEventListener("submit", submitPaste);
  byId("identifierForm").addEventListener("submit", submitIdentifier);
  byId("uploadForm").addEventListener("submit", submitUpload);
  byId("mohammadForm").addEventListener("submit", submitMohammadImport);
  byId("importActivateMohammad").addEventListener("click", importAndActivateMohammad);
  byId("genericFetchForm").addEventListener("submit", submitGenericFetch);
  byId("genericFetchAndActivate").addEventListener("click", fetchAndActivateGenericSource);
  byId("feedbackForm").addEventListener("submit", submitFeedback);
  byId("feedbackConfirmCorrect").addEventListener("click", () => saveFeedback(true));
  byId("refreshRagBank").addEventListener("click", refreshRagBank);
  byId("pasteSample").addEventListener("click", () => {
    byId("pasteTitle").value = "Example GEO reuse study";
    byId("pasteText").value = "We downloaded the GEO dataset GSE12345 from the Gene Expression Omnibus and reanalyzed the public microarray data to compare responder and non-responder cohorts.";
    switchTab("paste");
  });
  byId("downloadBatchJson").addEventListener("click", () => {
    if (state.currentBatchJobId) {
      window.open(`/jobs/${state.currentBatchJobId}/download?format=json`, "_blank");
      return;
    }
    downloadText("batch_results.json", JSON.stringify(state.batchResults, null, 2), "application/json");
  });
  byId("downloadBatchCsv").addEventListener("click", () => {
    if (state.currentBatchJobId) {
      window.open(`/jobs/${state.currentBatchJobId}/download?format=csv`, "_blank");
      return;
    }
    downloadText("batch_results.csv", objectToCsv(state.batchResults), "text/csv;charset=utf-8");
  });
  byId("downloadMohammadJsonl").addEventListener("click", () => {
    if (!state.currentImportJobId) return;
    window.open(`/import_jobs/${state.currentImportJobId}/download`, "_blank");
  });
  byId("activateMohammadJsonl").addEventListener("click", activateCurrentImportSource);
  byId("downloadGenericFetchJsonl").addEventListener("click", () => {
    if (!state.currentGenericFetchJobId) return;
    window.open(`/fetch_jobs/${state.currentGenericFetchJobId}/download`, "_blank");
  });
  byId("activateGenericFetchJsonl").addEventListener("click", activateCurrentGenericFetchSource);
  byId("refreshServerJobs").addEventListener("click", () => {
    loadServerJobs().catch((error) => {
      setStatus("error", `Server job refresh failed: ${error.message}`);
    });
  });
  document.querySelectorAll("[data-base-mode]").forEach((node) => {
    node.addEventListener("click", () => selectBaseMode(node.getAttribute("data-base-mode")));
  });
  document.querySelectorAll("[data-reviewer-mode]").forEach((node) => {
    node.addEventListener("click", () => selectReviewerMode(node.getAttribute("data-reviewer-mode")));
  });
  document.querySelectorAll("[data-mode-node]").forEach((node) => {
    node.addEventListener("click", () => toggleModeSelection(node.getAttribute("data-mode-node")));
  });
  const resetModeBtn = byId("resetModeSelection");
  if (resetModeBtn) resetModeBtn.addEventListener("click", resetModeSelection);
  byId("includeRagContextForLlm")?.addEventListener("change", updateSelectedModeBar);
  byId("llmRagTopK")?.addEventListener("input", updateSelectedModeBar);
  syncProcessMapSelection();
}

async function resumeActiveJobs() {
  const active = loadActiveJobsFromStorage();
  try {
    const serverJobs = await loadServerJobs();
    if (active.batch) {
      state.currentBatchJobId = active.batch;
      byId("batchSummary").textContent = `Resuming batch job ${active.batch} from the server. The file picker is expected to be empty after refresh.`;
      await pollBatchJob(active.batch);
      return;
    }
    if (active.mohammad) {
      state.currentImportJobId = active.mohammad;
      byId("mohammadSummary").textContent = `Resuming Mohammad import job ${active.mohammad} polling.`;
      await pollMohammadImportJob(active.mohammad);
      return;
    }
    if (active.fetch) {
      state.currentGenericFetchJobId = active.fetch;
      byId("genericFetchSummary").textContent = `Resuming article fetch job ${active.fetch} polling.`;
      await pollGenericFetchJob(active.fetch);
      return;
    }
    const firstActive = Array.isArray(serverJobs.active_jobs) ? serverJobs.active_jobs[0] : null;
    if (firstActive) {
      const kind = String(firstActive.job_kind || "");
      const jobId = String(firstActive.job_id || "");
      if (kind === "batch" && jobId) {
        state.currentBatchJobId = jobId;
        markActiveJob("batch", jobId);
        byId("batchSummary").textContent = `Reconnected to server batch job ${jobId}.`;
        await pollBatchJob(jobId);
        return;
      }
      if (kind === "mohammad" && jobId) {
        state.currentImportJobId = jobId;
        markActiveJob("mohammad", jobId);
        byId("mohammadSummary").textContent = `Reconnected to server Mohammad import job ${jobId}.`;
        await pollMohammadImportJob(jobId);
        return;
      }
      if (kind === "fetch" && jobId) {
        state.currentGenericFetchJobId = jobId;
        markActiveJob("fetch", jobId);
        byId("genericFetchSummary").textContent = `Reconnected to server article fetch job ${jobId}.`;
        await pollGenericFetchJob(jobId);
        return;
      }
    }
  } catch (error) {
    setStatus("error", `Failed to resume active job polling: ${error.message}`);
  }
  clearJobProgress();
}

loadMetadata();
initEvents();
resumeActiveJobs();
