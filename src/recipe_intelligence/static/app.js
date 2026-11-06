(() => {
  "use strict";

  const STORAGE_KEY = "recipe-intelligence.sessions.v1";
  const REQUEST_TIMEOUT_MS = 45_000;
  const MAX_LOCAL_SESSIONS = 30;
  const MAX_LOCAL_MESSAGES = 50;

  const elements = {
    sessionList: document.getElementById("session-list"),
    sessionEmpty: document.getElementById("session-empty"),
    newSession: document.getElementById("new-session"),
    deleteSession: document.getElementById("delete-session"),
    conversationTitle: document.getElementById("conversation-title"),
    welcome: document.getElementById("welcome"),
    messageList: document.getElementById("message-list"),
    processing: document.getElementById("processing"),
    messageForm: document.getElementById("message-form"),
    messageInput: document.getElementById("message-input"),
    sendButton: document.getElementById("send-button"),
    characterCount: document.getElementById("character-count"),
    connectionLabel: document.getElementById("connection-label"),
    toastRegion: document.getElementById("toast-region"),
    sidebar: document.getElementById("sidebar"),
    mobileMenu: document.getElementById("mobile-menu"),
  };

  let sessions = loadSessions();
  let activeSessionId = null;
  let sending = false;
  let connectionAvailable = false;

  function loadSessions() {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (item) =>
          item &&
          typeof item.id === "string" &&
          Array.isArray(item.messages) &&
          typeof item.title === "string",
      );
    } catch {
      return [];
    }
  }

  function persistSessions() {
    sessions = sessions.slice(0, MAX_LOCAL_SESSIONS);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch {
      showToast("Browser storage is full. Older local conversation messages may not be retained.", true);
    }
  }

  function showToast(message, isError = false) {
    const toast = document.createElement("div");
    toast.className = `toast${isError ? " error" : ""}`;
    toast.textContent = message;
    elements.toastRegion.append(toast);
    window.setTimeout(() => toast.remove(), 5200);
  }

  async function apiRequest(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch(path, {
        ...options,
        signal: controller.signal,
        headers: {
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      if (response.status === 204) return null;
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = payload?.error?.message || "The server could not complete that request.";
        const error = new Error(detail);
        error.status = response.status;
        error.code = payload?.error?.code;
        throw error;
      }
      return payload;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The request took too long. Please try again.");
      }
      if (error instanceof TypeError) {
        throw new Error("Could not reach the recipe service. Check that the application is running.");
      }
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function setConnectionState(available, text) {
    connectionAvailable = available;
    elements.connectionLabel.textContent = text;
    const dot = document.querySelector(".sidebar-footer .status-dot");
    dot.classList.toggle("offline", !available);
  }

  function activeSession() {
    return sessions.find((session) => session.id === activeSessionId) || null;
  }

  function updateSession(session, persist = true) {
    const index = sessions.findIndex((item) => item.id === session.id);
    if (index >= 0) sessions[index] = session;
    else sessions.unshift(session);
    sessions.sort((left, right) => (right.updatedAt || "").localeCompare(left.updatedAt || ""));
    if (persist) persistSessions();
    renderSessionList();
  }

  function renderSessionList() {
    elements.sessionList.replaceChildren();
    elements.sessionEmpty.hidden = sessions.length > 0;
    for (const session of sessions) {
      const row = document.createElement("div");
      row.className = `session-item${session.id === activeSessionId ? " active" : ""}`;

      const select = document.createElement("button");
      select.type = "button";
      select.className = "session-select";
      select.setAttribute("aria-current", session.id === activeSessionId ? "page" : "false");
      const glyph = document.createElement("span");
      glyph.className = "session-glyph";
      glyph.textContent = "◷";
      const name = document.createElement("span");
      name.className = "session-name";
      name.textContent = session.title || "New conversation";
      select.append(glyph, name);
      select.addEventListener("click", () => switchSession(session.id));

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "session-delete";
      remove.setAttribute("aria-label", `Delete ${session.title || "conversation"}`);
      remove.title = "Delete conversation";
      remove.textContent = "×";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteSession(session.id);
      });

      row.append(select, remove);
      elements.sessionList.append(row);
    }
  }

  function formatTime(value) {
    return new Intl.DateTimeFormat(undefined, {
      hour: "numeric",
      minute: "2-digit",
    }).format(value ? new Date(value) : new Date());
  }

  function makeMessage(role, text, data = {}) {
    return {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      role,
      text: String(text || ""),
      at: new Date().toISOString(),
      ...data,
    };
  }

  function appendMessage(session, message) {
    session.messages.push(message);
    session.messages = session.messages.slice(-MAX_LOCAL_MESSAGES);
    session.updatedAt = new Date().toISOString();
    updateSession(session);
    renderMessages(session);
  }

  function renderMessages(session) {
    elements.messageList.replaceChildren();
    elements.welcome.hidden = session.messages.length > 0;
    elements.conversationTitle.textContent =
      session.title === "New conversation" ? "Your kitchen, researched" : session.title;
    elements.deleteSession.disabled = false;
    for (const message of session.messages) {
      elements.messageList.append(renderMessage(message, session.id));
    }
    scrollConversationToBottom();
  }

  function renderMessage(message, sessionId) {
    const wrapper = document.createElement("article");
    wrapper.className = `message ${message.role}${message.isError ? " error" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = message.role === "user" ? "Y" : message.role === "tool" ? "↗" : "R";

    const body = document.createElement("div");
    body.className = "message-body";
    const meta = document.createElement("p");
    meta.className = "message-meta";
    meta.textContent = message.role === "user"
      ? `You · ${formatTime(message.at)}`
      : message.role === "tool"
        ? `One-shot tool · ${formatTime(message.at)}`
        : `Recipe assistant · ${formatTime(message.at)}`;
    body.append(meta);

    if (message.role === "user") {
      const text = document.createElement("p");
      text.className = "message-text";
      text.textContent = message.text;
      body.append(text);
    } else {
      const text = document.createElement("p");
      text.className = "assistant-copy";
      text.textContent = message.text;
      body.append(text);
      if (message.result) {
        const card = renderResult(message.result, {
          sessionId,
          isSessionResult: message.role === "assistant" && message.isSessionResult,
          kind: message.kind,
          state: message.state || {},
        });
        if (card) body.append(card);
      }
    }

    wrapper.append(avatar, body);
    return wrapper;
  }

  function createResultCard(title, count) {
    const card = document.createElement("section");
    card.className = "result-card";
    card.setAttribute("aria-label", title);
    const header = document.createElement("div");
    header.className = "result-card-heading";
    const label = document.createElement("strong");
    label.textContent = title;
    header.append(label);
    if (count !== undefined) {
      const countLabel = document.createElement("span");
      countLabel.className = "result-count";
      countLabel.textContent = String(count);
      header.append(countLabel);
    }
    card.append(header);
    return card;
  }

  function appendDataRow(parent, label, value) {
    const row = document.createElement("div");
    row.className = "data-row";
    const key = document.createElement("span");
    key.className = "data-label";
    key.textContent = label;
    const data = document.createElement("span");
    data.className = "data-value";
    data.textContent = value === null || value === undefined || value === ""
      ? "Not available in the record"
      : String(value);
    row.append(key, data);
    parent.append(row);
  }

  function appendChipList(parent, values) {
    if (!Array.isArray(values) || values.length === 0) return;
    const chips = document.createElement("div");
    chips.className = "recipe-ingredients";
    for (const value of values) {
      const chip = document.createElement("span");
      chip.className = "ingredient-chip";
      chip.textContent = typeof value === "string" ? value : JSON.stringify(value);
      chips.append(chip);
    }
    parent.append(chips);
  }

  function appendFilters(parent, filters) {
    if (!filters || typeof filters !== "object") return;
    const active = Object.entries(filters).filter(([, value]) =>
      Array.isArray(value) ? value.length > 0 : value !== null && value !== "" && value !== false,
    );
    if (!active.length) return;
    const line = document.createElement("div");
    line.className = "filters-line";
    line.textContent = `Applied filters: ${active.map(([key, value]) => `${key.replaceAll("_", " ")}: ${Array.isArray(value) ? value.join(", ") : value}`).join(" · ")}`;
    parent.append(line);
  }

  function renderRecipes(result, options) {
    const recipes = result.recipes || result.results || [];
    const card = createResultCard("Verified recipe records", `${recipes.length} result${recipes.length === 1 ? "" : "s"}`);
    if (recipes.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-results";
      empty.textContent = "No recipes matched this request. Try a broader title or relax a filter.";
      card.append(empty);
    }
    recipes.forEach((recipe, index) => {
      const item = document.createElement("article");
      item.className = "recipe-item";
      const titleRow = document.createElement("div");
      titleRow.className = "recipe-title-row";
      const titleGroup = document.createElement("div");
      const title = document.createElement("h3");
      title.className = "recipe-title";
      title.textContent = recipe.title || "Untitled recipe";
      const ref = document.createElement("div");
      ref.className = "recipe-ref";
      ref.textContent = recipe.reference || `${recipe.source}:${recipe.recipe_id}`;
      titleGroup.append(title, ref);
      titleRow.append(titleGroup);
      if (recipe.rank !== undefined && recipe.rank !== null) {
        const rank = document.createElement("span");
        rank.className = "recipe-rank";
        rank.textContent = `Rank ${index + 1}`;
        titleRow.append(rank);
      }
      item.append(titleRow);

      const ingredients = recipe.ingredients ?? recipe.original_record?.ingredients;
      appendChipList(item, ingredients);
      const instructions = recipe.instructions ?? recipe.original_record?.instructions;
      if (typeof instructions === "string" && instructions) {
        const paragraph = document.createElement("p");
        paragraph.className = "recipe-instructions";
        paragraph.textContent = instructions;
        item.append(paragraph);
      }

      const actions = document.createElement("div");
      actions.className = "recipe-card-actions";
      const detailButton = document.createElement("button");
      detailButton.type = "button";
      detailButton.className = "small-action";
      detailButton.textContent = "Inspect full recipe";
      detailButton.addEventListener("click", () => inspectRecipe(recipe, options.sessionId));
      actions.append(detailButton);
      if (options.isSessionResult && index < 3) {
        const selectButton = document.createElement("button");
        selectButton.type = "button";
        selectButton.className = "small-action";
        selectButton.textContent = "Use this recipe";
        selectButton.addEventListener("click", () =>
          sendMessage(`What about the ${["first", "second", "third"][index]} result?`),
        );
        actions.append(selectButton);
      }
      item.append(actions);
      card.append(item);
    });
    appendFilters(card, result.filters_applied || result.filters);
    return card;
  }

  function renderMatching(result, state) {
    const report = result.report || result;
    const matches = report.results || [];
    const card = createResultCard("Ingredient matches", `${matches.length} eligible`);
    const summary = document.createElement("div");
    summary.className = "evidence-card";
    const available = result.available_ingredients || state.available_ingredients || [];
    appendDataRow(summary, "Available ingredients", available.join(", ") || "As entered");
    appendDataRow(summary, "Candidates analyzed", report.candidate_recipes_analyzed);
    appendDataRow(summary, "Ranking", report.ranking_method);
    appendDataRow(summary, "Quantity check", report.quantity_sufficiency_verified ? "Verified" : "Not verified");
    const exclusions = report.hard_constraints?.excluded_ingredients
      || report.excluded_ingredients
      || state.hard_exclusions
      || [];
    const maxSteps = report.hard_constraints?.maximum_instruction_steps
      ?? report.max_instruction_steps
      ?? state.maximum_instruction_steps
      ?? "None";
    appendDataRow(summary, "Hard exclusions", exclusions.join(", ") || "None");
    appendDataRow(summary, "Maximum steps", maxSteps);
    appendDataRow(summary, "Ranking preferences", (report.ranking_preferences || state.ranking_preferences || []).join(", ") || "None");
    appendDataRow(summary, "Eligibility", "Preferences only affect ranking; hard constraints remain enforced");
    card.append(summary);
    if (matches.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-results";
      empty.textContent = "No eligible recipes matched. Hard exclusions and step limits were not relaxed.";
      card.append(empty);
    }
    for (const [index, recipe] of matches.entries()) {
      const item = document.createElement("article");
      item.className = "recipe-item";
      const title = document.createElement("h3");
      title.className = "recipe-title";
      title.textContent = `${index + 1}. ${recipe.title || "Untitled recipe"}`;
      const ref = document.createElement("div");
      ref.className = "recipe-ref";
      ref.textContent = `${recipe.source}:${recipe.recipe_id}`;
      const score = document.createElement("span");
      score.className = "match-score";
      score.textContent = `Coverage score · ${Number(recipe.score).toFixed(3)}`;
      item.append(title, ref, score);
      appendDataRow(item, "Matched", recipe.matched_ingredients?.map((value) => value.required_ingredient).join(", "));
      appendDataRow(item, "Missing", recipe.missing_ingredients?.join(", ") || "None");
      appendDataRow(item, "Additional", recipe.additional_ingredients?.join(", ") || "None");
      appendDataRow(item, "Instruction steps", recipe.instruction_step_count === null ? "Unknown" : `${recipe.instruction_step_count} (${recipe.instruction_step_method})`);
      const inspect = document.createElement("button");
      inspect.type = "button";
      inspect.className = "small-action";
      inspect.textContent = "Inspect recipe";
      inspect.addEventListener("click", () =>
        inspectRecipe(recipe, activeSessionId),
      );
      item.append(inspect);
      card.append(item);
    }
    return card;
  }

  function evidencePill(strength, sufficientlySupported) {
    const pill = document.createElement("span");
    const normalized = String(strength || "no_evidence").toLowerCase();
    pill.className = `status-pill${normalized === "direct" && sufficientlySupported ? "" : normalized === "no_evidence" ? " muted" : " warning"}`;
    pill.textContent = normalized.replaceAll("_", " ");
    return pill;
  }

  function renderSubstitution(result) {
    const proposal = result.proposal || result;
    const options = proposal.options || [];
    const card = createResultCard(
      proposal.original_title || "Substitution evidence",
      proposal.status === "supported" ? "Local evidence found" : "No reliable adaptation",
    );
    const intro = document.createElement("div");
    intro.className = "evidence-card";
    appendDataRow(intro, "Original recipe", `${proposal.original_title || "Untitled recipe"} · ${proposal.original_recipe?.source}:${proposal.original_recipe?.recipe_id}`);
    appendDataRow(intro, "Ingredient replaced", proposal.ingredient_to_replace);
    appendDataRow(intro, "Source ingredients", proposal.original_ingredient_list?.join("; "));
    card.append(intro);
    if (options.length === 0) {
      const empty = document.createElement("p");
      empty.className = "empty-results";
      empty.textContent = "The service found no alternative with sufficient local evidence.";
      card.append(empty);
    }
    for (const option of options) {
      const section = document.createElement("section");
      section.className = "evidence-card";
      const heading = document.createElement("h3");
      heading.className = "evidence-title";
      heading.textContent = option.replacement_ingredient;
      heading.append(evidencePill(option.evidence_strength, option.sufficiently_supported));
      if (option.sufficiently_supported) {
        const supported = document.createElement("span");
        supported.className = "status-pill";
        supported.textContent = "Supported by local evidence";
        heading.append(supported);
      } else {
        const uncertainty = document.createElement("span");
        uncertainty.className = "status-pill warning";
        uncertainty.textContent = "Not established as reliable";
        heading.append(uncertainty);
      }
      section.append(heading);
      appendDataRow(section, "Reasoning", option.reasoning);
      if (option.rejection_reason) appendDataRow(section, "Decision", option.rejection_reason);
      const supportIds = (option.supporting_recipes || []).map((item) =>
        `${item.reference?.source}:${item.reference?.recipe_id}`,
      );
      if (supportIds.length) {
        appendDataRow(section, "Supporting recipe IDs", supportIds.join(", "));
      }
      if ((option.cooccurrence_recipe_ids || []).length) {
        appendDataRow(section, "Co-occurrence only (not proof)", option.cooccurrence_recipe_ids.map((item) => `${item.source}:${item.recipe_id}`).join(", "));
      }
      if (option.revised_ingredient_list) {
        appendDataRow(section, "Revised ingredients", option.revised_ingredient_list.join("; "));
      }
      if (option.instruction_adjustments?.length) {
        appendDataRow(section, "Instruction changes", option.instruction_adjustments.map((item) => `${item.original_instruction} — ${item.adjustment_needed}`).join(" | "));
      }
      if (option.expected_trade_offs?.length) appendDataRow(section, "Expected trade-offs", option.expected_trade_offs.join("; "));
      if (option.remaining_uncertainty?.length) appendDataRow(section, "Remaining uncertainty", option.remaining_uncertainty.join("; "));
      section.append(createCitationList(supportIds));
      card.append(section);
    }
    const limitation = document.createElement("div");
    limitation.className = "restriction-box";
    limitation.textContent = "Dataset evidence is not proof of interchangeability or medical/allergy safety. Unsupported options remain explicitly unverified.";
    card.append(limitation);
    return card;
  }

  function createCitationList(ids) {
    const list = document.createElement("div");
    list.className = "citation-list";
    for (const id of ids) {
      const citation = document.createElement("span");
      citation.className = "citation";
      citation.textContent = id;
      list.append(citation);
    }
    return list;
  }

  function renderAnalytics(result) {
    const statistic = result.result || result.statistics?.[0] || result;
    const card = createResultCard("Computed from the local dataset", statistic.statistic);
    const details = document.createElement("div");
    details.className = "evidence-card";
    appendDataRow(details, "Result", `${statistic.value ?? "Not available"} ${statistic.unit || ""}`);
    appendDataRow(details, "Recipes analyzed", statistic.analyzed_recipe_count);
    appendDataRow(details, "Numerator", statistic.numerator);
    appendDataRow(details, "Denominator", statistic.denominator);
    appendDataRow(details, "Excluded records", statistic.excluded_recipe_count);
    appendDataRow(details, "Filters", JSON.stringify(statistic.filters || {}));
    appendDataRow(details, "Calculation method", statistic.calculation_method);
    card.append(details);
    return card;
  }

  function renderGeneric(result) {
    const card = createResultCard("Verified service result");
    const body = document.createElement("div");
    body.className = "evidence-card";
    for (const [key, value] of Object.entries(result)) {
      appendDataRow(body, key.replaceAll("_", " "), Array.isArray(value) ? value.join(", ") : typeof value === "object" && value !== null ? JSON.stringify(value) : value);
    }
    card.append(body);
    return card;
  }

  function renderResult(result, options) {
    if (!result || typeof result !== "object") return null;
    if (result.report?.results || result.results && result.candidate_recipes_analyzed !== undefined) {
      return renderMatching(result, options.state || {});
    }
    if (Array.isArray(result.recipes) || Array.isArray(result.results)) {
      return renderRecipes(result, options);
    }
    if (Array.isArray(result.options) && result.original_recipe) {
      return renderSubstitution(result);
    }
    if (result.proposal?.options || result.proposal?.original_recipe) {
      return renderSubstitution(result);
    }
    if (result.result?.statistic || result.statistics?.length) {
      return renderAnalytics(result);
    }
    return renderGeneric(result);
  }

  async function inspectRecipe(recipe, sessionId) {
    const ref = recipe.reference || `${recipe.source}:${recipe.recipe_id}`;
    const source = recipe.source;
    const id = recipe.recipe_id;
    try {
      const details = await apiRequest(
        `/api/v1/recipes/${encodeURIComponent(id)}${source ? `?source=${encodeURIComponent(source)}` : ""}`,
      );
      const session = sessions.find((item) => item.id === sessionId) || activeSession();
      if (!session) return;
      const detail = details.recipe || {};
      appendMessage(
        session,
        makeMessage("tool", `Full original record · ${details.reference || ref}`, {
          result: {
            recipes: [{
              source: details.source,
              recipe_id: details.recipe_id,
              reference: details.reference || ref,
              title: detail.title,
              ingredients: detail.ingredients,
              instructions: detail.instructions,
            }],
          },
          kind: "recipe_details",
        }),
      );
    } catch (error) {
      showToast(error.message, true);
    }
  }

  function scrollConversationToBottom() {
    requestAnimationFrame(() => {
      const last = elements.messageList.lastElementChild;
      last?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }

  function addUserMessage(session, text) {
    appendMessage(session, makeMessage("user", text));
    if (session.title === "New conversation") {
      session.title = text.trim().slice(0, 42) || "New conversation";
      updateSession(session);
      elements.conversationTitle.textContent = session.title;
    }
  }

  function errorMessage(error) {
    if (error.status === 404) return "This conversation is no longer available. Start a new conversation to continue.";
    if (error.status === 422) return `I couldn’t use that request: ${error.message}`;
    if (error.status === 503) return `The service is temporarily unavailable. ${error.message}`;
    return error.message || "Something went wrong while contacting the recipe service.";
  }

  async function sendMessage(rawText) {
    const text = String(rawText || "").trim();
    if (!text || sending) return;
    let session = activeSession();
    if (!session) {
      try {
        session = await createSession(false);
      } catch (error) {
        showToast(errorMessage(error), true);
        return;
      }
    }
    sending = true;
    setBusy(true);
    addUserMessage(session, text);
    try {
      const response = await apiRequest(
        `/api/v1/sessions/${encodeURIComponent(session.id)}/messages`,
        { method: "POST", body: JSON.stringify({ message: text }) },
      );
      if (response.status === "unavailable") {
        setConnectionState(true, "Database connected · Gemini unavailable");
      } else {
        setConnectionState(true, "Connected to your recipe database");
      }
      const message = makeMessage("assistant", response.response || "The service returned no response text.", {
        result: response.result || null,
        isSessionResult: true,
        kind: response.status,
        state: response.state,
      });
      appendMessage(session, message);
      if (response.state) session.state = response.state;
      updateSession(session);
    } catch (error) {
      const explanation = errorMessage(error);
      const isGone = error.status === 404 || error.status === 400;
      appendMessage(
        session,
        makeMessage("assistant", explanation, { isError: true }),
      );
      setConnectionState(!isGone, isGone ? "Conversation unavailable" : "Recipe service unavailable");
      if (isGone) showToast("This session can’t be loaded. Start a new conversation.", true);
    } finally {
      sending = false;
      setBusy(false);
      elements.messageInput.focus();
    }
  }

  function setBusy(busy) {
    elements.processing.hidden = !busy;
    elements.sendButton.disabled = busy;
    elements.messageInput.disabled = busy;
    elements.newSession.disabled = busy;
    for (const form of document.querySelectorAll(".tool-form")) {
      const button = form.querySelector(".tool-submit");
      if (button) button.disabled = busy;
    }
    if (busy) scrollConversationToBottom();
  }

  async function createSession(makeActive = true) {
    const created = await apiRequest("/api/v1/sessions", { method: "POST" });
    const session = {
      id: created.session_id,
      title: "New conversation",
      updatedAt: created.created_at || new Date().toISOString(),
      messages: [],
      state: created.state || null,
    };
    updateSession(session);
    if (makeActive) {
      activeSessionId = session.id;
      renderSessionList();
      renderMessages(session);
      closeMobileSidebar();
    }
    return session;
  }

  async function switchSession(sessionId) {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    try {
      const state = await apiRequest(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
      session.state = state.state;
      session.updatedAt = state.updated_at || session.updatedAt;
      activeSessionId = sessionId;
      updateSession(session);
      renderMessages(session);
      closeMobileSidebar();
    } catch (error) {
      if (error.status === 404) {
        sessions = sessions.filter((item) => item.id !== sessionId);
        persistSessions();
        renderSessionList();
      }
      showToast(errorMessage(error), true);
    }
  }

  async function deleteSession(sessionId) {
    const session = sessions.find((item) => item.id === sessionId);
    if (!session) return;
    if (!window.confirm(`Delete “${session.title}” from this browser and the recipe service?`)) return;
    try {
      await apiRequest(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      sessions = sessions.filter((item) => item.id !== sessionId);
      persistSessions();
      if (activeSessionId === sessionId) {
        activeSessionId = null;
        if (sessions.length) await switchSession(sessions[0].id);
        else await createSession(true);
      } else {
        renderSessionList();
      }
    } catch (error) {
      showToast(errorMessage(error), true);
    }
  }

  function splitTerms(value) {
    return String(value || "")
      .split(/,|;|\band\b/gi)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function appendToolResult(label, response, category) {
    const session = activeSession();
    if (!session) return;
    appendMessage(
      session,
      makeMessage("tool", `${label} completed using the API. This one-shot result does not change session context.`, {
        result: response,
        kind: category,
      }),
    );
  }

  async function submitTool(form, submitter, callback) {
    if (sending) return;
    sending = true;
    if (submitter) submitter.disabled = true;
    try {
      await callback();
    } catch (error) {
      const session = activeSession();
      if (session) {
        appendMessage(
          session,
          makeMessage("assistant", errorMessage(error), { isError: true }),
        );
      } else {
        showToast(errorMessage(error), true);
      }
    } finally {
      sending = false;
      if (submitter) submitter.disabled = false;
    }
  }

  function activeFormButton(form) {
    return form.querySelector(".tool-submit");
  }

  elements.messageForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = elements.messageInput.value;
    elements.messageInput.value = "";
    updateCharacterCount();
    resizeInput();
    void sendMessage(text);
  });

  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      elements.messageForm.requestSubmit();
    }
  });
  elements.messageInput.addEventListener("input", () => {
    updateCharacterCount();
    resizeInput();
  });

  function updateCharacterCount() {
    elements.characterCount.textContent = `${elements.messageInput.value.length} / 2000`;
  }

  function resizeInput() {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 140)}px`;
  }

  elements.newSession.addEventListener("click", async () => {
    if (sending) return;
    try {
      setConnectionState(true, "Connected to your recipe database");
      await createSession(true);
    } catch (error) {
      setConnectionState(false, "Recipe service unavailable");
      showToast(errorMessage(error), true);
    }
  });
  elements.deleteSession.addEventListener("click", () => {
    if (activeSessionId) void deleteSession(activeSessionId);
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      const prompt = button.dataset.prompt || "";
      if (prompt.startsWith("Select a recipe")) {
        elements.messageInput.value = "Find a recipe first, then select one of its results to explore a substitution.";
      } else {
        elements.messageInput.value = prompt;
      }
      updateCharacterCount();
      resizeInput();
      elements.messageInput.focus();
    });
  });

  document.getElementById("search-tool").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    void submitTool(form, activeFormButton(form), async () => {
      const filters = { exclude_ingredients: splitTerms(data.get("exclude")) };
      const result = await apiRequest("/api/v1/recipes/search", {
        method: "POST",
        body: JSON.stringify({
          query: String(data.get("query")).trim(),
          field: data.get("field"),
          limit: Number(data.get("limit")),
          include_details: true,
          filters,
        }),
      });
      appendToolResult("Recipe search", result, "search");
    });
  });

  document.getElementById("matching-tool").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const preferences = [...form.querySelectorAll('input[name="preference"]:checked')].map((input) => input.value);
    const steps = String(data.get("steps") || "").trim();
    void submitTool(form, activeFormButton(form), async () => {
      const available = splitTerms(data.get("available"));
      const result = await apiRequest("/api/v1/matching", {
        method: "POST",
        body: JSON.stringify({
          available_ingredients: available,
          excluded_ingredients: splitTerms(data.get("exclude")),
          max_instruction_steps: steps ? Number(steps) : null,
          ranking_preferences: preferences,
        }),
      });
      appendToolResult(
        "Ingredient matching",
        { ...result, available_ingredients: available },
        "matching",
      );
    });
  });

  document.getElementById("substitution-tool").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const source = String(data.get("source") || "");
    void submitTool(form, activeFormButton(form), async () => {
      const result = await apiRequest("/api/v1/substitutions", {
        method: "POST",
        body: JSON.stringify({
          recipe_id: String(data.get("recipe_id")).trim(),
          source: source || null,
          ingredient_to_replace: String(data.get("ingredient")).trim(),
          replacement_ingredients: splitTerms(data.get("replacement")),
          dietary_restrictions: splitTerms(data.get("restrictions")),
        }),
      });
      appendToolResult("Substitution evidence", result, "substitution");
    });
  });

  document.getElementById("analytics-tool").addEventListener("submit", (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const ingredient = String(data.get("ingredient") || "").trim();
    const filters = { exclude_ingredients: splitTerms(data.get("exclude")) };
    void submitTool(form, activeFormButton(form), async () => {
      const statistic = String(data.get("statistic"));
      if (statistic === "ingredient_prevalence" && !ingredient) {
        throw new Error("Enter an ingredient to calculate prevalence.");
      }
      const result = await apiRequest("/api/v1/analytics", {
        method: "POST",
        body: JSON.stringify({
          statistic,
          ingredient: ingredient || null,
          filters,
        }),
      });
      appendToolResult("Dataset analytics", result, "analytics");
    });
  });

  elements.mobileMenu.addEventListener("click", () => {
    const open = elements.sidebar.classList.toggle("open");
    elements.mobileMenu.setAttribute("aria-expanded", String(open));
  });

  function closeMobileSidebar() {
    elements.sidebar.classList.remove("open");
    elements.mobileMenu.setAttribute("aria-expanded", "false");
  }

  document.addEventListener("click", (event) => {
    if (window.innerWidth <= 680 && elements.sidebar.classList.contains("open")) {
      if (!elements.sidebar.contains(event.target) && !elements.mobileMenu.contains(event.target)) {
        closeMobileSidebar();
      }
    }
  });

  async function initialize() {
    renderSessionList();
    setConnectionState(false, "Connecting to your recipe database…");
    try {
      const health = await apiRequest("/health");
      setConnectionState(true, health.gemini === "disabled"
        ? "Database connected · Gemini disabled"
        : "Connected to your recipe database");
    } catch (error) {
      setConnectionState(false, "Recipe service unavailable");
      showToast(errorMessage(error), true);
    }

    const priorIds = sessions.map((session) => session.id);
    for (const id of priorIds) {
      try {
        const loaded = await apiRequest(`/api/v1/sessions/${encodeURIComponent(id)}`);
        const session = sessions.find((item) => item.id === id);
        if (session) {
          session.state = loaded.state;
          session.updatedAt = loaded.updated_at || session.updatedAt;
        }
      } catch (error) {
        if (error.status === 404) {
          sessions = sessions.filter((item) => item.id !== id);
        } else {
          setConnectionState(false, "Recipe service unavailable");
          break;
        }
      }
    }
    persistSessions();
    renderSessionList();
    if (sessions.length) {
      activeSessionId = sessions[0].id;
      renderSessionList();
      renderMessages(sessions[0]);
    } else if (connectionAvailable) {
      try {
        await createSession(true);
      } catch (error) {
        showToast(errorMessage(error), true);
      }
    }
  }

  void initialize();
})();
