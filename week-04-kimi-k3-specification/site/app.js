const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const format = new Intl.NumberFormat("en-US");

// Page numbers refer to the pinned v2 report, including its front matter.
const chapterPages = new Map(Object.entries({
  "lean-two-lives": 7, "reading-lean": 3,
  "transformer-primer": 3, "scaling-pressure": 3, "why-k3": 3,
  "paper-reader": 3, architecture: 3, orientation: 3, "formal-method": 7,
  configuration: 11, "kda-recurrence": 4, "kda-layer": 4, mla: 5,
  attnres: 6, "latent-moe": 6, situ: 7, routing: 8, vision: 9,
  composition: 3, "graph-proof": 3, muon: 10, "training-story": 10,
  pretraining: 11, "posttraining-story": 12, mopd: 13,
  microscaling: 14, eagle: 14, "agent-environments": 14, infrastructure: 17,
  evaluations: 25, "case-studies": 33
}));

function setupReadingProgress() {
  const bar = $("#readingProgress");
  const update = () => {
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = distance <= 0 ? 0 : window.scrollY / distance;
    bar.style.width = `${clamp(ratio, 0, 1) * 100}%`;
  };
  window.addEventListener("scroll", update, { passive: true });
  update();
}

function setupScrollSpy() {
  const links = $$(".article-toc a");
  const sections = $$("main > section.chapter-anchor");
  let current = "";
  let scheduled = false;
  const update = () => {
    scheduled = false;
    const readingLine = $("#chapterBar").getBoundingClientRect().bottom + 80;
    const section = scrollY + innerHeight >= document.documentElement.scrollHeight - 2
      ? sections.at(-1)
      : sections.reduce((selected, candidate) =>
        candidate.getBoundingClientRect().top <= readingLine ? candidate : selected, sections[0]);
    if (!section || section.id === current) return;
    current = section.id;
    links.forEach((link) => {
      const active = link.hash === `#${current}`;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    document.dispatchEvent(new CustomEvent("chapterchange", { detail: { section } }));
  };
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(update);
  };
  window.addEventListener("scroll", schedule, { passive: true });
  window.addEventListener("resize", schedule);
  update();
}

function setupPaperReader() {
  let frame = $("#reportPdf");
  const pane = $("#pdfPane");
  if (!frame || !pane) return null;
  const report = new URL(frame.src, document.baseURI);
  let currentPage = Number(new URLSearchParams(report.hash.slice(1)).get("page")) || 3;
  let docked = false;
  let opener = null;
  const toggle = $("#paperToggle");
  const dockToggle = $("#paperDockToggle");
  const close = $("#paperClose");
  const pageInput = $("#paperPage");
  const previous = $("#paperPrevious");
  const next = $("#paperNext");
  const follow = $("#followChapters");

  const setDocked = (open, returnFocus = false) => {
    if (open === docked) return;
    // Preserve the visible paragraph when the article reflows into the narrower column.
    const readingLine = $("#chapterBar").getBoundingClientRect().bottom + 30;
    const anchor = $$("main > section.chapter-anchor p, main > section.chapter-anchor h2, main > section.chapter-anchor h3")
      .find((element) => !pane.contains(element) && element.getBoundingClientRect().bottom >= readingLine);
    const offset = anchor ? anchor.getBoundingClientRect().top : 0;
    if (open) opener = document.activeElement;
    docked = open;
    pane.classList.toggle("is-docked", open);
    document.body.classList.toggle("report-open", open);
    toggle.setAttribute("aria-pressed", String(open));
    toggle.textContent = open ? "Close PDF" : "Open PDF";
    dockToggle.setAttribute("aria-pressed", String(open));
    dockToggle.hidden = open;
    close.hidden = !open;
    if (anchor && window.matchMedia("(min-width: 1101px)").matches) {
      requestAnimationFrame(() => window.scrollBy({ top: anchor.getBoundingClientRect().top - offset, behavior: "instant" }));
    }
    if (open) { frame.loading = "eager"; close.focus({ preventScroll: true }); }
    else if (returnFocus && opener?.isConnected) opener.focus({ preventScroll: true });
  };

  const selectPage = () => {
    $$(".paper-jump").forEach((item) => {
      const active = Number(item.dataset.page) === currentPage;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    pageInput.value = String(currentPage);
    previous.disabled = currentPage === 1;
    next.disabled = currentPage === 47;
    $("#paperExternal").href = `${report.origin}${report.pathname}#page=${currentPage}`;
  };

  const showPage = (page, { dock = false } = {}) => {
    if (!Number.isInteger(page) || page < 1 || page > 47) return;
    if (dock) setDocked(true);
    if (currentPage !== page) {
      report.hash = `page=${page}&view=FitH`;
      // Native viewers can ignore a fragment change on an existing iframe.
      const nextFrame = frame.cloneNode(false);
      nextFrame.loading = "eager";
      nextFrame.src = report.href;
      frame.replaceWith(nextFrame);
      frame = nextFrame;
      currentPage = page;
    }
    selectPage();
  };

  document.addEventListener("click", (event) => {
    const button = event.target.closest(".paper-jump");
    if (button) showPage(Number(button.dataset.page), { dock: button.dataset.dock === "true" });
  });
  toggle.addEventListener("click", () => {
    if (docked) setDocked(false, true);
    else showPage(chapterPages.get($("#chapterSelect").value) || currentPage, { dock: true });
  });
  dockToggle.addEventListener("click", () => setDocked(true));
  close.addEventListener("click", () => setDocked(false, true));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && docked) setDocked(false, true);
  });
  $("#paperPageForm").addEventListener("submit", (event) => {
    event.preventDefault();
    if (pageInput.reportValidity()) showPage(Number(pageInput.value));
  });
  previous.addEventListener("click", () => showPage(currentPage - 1));
  next.addEventListener("click", () => showPage(currentPage + 1));
  document.addEventListener("chapterchange", (event) => {
    const page = chapterPages.get(event.detail.section.id);
    if (follow.checked && docked && page) showPage(page);
  });
  follow.addEventListener("change", () => {
    const page = chapterPages.get($("#chapterSelect").value);
    if (follow.checked && page) showPage(page);
  });
  [toggle, dockToggle, $("#paperPageForm")].forEach((element) => { element.hidden = false; });
  selectPage();
  return { showPage };
}

function setupBookNavigation(reader) {
  const bar = $("#chapterBar");
  const select = $("#chapterSelect");
  const sections = $$("main > section.chapter-anchor");
  if (!bar || !select || !reader) return;
  let currentIndex = 0;
  sections.forEach((section, index) => {
    const title = $("h2", section)?.textContent.trim() || section.id;
    select.add(new Option(`${index + 1}. ${title}`, section.id));
    const page = chapterPages.get(section.id);
    const heading = $(".spread-header, .longform-heading, .distill-heading, .essay-heading, .section-heading", section);
    if (!heading || !page) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "paper-jump chapter-paper-jump";
    button.dataset.page = String(page);
    button.dataset.dock = "true";
    button.textContent = `PDF · p. ${page}`;
    button.setAttribute("aria-label", `Read report page ${page} beside ${title}`);
    button.setAttribute("aria-controls", "pdfPane");
    heading.append(button);
  });
  const previous = $("#previousChapter");
  const next = $("#nextChapter");
  const paper = $("#chapterPaper");
  const navigate = (index) => {
    const section = sections[clamp(index, 0, sections.length - 1)];
    history.replaceState(null, "", `#${section.id}`);
    section.scrollIntoView({ behavior: "instant", block: "start" });
  };
  select.addEventListener("change", () => navigate(sections.findIndex((section) => section.id === select.value)));
  previous.addEventListener("click", () => navigate(currentIndex - 1));
  next.addEventListener("click", () => navigate(currentIndex + 1));
  paper.addEventListener("click", () => {
    const page = chapterPages.get(sections[currentIndex].id);
    if (page) reader.showPage(page, { dock: true });
  });
  document.addEventListener("chapterchange", (event) => {
    currentIndex = sections.indexOf(event.detail.section);
    select.value = event.detail.section.id;
    $("#chapterPosition").textContent = `${currentIndex + 1} / ${sections.length}`;
    previous.disabled = currentIndex === 0;
    next.disabled = currentIndex === sections.length - 1;
    const page = chapterPages.get(event.detail.section.id);
    paper.hidden = !page;
    if (page) paper.textContent = `p. ${page}`;
  });
  bar.hidden = false;
  const measure = () => {
    document.documentElement.style.setProperty("--topbar-height", `${$(".topbar").offsetHeight}px`);
    document.documentElement.style.setProperty("--reader-offset", `${$(".topbar").offsetHeight + bar.offsetHeight + 12}px`);
  };
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    observer.observe($(".topbar"));
  }
  window.addEventListener("resize", measure);
  measure();
}

function setupAttentionPrimer() {
  const tokenRoot = $("#attentionTokens");
  const barRoot = $("#attentionBars");
  const readout = $("#attentionReadout");
  if (!tokenRoot || !barRoot || !readout) return;

  const tokens = ["The", "robot", "picked", "up", "the", "box", "because", "it", "was", "heavy"];
  const semanticLinks = new Map([
    [2, new Map([[1, 1.2]])],
    [3, new Map([[2, 1.0]])],
    [5, new Map([[4, 0.8], [2, 0.35]])],
    [6, new Map([[2, 0.45], [5, 0.45]])],
    [7, new Map([[5, 2.1], [1, 0.65]])],
    [8, new Map([[7, 1.35], [5, 0.8]])],
    [9, new Map([[5, 1.8], [7, 1.1], [8, 0.55]])]
  ]);

  const render = (queryIndex) => {
    const boosts = semanticLinks.get(queryIndex) || new Map();
    const raw = tokens.slice(0, queryIndex + 1).map((_token, keyIndex) => {
      const recency = 0.16 * Math.exp(-0.34 * (queryIndex - keyIndex));
      const self = keyIndex === queryIndex ? 0.42 : 0;
      return 0.035 + recency + self + (boosts.get(keyIndex) || 0);
    });
    const total = raw.reduce((sum, value) => sum + value, 0);
    const weights = raw.map((value) => value / total);
    barRoot.replaceChildren(...weights.map((weight, keyIndex) => {
      const row = document.createElement("div");
      row.className = "attention-bar-row";
      row.innerHTML = `<span>${tokens[keyIndex]}</span><i><b style="width:${(100 * weight).toFixed(1)}%"></b></i><output>${weight.toFixed(2)}</output>`;
      return row;
    }));
    $$("button", tokenRoot).forEach((button, index) => {
      const active = index === queryIndex;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    const strongest = weights.reduce((best, value, index) =>
      value > best.value ? { value, index } : best, { value: -1, index: 0 });
    readout.textContent = `query “${tokens[queryIndex]}” gives its largest weight to “${tokens[strongest.index]}”; positions to the right are masked.`;
  };

  tokenRoot.replaceChildren(...tokens.map((token, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = token;
    button.setAttribute("aria-label", `Use ${token} as the attention query`);
    button.addEventListener("click", () => render(index));
    return button;
  }));
  render(7);
}

function setupContextLab() {
  const input = $("#contextPower");
  const lengthOutput = $("#contextLengthValue");
  const pairsOutput = $("#attentionPairsValue");
  const rowsOutput = $("#kvRowsValue");
  if (!input || !lengthOutput || !pairsOutput || !rowsOutput) return;
  const update = () => {
    const tokens = Math.max(1, Math.round(10 ** Number(input.value)));
    const tokenCount = BigInt(tokens);
    const causalPairs = tokenCount * (tokenCount + 1n) / 2n;
    lengthOutput.textContent = format.format(tokens);
    pairsOutput.textContent = format.format(causalPairs);
    rowsOutput.textContent = format.format(tokens);
  };
  input.addEventListener("input", update);
  update();
}

function setupLinkedSymbols() {
  const symbols = $$('[data-symbol]');
  symbols.forEach((symbol) => {
    const toggle = (active) => {
      symbols.filter((candidate) => candidate.dataset.symbol === symbol.dataset.symbol)
        .forEach((candidate) => candidate.classList.toggle("symbol-active", active));
    };
    symbol.addEventListener("mouseenter", () => toggle(true));
    symbol.addEventListener("mouseleave", () => toggle(false));
    symbol.addEventListener("focus", () => toggle(true));
    symbol.addEventListener("blur", () => toggle(false));
  });
}

function setupLayerSchedule() {
  const container = $("#layerSchedule");
  const readout = $("#scheduleReadout");
  for (let layer = 1; layer <= 93; layer += 1) {
    const kind = layer === 93 || layer % 4 === 0 ? "mla" : "kda";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `layer-cell ${kind}`;
    button.title = `Layer ${layer}: ${kind.toUpperCase()}`;
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", () => {
      $$(".layer-cell", container).forEach((cell) => cell.classList.remove("active"));
      button.classList.add("active");
      const reason = layer === 93 ? "final layer forced to global attention" :
        kind === "mla" ? "fourth layer in the 3:1 block" : "recurrent layer in the 3:1 block";
      readout.textContent = `layer ${layer}: ${kind.toUpperCase()}, ${reason}`;
    });
    container.append(button);
  }
}

function setupKDA() {
  const alphaInput = $("#kdaAlpha");
  const betaInput = $("#kdaBeta");
  const incoming = [[1, 0.2], [-0.3, 0.7]];
  const key = [0.6, -0.8];
  const value = [0.4, 1.1];
  const renderMatrix = (element, matrix) => {
    element.replaceChildren(...matrix.flat().map((entry) => {
      const cell = document.createElement("i");
      cell.textContent = entry.toFixed(2).replace("-0.00", "0.00");
      return cell;
    }));
  };
  const update = () => {
    const alpha = Number(alphaInput.value);
    const beta = Number(betaInput.value);
    $("#kdaAlphaValue").textContent = alpha.toFixed(2);
    $("#kdaBetaValue").textContent = beta.toFixed(2);
    const decayed = incoming.map((row) => row.map((entry) => alpha * entry));
    const correction = [0, 1].map((column) =>
      key.reduce((sum, keyEntry, row) => sum + keyEntry * decayed[row][column], 0));
    const updated = decayed.map((row, i) => row.map((entry, j) =>
      entry - beta * key[i] * correction[j] + beta * key[i] * value[j]));
    renderMatrix($("#kdaDecayed"), decayed);
    renderMatrix($("#kdaUpdated"), updated);
    $("#kdaExplanation").textContent =
      `key = [0.6, −0.8], value = [0.4, 1.1]. The old key-aligned value is ${correction.map((x) => x.toFixed(3)).join(", ")}; β controls how strongly it is replaced.`;
  };
  alphaInput.addEventListener("input", update);
  betaInput.addEventListener("input", update);
  update();
}

// Keep chart labels at their CSS size on phones and draw sharply on high-DPI screens.
function chartContext(canvas, height = 230) {
  const width = canvas.clientWidth || 720;
  const ratio = window.devicePixelRatio || 1;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.font = '13px "IBM Plex Sans", sans-serif';
  return { context, width, height };
}

function drawDecayCurve(selectedZ) {
  const { context, width, height } = chartContext($("#decayCanvas"));
  const pad = { left: 42, right: 20, top: 40, bottom: 36 };
  const xToPixel = (x) => pad.left + ((x + 8) / 16) * (width - pad.left - pad.right);
  const yToPixel = (y) => pad.top + ((0 - y) / 5) * (height - pad.top - pad.bottom);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  context.strokeStyle = "#e0e5e8";
  context.lineWidth = 1;
  [-5, -4, -3, -2, -1, 0].forEach((y) => {
    context.beginPath(); context.moveTo(pad.left, yToPixel(y)); context.lineTo(width - pad.right, yToPixel(y)); context.stroke();
    context.fillStyle = "#69737d";
    context.textAlign = "right"; context.fillText(String(y).replace("-", "−"), pad.left - 12, yToPixel(y) + 4);
  });
  context.textAlign = "left";
  context.fillText("Log-decay g", 0, 18);
  [-8, -4, 0, 4, 8].forEach((x) => {
    context.textAlign = "center"; context.fillText(String(x).replace("-", "−"), xToPixel(x), height - 14);
  });
  context.textAlign = "right"; context.fillText("Logit z", width - pad.right, 18);
  context.strokeStyle = "#286493";
  context.lineWidth = 3;
  context.beginPath();
  for (let i = 0; i <= 320; i += 1) {
    const x = -8 + (16 * i) / 320;
    const y = -5 * sigmoid(x);
    const px = xToPixel(x), py = yToPixel(y);
    if (i === 0) context.moveTo(px, py); else context.lineTo(px, py);
  }
  context.stroke();
  const selectedY = -5 * sigmoid(selectedZ);
  context.fillStyle = "#b16943";
  context.beginPath(); context.arc(xToPixel(selectedZ), yToPixel(selectedY), 6, 0, Math.PI * 2); context.fill();
}

function setupDecay() {
  const input = $("#decayLogit");
  const update = () => {
    const z = Number(input.value);
    const g = -5 / (1 + Math.exp(-z));
    const retention = Math.exp(g);
    $("#decayLogitValue").textContent = z.toFixed(2);
    $("#logDecayValue").textContent = g.toFixed(4).replace("-", "−");
    $("#retentionValue").textContent = retention.toFixed(4);
    drawDecayCurve(z);
  };
  input.addEventListener("input", update);
  new ResizeObserver(update).observe($("#decayCanvas"));
  document.fonts.ready.then(update);
  update();
}

function setupCache() {
  const input = $("#cacheTokens");
  const update = () => {
    const tokens = Number(input.value);
    const full = tokens * 30720;
    const latent = tokens * 576;
    $("#cacheTokensValue").textContent = format.format(tokens);
    $("#fullCacheValue").textContent = format.format(full);
    $("#latentCacheValue").textContent = format.format(latent);
    $("#savedCacheValue").textContent = format.format(full - latent);
  };
  input.addEventListener("input", update);
  update();
}

function setupDepthBlocks() {
  const input = $("#depthLayers");
  const container = $("#depthBlocks");
  const update = () => {
    const layers = Number(input.value);
    const complete = Math.floor(layers / 12);
    const partial = layers % 12;
    $("#depthLayersValue").textContent = String(layers);
    const blocks = [];
    const embedding = document.createElement("div");
    embedding.className = "depth-block embedding";
    embedding.style.height = "74px";
    embedding.innerHTML = "<span>embedding</span>";
    blocks.push(embedding);
    for (let index = 0; index < complete; index += 1) {
      const block = document.createElement("div");
      block.className = "depth-block";
      block.style.height = `${56 + index * 3}px`;
      block.innerHTML = `<span>b${index + 1}</span>`;
      blocks.push(block);
    }
    if (partial > 0) {
      const block = document.createElement("div");
      block.className = "depth-block partial";
      block.style.height = `${28 + (partial / 12) * 55}px`;
      block.innerHTML = `<span>${partial}/12</span>`;
      blocks.push(block);
    }
    container.replaceChildren(...blocks);
    const visible = 1 + complete + (partial > 0 ? 1 : 0);
    $("#depthExplanation").textContent = `${complete} completed block${complete === 1 ? "" : "s"}, partial size ${partial}, ${visible} visible depth source${visible === 1 ? "" : "s"}.`;
  };
  input.addEventListener("input", update);
  update();
}

function situValue(x) {
  const sigmoid = 1 / (1 + Math.exp(-x));
  return (4 * Math.tanh(x / 4) * sigmoid) * (25 * Math.tanh(x / 25));
}

function swigluValue(x) {
  return (x / (1 + Math.exp(-x))) * x;
}

function drawSitu(selectedX) {
  const { context, width, height } = chartContext($("#situCanvas"), 260);
  const pad = { left: 44, right: 20, top: 40, bottom: 36 };
  const xPixel = (x) => pad.left + ((x + 10) / 110) * (width - pad.left - pad.right);
  const yPixel = (y) => height - pad.bottom - (y / 140) * (height - pad.top - pad.bottom);
  context.strokeStyle = "#e0e5e8"; context.lineWidth = 1;
  [0, 50, 100, 140].forEach((y) => {
    context.beginPath(); context.moveTo(pad.left, yPixel(y)); context.lineTo(width - pad.right, yPixel(y)); context.stroke();
    context.fillStyle = "#69737d";
    context.textAlign = "right"; context.fillText(String(y), pad.left - 12, yPixel(y) + 4);
  });
  context.textAlign = "left"; context.fillText("Output (g = u = x)", 0, 18);
  [-10, 0, 25, 50, 75, 100].forEach((x) => {
    // The two leftmost labels would collide on a narrow phone screen.
    if (width < 430 && x === -10) return;
    context.textAlign = "center"; context.fillText(String(x), xPixel(x), height - 14);
  });
  context.textAlign = "right"; context.fillText("Input x", width - pad.right, 18);
  // Clip the drawing, not the values: SwiGLU must leave the panel rather than flatten.
  context.save();
  context.beginPath();
  context.rect(pad.left, pad.top, width - pad.left - pad.right, height - pad.top - pad.bottom);
  context.clip();
  const plot = (fn, color, widthValue) => {
    context.strokeStyle = color; context.lineWidth = widthValue; context.beginPath();
    for (let i = 0; i <= 440; i += 1) {
      const x = -10 + (110 * i) / 440;
      const px = xPixel(x), py = yPixel(fn(x));
      if (i === 0) context.moveTo(px, py); else context.lineTo(px, py);
    }
    context.stroke();
  };
  plot(swigluValue, "#683c5b", 2);
  plot(situValue, "#286493", 3);
  context.fillStyle = "#b16943";
  context.beginPath(); context.arc(xPixel(selectedX), yPixel(situValue(selectedX)), 6, 0, Math.PI * 2); context.fill();
  context.restore();
}

function setupSitu() {
  const input = $("#situInput");
  const update = () => {
    const x = Number(input.value);
    $("#situInputValue").textContent = x.toFixed(1);
    $("#situOutput").textContent = `SiTU(${x.toFixed(1)}) = ${situValue(x).toFixed(2)}`;
    drawSitu(x);
  };
  input.addEventListener("input", update);
  new ResizeObserver(update).observe($("#situCanvas"));
  document.fonts.ready.then(update);
  update();
}

function setupRouter() {
  const raw = [0.74, 0.59, 0.48, 0.35];
  const biases = [0, 0.05, 0.22, -0.04];
  const controls = $("#routerControls");
  const bars = $("#routerBars");
  const inputs = biases.map((bias, index) => {
    const label = document.createElement("label");
    label.innerHTML = `expert ${index + 1}<input type="range" min="-0.4" max="0.4" step="0.01" value="${bias}"><output>${bias.toFixed(2)}</output>`;
    controls.append(label);
    return $("input", label);
  });
  const update = () => {
    const currentBias = inputs.map((input, index) => {
      const value = Number(input.value);
      $("output", input.parentElement).textContent = value.toFixed(2);
      return value;
    });
    const adjusted = raw.map((score, index) => score + currentBias[index]);
    const selected = adjusted.map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score || a.index - b.index).slice(0, 2).map((item) => item.index);
    const weightTotal = selected.reduce((sum, index) => sum + raw[index], 0);
    const rows = raw.map((score, index) => {
      const row = document.createElement("div");
      row.className = `router-row ${selected.includes(index) ? "selected" : ""}`;
      const weight = selected.includes(index) ? score / weightTotal : 0;
      row.innerHTML = `<span>E${index + 1}</span><div class="router-track"><i class="router-raw" style="width:${score * 78}%"></i><i class="router-adjusted" style="width:${clamp(adjusted[index], 0, 1.2) * 78}%"></i></div><span>${adjusted[index].toFixed(2)}</span><span>p=${weight.toFixed(2)}</span>`;
      return row;
    });
    bars.replaceChildren(...rows);
    $("#routerExplanation").textContent = `Selected experts: ${selected.map((index) => `E${index + 1}`).join(" and ")}. Selection uses raw + bias; p uses the selected raw scores only.`;
  };
  inputs.forEach((input) => input.addEventListener("input", update));
  update();
}

function renderVision(mode) {
  const stages = mode === "image" ? [
    ["patches", "1 × H × W × d"], ["+ positions", "64 × 64 table resized"],
    ["27 blocks", "RoPE attention, H·W"], ["pool frames", "H × W × d"],
    ["2 × 2 merge", "H/2 × W/2 × 4d"], ["project", "tokens × 7168"]
  ] : [
    ["video chunk", "F × H × W × d, F ≤ 4"], ["+ positions", "table + frame rows"],
    ["27 blocks", "RoPE attention, F·H·W"], ["pool frames", "H × W × d"],
    ["2 × 2 merge", "H/2 × W/2 × 4d"], ["project", "tokens × 7168"]
  ];
  $("#visionFlow").replaceChildren(...stages.map(([name, shape]) => {
    const stage = document.createElement("div");
    stage.className = "vision-stage";
    stage.innerHTML = `<div><b>${name}</b><small>${shape}</small></div>`;
    return stage;
  }));
}

function setupVision() {
  $$(".vision-mode").forEach((button) => button.addEventListener("click", () => {
    $$(".vision-mode").forEach((item) => item.classList.toggle("active", item === button));
    renderVision(button.dataset.mode);
  }));
  renderVision("image");
}

function setupMuon() {
  const matrix = $("#muonMatrix");
  const colors = ["#a7cfbd", "#ceb0c5", "#aac8dc", "#e5c896"];
  const cells = [];
  for (let row = 0; row < 6; row += 1) {
    for (let column = 0; column < 12; column += 1) {
      const head = Math.floor(column / 3);
      const cell = document.createElement("span");
      cell.className = "muon-cell";
      cell.dataset.head = String(head);
      cell.style.background = colors[head];
      cell.title = `row ${row + 1}, head ${head + 1}, local column ${(column % 3) + 1}`;
      cell.addEventListener("mouseenter", () => {
        matrix.dataset.hover = String(head);
        cells.forEach((item) => item.classList.toggle("hovered", item.dataset.head === String(head)));
        $("#muonExplanation").textContent = `Head ${head + 1}: columns ${head * 3 + 1}–${head * 3 + 3} become one 6 × 3 matrix.`;
      });
      cell.addEventListener("mouseleave", () => {
        delete matrix.dataset.hover;
        cells.forEach((item) => item.classList.remove("hovered"));
        $("#muonExplanation").textContent = "Each colored group is orthogonalized independently.";
      });
      cells.push(cell);
    }
  }
  matrix.replaceChildren(...cells);
}

function setupReward() {
  const teacher = $("#teacherProb");
  const student = $("#studentProb");
  const update = () => {
    const p = Number(teacher.value), q = Number(student.value);
    const raw = Math.log(p / q);
    const clipped = clamp(raw, -5, 5);
    $("#teacherProbValue").textContent = p.toFixed(3);
    $("#studentProbValue").textContent = q.toFixed(3);
    $("#rawReward").textContent = raw.toFixed(3).replace("-", "−");
    $("#clippedReward").textContent = clipped.toFixed(3).replace("-", "−");
  };
  teacher.addEventListener("input", update); student.addEventListener("input", update); update();
}

function setupMX() {
  const pattern = [0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 3, -3, 4, -4, 6, -6, 0];
  $("#mxValues").replaceChildren(...Array.from({ length: 32 }, (_, index) => {
    const cell = document.createElement("span");
    cell.textContent = String(pattern[(index * 5) % pattern.length]);
    return cell;
  }));
  const levels = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];
  const input = $("#mxInput");
  const update = () => {
    const value = Number(input.value);
    const nearest = levels.reduce((best, level) => Math.abs(level - value) < Math.abs(best - value) ? level : best, levels[0]);
    $("#mxInputValue").textContent = value.toFixed(2);
    $("#mxQuantized").textContent = nearest.toFixed(2);
    $("#mxError").textContent = Math.abs(nearest - value).toFixed(2);
  };
  input.addEventListener("input", update); update();
}

function setupEagle() {
  $("#eagleSteps").replaceChildren(...Array.from({ length: 7 }, (_, index) => {
    const step = document.createElement("div");
    step.className = "eagle-step";
    step.innerHTML = `<span>${index + 1}</span>`;
    step.title = index === 0 ? "Consumes fused target features" : "Consumes previous draft hidden output";
    return step;
  }));
}

function setupDistributions() {
  const target = [0.55, 0.30, 0.15];
  const initialDraft = [0.25, 0.45, 0.30];
  const controls = $("#distributionControls");
  const inputs = initialDraft.map((value, index) => {
    const label = document.createElement("label");
    label.innerHTML = `<span>draft <span data-tex="q_{${index + 1}}">q${index + 1}</span></span><input type="range" min="0.01" max="1" step="0.01" value="${value}"><output>${value.toFixed(2)}</output>`;
    renderLeanMath(label);
    controls.append(label);
    return $("input", label);
  });
  const update = () => {
    const raw = inputs.map((input) => Number(input.value));
    const total = raw.reduce((sum, value) => sum + value, 0);
    const draft = raw.map((value) => value / total);
    inputs.forEach((input, index) => { $("output", input.parentElement).textContent = draft[index].toFixed(2); });
    const overlap = target.map((value, index) => Math.min(value, draft[index]));
    const acceptance = overlap.reduce((sum, value) => sum + value, 0);
    const tv = 0.5 * target.reduce((sum, value, index) => sum + Math.abs(value - draft[index]), 0);
    const bars = target.map((value, index) => {
      const token = document.createElement("div");
      token.className = "distribution-token";
      token.title = `token ${index + 1}: p=${value.toFixed(2)}, q=${draft[index].toFixed(2)}, overlap=${overlap[index].toFixed(2)}`;
      token.innerHTML = `<i class="target-bar" style="height:${value * 100}%"></i><i class="draft-bar" style="height:${draft[index] * 100}%"></i><i class="overlap-bar" style="height:${overlap[index] * 100}%"></i>`;
      return token;
    });
    $("#distributionBars").replaceChildren(...bars);
    $("#acceptanceValue").textContent = acceptance.toFixed(3);
    $("#tvValue").textContent = tv.toFixed(3);
    $("#lossValue").textContent = (-Math.log(acceptance)).toFixed(3);
  };
  inputs.forEach((input) => input.addEventListener("input", update)); update();
}

function setupAtlas() {
  const input = $("#atlasSearch");
  const cards = $$("#moduleAtlas article");
  input.addEventListener("input", () => {
    const query = input.value.trim().toLowerCase();
    cards.forEach((card) => {
      const haystack = `${card.dataset.search} ${card.textContent}`.toLowerCase();
      card.hidden = query !== "" && !haystack.includes(query);
    });
  });
}

setupReadingProgress();
const paperReader = setupPaperReader();
setupBookNavigation(paperReader);
setupScrollSpy();
setupAttentionPrimer();
setupContextLab();
setupLinkedSymbols();
setupLayerSchedule();
setupKDA();
setupDecay();
setupCache();
setupDepthBlocks();
setupSitu();
setupRouter();
setupVision();
setupMuon();
setupReward();
setupMX();
setupEagle();
setupDistributions();
setupAtlas();

if (window.location.hash) {
  const scrollToHash = () => document.querySelector(window.location.hash)?.scrollIntoView();
  requestAnimationFrame(() => requestAnimationFrame(scrollToHash));
  window.addEventListener("load", () => {
    document.fonts.ready.then(scrollToHash);
  }, { once: true });
  [100, 400, 1000].forEach((delay) => window.setTimeout(scrollToHash, delay));
}
