const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const format = new Intl.NumberFormat("en-US");

function renderMath() {
  if (!window.katex) return;
  $$('[data-latex]').forEach((element) => {
    const displayMode = element.closest(".equation-block, .equation-stack") !== null;
    window.katex.render(element.dataset.latex, element, {
      displayMode,
      throwOnError: false,
      strict: "warn"
    });
    element.classList.add("katex-rendered");
  });
}

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
  if (links.length === 0) return;
  const byId = new Map(links.map((link) => [link.hash.slice(1), link]));
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach((link) => link.classList.toggle("active", link === byId.get(visible.target.id)));
  }, { rootMargin: "-18% 0px -70% 0px", threshold: [0, 0.1, 0.35] });
  byId.forEach((_link, id) => {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  });
}

function setupPaperReader() {
  let frame = $("#reportPdf");
  const buttons = $$(".paper-jump");
  if (!frame || buttons.length === 0) return;
  const report = new URL(frame.src, document.baseURI);
  const selectPage = (page) => {
    buttons.forEach((item) => {
      const active = Number(item.dataset.page) === page;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
  };
  selectPage(Number(new URLSearchParams(report.hash.slice(1)).get("page")));
  buttons.forEach((button) => button.addEventListener("click", () => {
    const page = Number(button.dataset.page);
    if (!Number.isInteger(page) || page < 1) return;
    report.hash = `page=${page}&view=FitH`;
    // Native PDF viewers can ignore fragment-only changes on an existing frame.
    const nextFrame = frame.cloneNode(false);
    nextFrame.src = report.href;
    frame.replaceWith(nextFrame);
    frame = nextFrame;
    selectPage(page);
  }));
}

function setupTranslationLens() {
  const root = $("#translationLens");
  if (!root) return;
  const buttons = $$('[data-lens]', root);
  const panels = $$('[data-lens-panel]', root);
  buttons.forEach((button) => button.addEventListener("click", () => {
    const selected = button.dataset.lens;
    buttons.forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    panels.forEach((panel) => {
      const active = panel.dataset.lensPanel === selected;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }));
  panels.forEach((panel) => { panel.hidden = !panel.classList.contains("active"); });
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
      readout.textContent = `layer ${layer}: ${kind.toUpperCase()} — ${reason}`;
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

function drawDecayCurve(selectedZ) {
  const canvas = $("#decayCanvas");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = { left: 45, right: 18, top: 18, bottom: 30 };
  const xToPixel = (x) => pad.left + ((x + 8) / 16) * (width - pad.left - pad.right);
  const yToPixel = (y) => pad.top + ((0 - y) / 5) * (height - pad.top - pad.bottom);
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#d8d1c5";
  context.lineWidth = 1;
  [-5, -4, -3, -2, -1, 0].forEach((y) => {
    context.beginPath(); context.moveTo(pad.left, yToPixel(y)); context.lineTo(width - pad.right, yToPixel(y)); context.stroke();
    context.fillStyle = "#798078"; context.font = "10px monospace"; context.fillText(String(y), 17, yToPixel(y) + 3);
  });
  context.strokeStyle = "#0c5842";
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
  context.fillStyle = "#9b5d12";
  context.beginPath(); context.arc(xToPixel(selectedZ), yToPixel(selectedY), 6, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#687169"; context.font = "10px monospace";
  context.fillText("logit z", width - 62, height - 8);
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
  const canvas = $("#situCanvas");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = { left: 44, right: 15, top: 15, bottom: 28 };
  const xPixel = (x) => pad.left + ((x + 10) / 110) * (width - pad.left - pad.right);
  const yPixel = (y) => height - pad.bottom - (clamp(y, 0, 140) / 140) * (height - pad.top - pad.bottom);
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#ddd7cc"; context.lineWidth = 1;
  [0, 50, 100].forEach((y) => {
    context.beginPath(); context.moveTo(pad.left, yPixel(y)); context.lineTo(width - pad.right, yPixel(y)); context.stroke();
    context.fillStyle = "#7b827c"; context.font = "10px monospace"; context.fillText(String(y), 10, yPixel(y) + 3);
  });
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
  plot(situValue, "#0c5842", 3);
  context.fillStyle = "#9b5d12";
  context.beginPath(); context.arc(xPixel(selectedX), yPixel(situValue(selectedX)), 6, 0, Math.PI * 2); context.fill();
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
    ["patches", "1 × H × W × d"], ["spatial attention", "within one frame"],
    ["temporal attention", "identity at F = 1"], ["pool frames", "H × W × d"],
    ["2 × 2 merge", "H/2 × W/2 × 4d"], ["project", "tokens × 7168"]
  ] : [
    ["video patches", "F × H × W × d"], ["spatial attention", "within each frame"],
    ["temporal attention", "across F at each patch"], ["pool frames", "H × W × d"],
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
    label.innerHTML = `draft q${index + 1}<input type="range" min="0.01" max="1" step="0.01" value="${value}"><output>${value.toFixed(2)}</output>`;
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

renderMath();
setupReadingProgress();
setupScrollSpy();
setupPaperReader();
setupTranslationLens();
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
