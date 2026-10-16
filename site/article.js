/* Interactive illustrations and compiler-recorded information for the article examples. */
function setupLeanInfo() {
  if (!document.querySelector('[data-lean-info]')) return;
  const popup = document.createElement('pre');
  popup.id = 'lean-info';
  popup.className = 'lean-hover';
  popup.setAttribute('role', 'tooltip');
  popup.hidden = true;
  document.body.append(popup);
  let active = null;
  let focused = false;
  const close = () => {
    active?.removeAttribute('aria-describedby');
    active = null;
    popup.hidden = true;
  };
  const show = token => {
    if (!token) return;
    active?.removeAttribute('aria-describedby');
    active = token;
    token.setAttribute('aria-describedby', popup.id);
    const proof = token.closest('[data-lean-goal]');
    popup.textContent = token.dataset.leanInfo + (proof && proof !== token
      ? `\n\n${proof.dataset.leanGoal}` : '');
    popup.hidden = false;
    const rect = token.getBoundingClientRect();
    const size = popup.getBoundingClientRect();
    popup.style.left = `${Math.max(12, Math.min(rect.left, innerWidth - size.width - 12))}px`;
    const top = rect.bottom + size.height + 12 <= innerHeight
      ? rect.bottom + 8 : rect.top - size.height - 8;
    popup.style.top = `${Math.max(12, Math.min(top, innerHeight - size.height - 12))}px`;
  };
  document.addEventListener('pointerover', event => {
    if (focused) return;
    const token = event.target.closest('[data-lean-info]');
    if (token) show(token);
  });
  document.addEventListener('pointermove', event => {
    if (!event.movementX && !event.movementY) return;
    focused = false;
    const token = event.target.closest('[data-lean-info]');
    if (token && token !== active) show(token);
  });
  document.addEventListener('pointerout', event => {
    if (!active || !event.target.closest('[data-lean-info]')) return;
    if (event.relatedTarget && (popup.contains(event.relatedTarget)
      || active.contains(event.relatedTarget))) return;
    if (document.activeElement !== active) close();
  });
  document.addEventListener('focusin', event => {
    const token = event.target.closest('[data-lean-info]');
    if (!token) return;
    focused = true;
    token.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
    show(token);
  });
  document.addEventListener('focusout', event => {
    focused = false;
    if (event.target === active && !popup.contains(event.relatedTarget)) close();
  });
  document.addEventListener('click', event => {
    focused = false;
    const token = event.target.closest('[data-lean-info]');
    if (token) show(token);
    else if (!popup.contains(event.target)) close();
  });
  popup.addEventListener('pointerleave', () => {
    if (document.activeElement !== active) close();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !popup.hidden) {
      close();
      event.stopImmediatePropagation();
    }
  });
  document.addEventListener('scroll', event => {
    if (popup.contains(event.target) || !active) return;
    if (focused && document.activeElement === active) {
      requestAnimationFrame(() => { if (active) show(active); });
      return;
    }
    const rect = active.getBoundingClientRect();
    if (rect.bottom <= 0 || rect.top >= innerHeight || rect.right <= 0 || rect.left >= innerWidth) close();
    else show(active);
  }, true);
  window.addEventListener('resize', () => { if (active) show(active); });
}

setupLeanInfo();

// Add color inside the existing tokens, preserving their types, focus targets, and source text.
function colorLean(root, ranges, start = 0) {
  if (!ranges?.length) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  let offset = start;
  let first = 0;
  for (const node of nodes) {
    const points = [...node.textContent];
    const end = offset + points.length;
    while (first < ranges.length && ranges[first][1] <= offset) first++;
    const fragment = document.createDocumentFragment();
    let cursor = offset;
    for (let i = first; i < ranges.length && ranges[i][0] < end; i++) {
      const [left, right, kind] = ranges[i];
      const from = Math.max(left, offset);
      const to = Math.min(right, end);
      fragment.append(document.createTextNode(points.slice(cursor - offset, from - offset).join('')));
      const span = document.createElement('span');
      span.className = `lean-syntax ${kind}`;
      span.textContent = points.slice(from - offset, to - offset).join('');
      fragment.append(span);
      cursor = to;
    }
    fragment.append(document.createTextNode(points.slice(cursor - offset).join('')));
    node.replaceWith(fragment);
    offset = end;
  }
}
document.querySelectorAll('[data-lean-syntax]').forEach(code => {
  colorLean(code, JSON.parse(code.dataset.leanSyntax));
  code.removeAttribute('data-lean-syntax');
});

// Both the article excerpts and the existing source readers use the same compiler recording.
const leanScript = [...document.scripts].find(script => /\/assets\/article\.js(?:\?|$)/.test(script.src));
const leanBase = leanScript ? new URL('./lean/', leanScript.src) : null;
let leanIndex = null;
const leanModules = new Map();
async function leanSource(path) {
  if (!leanBase) throw new Error('Lean assets are unavailable');
  leanIndex ??= fetch(new URL('index.json', leanBase), { cache: 'no-store' }).then(response => {
    if (!response.ok) throw new Error('Lean source index is unavailable');
    return response.json();
  });
  const index = await leanIndex;
  if (!index[path]) throw new Error(`No checked Lean source for ${path}`);
  if (!leanModules.has(path)) leanModules.set(path, fetch(new URL(index[path], leanBase), {
    cache: 'no-store',
  }).then(response => {
    if (!response.ok) throw new Error(`Checked Lean source is unavailable: ${path}`);
    return response.json();
  }));
  return leanModules.get(path);
}
function leanFragment(model, start = 0, end = [...model.code].length) {
  const fragment = document.createDocumentFragment();
  const points = [...model.code];
  let lineNumber = points.slice(0, start).filter(char => char === '\n').length + 1;
  const lineStarts = [];
  let offset = start;
  for (const line of points.slice(start, end).join('').split('\n')) {
    lineStarts.push(offset);
    offset += [...line].length + 1;
  }
  // Offsets in the recording count Unicode code points; DOM strings count UTF-16 units.
  for (const lineStart of lineStarts) {
    if (lineStart >= end) break;
    let lineEnd = lineStart;
    while (lineEnd < end && points[lineEnd] !== '\n') lineEnd++;
    const goal = model.goals.find(goal => goal.start === lineStart && goal.end === lineEnd);
    const line = document.createElement('span');
    line.dataset.leanLine = String(lineNumber++);
    if (goal) {
      line.className = 'lean-step';
      line.tabIndex = 0;
      line.dataset.leanGoal = model.info[goal.info];
      line.dataset.leanInfo = model.info[goal.info];
    }
    let cursor = lineStart;
    for (const token of model.spans) {
      if (token.start < lineStart || token.end > lineEnd) continue;
      line.append(document.createTextNode(points.slice(cursor, token.start).join('')));
      const span = document.createElement('span');
      span.className = 'lean-token';
      span.tabIndex = 0;
      span.dataset.leanInfo = model.info[token.info];
      span.textContent = points.slice(token.start, token.end).join('');
      line.append(span);
      cursor = token.end;
    }
    line.append(document.createTextNode(points.slice(cursor, lineEnd).join('')));
    fragment.append(line);
    if (lineEnd < end && points[lineEnd] === '\n') fragment.append(document.createTextNode('\n'));
  }
  colorLean(fragment, model.syntax, start);
  return fragment;
}
document.querySelectorAll('.lean-context').forEach(details => {
  let loaded = false;
  details.addEventListener('toggle', async () => {
    if (!details.open || loaded) return;
    const pre = details.querySelector('pre');
    const code = pre.querySelector('code');
    pre.hidden = false;
    code.textContent = 'Loading Lean source…';
    try {
      const model = await leanSource(details.dataset.leanSource);
      code.replaceChildren(leanFragment(model));
      loaded = true;
    } catch (error) {
      code.textContent = error.message;
    }
  });
});
window.renderLeanMath = (root = document) => root.querySelectorAll('[data-tex], [data-latex]').forEach(element => {
  if (!window.katex) return;
  katex.render(element.dataset.tex ?? element.dataset.latex, element, {
    displayMode: !!element.closest('.math-display, .equation-block, .equation-stack'),
    throwOnError: false,
  });
  element.classList.add('katex-rendered');
});
renderLeanMath();

// Markdown protects dollar-delimited math and emits these unambiguous delimiters.
// Code, diagram labels, and already-rendered equations remain untouched.
if (window.renderMathInElement) renderMathInElement(document.body, {
  delimiters: [
    { left: '\\(', right: '\\)', display: false },
    { left: '\\[', right: '\\]', display: true },
    { left: '$$', right: '$$', display: true },
    { left: '\\begin{equation}', right: '\\end{equation}', display: true },
    { left: '\\begin{align}', right: '\\end{align}', display: true },
    { left: '\\begin{gather}', right: '\\end{gather}', display: true },
  ],
  ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'svg', 'canvas'],
  ignoredClasses: ['katex', 'katex-rendered'],
  throwOnError: false,
});

const reduction = document.querySelector('#reduction-example');
if (reduction) {
  const magnitude = reduction.querySelector('select');
  const output = reduction.querySelector('output');
  function update() {
    const a = Math.fround(Number(magnitude.value));
    const leftFirst = Math.fround(a + -a);
    const rightFirst = Math.fround(-a + 1);
    const left = Math.fround(leftFirst + 1);
    const right = Math.fround(a + rightFirst);
    const format = value => Math.abs(value) >= 1e10 ? value.toExponential(8) : String(value);
    output.replaceChildren();
    output.className = 'sum-trees';
    for (const [label, intermediate, result] of [
      ['Cancel the large terms first', leftFirst, left],
      ['Add the small term first', rightFirst, right],
    ]) {
      const row = document.createElement('section');
      row.className = 'sum-tree';
      const heading = document.createElement('h3');
      heading.textContent = label;
      const calculation = document.createElement('p');
      const isLeft = result === left && label.startsWith('Cancel');
      calculation.textContent = isLeft ? 'The large terms cancel exactly. The final addition keeps the one.'
        : rightFirst === -a ? 'The one disappears in the first rounding. The large terms then cancel to zero.'
        : 'The one survives the first addition in this example.';
      const diagram = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      diagram.setAttribute('viewBox', '0 0 340 230');
      diagram.setAttribute('role', 'img');
      diagram.setAttribute('aria-label', `${label}: first addition ${format(intermediate)}, final sum ${result}`);
      const middle = isLeft ? 105 : 235;
      const join = isLeft ? 'M55 48L105 96M155 48L105 96' : 'M185 48L235 96M285 48L235 96';
      const last = isLeft ? 'M105 133L170 180M285 48L170 180' : 'M235 133L170 180M55 48L170 180';
      const operands = isLeft ? [[55,'a'],[155,'−a'],[285,'1']] : [[55,'a'],[185,'−a'],[285,'1']];
      const color = result === 1 ? '#397959' : '#b16943';
      diagram.innerHTML = `<g fill="none" stroke="#a8b7c3" stroke-width="1.6"><path d="${join}"/><path d="${last}"/></g>
        <g font-family="IBM Plex Sans, sans-serif" font-size="19" text-anchor="middle" fill="#334655">
          ${operands.map(([x,value])=>`<text x="${x}" y="35">${value}</text>`).join('')}
        </g><circle cx="${middle}" cy="115" r="21" fill="#eef3f7" stroke="#c3d3df"/>
        <text x="${middle}" y="121" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="20" fill="#335a77">+</text>
        <text x="${isLeft?middle:265}" y="${isLeft?155:137}" text-anchor="${isLeft?'middle':'start'}" font-family="IBM Plex Sans, sans-serif" font-size="13" fill="#69737d">${isLeft?'0':(rightFirst===-a?'−a':format(rightFirst))}</text>
        <circle cx="170" cy="200" r="24" fill="${result===1?'#e8f3eb':'#f8eee7'}" stroke="${color}"/>
        <text x="170" y="207" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" font-size="23" fill="${color}">${result}</text>`;
      row.append(heading, diagram, calculation);
      output.append(row);
    }
  }
  magnitude.addEventListener('change', update);
  update();
}

const marginExample = document.querySelector('#margin-example');
if (marginExample) {
  const scores = marginExample.querySelector('select');
  const allowance = marginExample.querySelector('input');
  const output = marginExample.querySelector('output');
  const value = marginExample.querySelector('.margin-value');
  function updateMargin() {
    const gap = Number(scores.value);
    const error = Number(allowance.value);
    const winner = 8;
    const competitor = winner - gap;
    const padding = Math.max(gap * .25, error * .3, .0002);
    const low = competitor - error - padding;
    const high = winner + error + padding;
    const x = score => 110 + 410 * (score - low) / (high - low);
    const format = score => Number(score.toFixed(5)).toString();
    value.textContent = format(error);
    const guaranteed = gap > 2 * error;
    const diagram = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    diagram.setAttribute('viewBox', '0 0 600 200');
    diagram.setAttribute('role', 'img');
    diagram.setAttribute('aria-label', `Winner interval ${format(winner-error)} to ${format(winner+error)}; competitor interval ${format(competitor-error)} to ${format(competitor+error)}.`);
    diagram.innerHTML = `<g font-family="IBM Plex Sans, sans-serif" font-size="13" fill="#41505d">
      <text x="0" y="27">Winner: ${format(winner)}</text><text x="0" y="113">Other: ${format(competitor)}</text>
      <text x="315" y="194" text-anchor="middle">Score → (axis zooms to the selected example)</text></g>
      ${[[winner,57,'#397959'],[competitor,143,'#b16943']].map(([score,y,color]) => `
        <path d="M110 ${y}H520" stroke="#e0e5e8"/>
        <path d="M${x(score-error)} ${y}H${x(score+error)}" stroke="${color}" stroke-width="12" stroke-opacity=".3"/>
        <circle cx="${x(score)}" cy="${y}" r="5" fill="${color}"/>
        <g fill="${color}" font-family="IBM Plex Sans, sans-serif" font-size="12" text-anchor="middle">
          <text x="${x(score)}" y="${y+25}">${error === 0 ? format(score) : `${format(score-error)} … ${format(score+error)}`}</text></g>`).join('')}`;
    const verdict = document.createElement('p');
    verdict.className = 'margin-verdict';
    verdict.textContent = guaranteed
      ? `The winner stays ahead even in the worst case: ${format(winner-error)} > ${format(competitor+error)}. Margin ${format(gap)} > 2ε = ${format(2*error)}.`
      : `The guarantee no longer applies: the winner can fall to ${format(winner-error)}, while the other score can rise to ${format(competitor+error)}. Margin ${format(gap)} ≤ 2ε = ${format(2*error)}.`;
    output.replaceChildren(diagram, verdict);
  }
  scores.addEventListener('change', () => {
    const close = Number(scores.value) < 1;
    allowance.max = close ? '0.005' : '3';
    allowance.step = close ? '0.0001' : '0.001';
    allowance.value = close ? '0.0001' : '0.01';
    updateMargin();
  });
  allowance.addEventListener('input', updateMargin);
  updateMargin();
}

const prompts = document.querySelector('#prompt-example');
if (prompts) {
  const controls = [...prompts.querySelectorAll('select')];
  const output = prompts.querySelector('output');
  function update() {
    const [task, opener, ...content] = controls.map(control => Number(control.value));
    const quote = task === 2;
    const token = quote ? 9 + opener : 11 + opener;
    const target = quote ? token : 13 + opener;
    const alternative = quote ? 10 - opener : 14 - opener;
    output.replaceChildren();
    const line = document.createElement('div');
    line.className = 'prompt-line';
    const tokens = [1,task,content[0],token,content[1],content[2]];
    const roles = ['start','task','content','delimiter','content','content'];
    tokens.forEach((value,index)=>{
      const cell = document.createElement('span');
      cell.className = 'prompt-token'+(index===3?' delimiter':'');
      const label = document.createElement('small');
      label.textContent = roles[index];
      cell.append(String(value),label);line.append(cell);
    });
    const decision = document.createElement('p');
    decision.className = 'prompt-choice';
    decision.textContent = `The task requires token ${target} to score above token ${alternative}.`;
    output.append(line,decision);
  }
  controls.forEach(control => control.addEventListener('change', update));
  update();
}

const tokenFigure = document.querySelector('.token-figure');
if (tokenFigure) {
  const controls = document.createElement('nav');
  controls.className = 'mask-controls';
  controls.setAttribute('aria-label','Select a token to inspect the causal mask');
  const diagram = document.createElement('div');
  diagram.className = 'causal-mask';
  const words = ['The','cat','sat','on'];
  let selected = 2;
  const buttons = words.map((word,index)=>{
    const button = document.createElement('button');
    button.type = 'button';button.textContent = word;
    button.addEventListener('click',()=>{selected=index;renderMask();});
    controls.append(button);return button;
  });
  function renderMask() {
    buttons.forEach((button,index)=>button.setAttribute('aria-pressed',String(index===selected)));
    let cells = '';
    for(let row=0;row<4;row++)for(let column=0;column<4;column++){
      const active=row===selected,allowed=column<=row;
      cells+=`<rect x="${112+column*54}" y="${40+row*54}" width="48" height="48" rx="2" fill="${active?(allowed?'#588bad':'#f0e6e0'):(allowed?'#d9e6ee':'#f3f5f6')}"/>`;
      if(active)cells+=`<text x="${136+column*54}" y="${69+row*54}" text-anchor="middle" font-size="18" fill="${allowed?'white':'#ab7050'}">${allowed?'✓':'×'}</text>`;
    }
    diagram.innerHTML=`<svg viewBox="0 0 380 270" role="img" aria-label="Causal mask: ${words[selected]} can attend to itself and earlier tokens only"><g font-family="IBM Plex Sans,sans-serif" font-size="14" fill="#586c7b">${words.map((word,index)=>`<text x="${136+index*54}" y="25" text-anchor="middle">${word}</text><text x="98" y="${69+index*54}" text-anchor="end">${word}</text>`).join('')}</g>${cells}</svg><p>At “${words[selected]}”, the model may read ${words.slice(0,selected+1).map(w=>'“'+w+'”').join(', ')}. The shaded cells show access, not attention weights.</p>`;
  }
  const caption = tokenFigure.querySelector('figcaption');
  tokenFigure.insertBefore(controls,caption);tokenFigure.insertBefore(diagram,caption);renderMask();
}

const tensorRecurrence = document.querySelector('#tensor-recurrence');
if (tensorRecurrence) {
  const slider = tensorRecurrence.querySelector('input');
  const output = tensorRecurrence.querySelector('output');
  function renderRecurrence() {
    const count = Number(slider.value);
    tensorRecurrence.querySelector('.recurrence-count').textContent = String(count);
    output.replaceChildren();
    const results = [];
    for (const start of [1, 2]) {
      const lane = document.createElement('div');
      lane.className = 'recurrence-lane';
      const label = document.createElement('span');
      label.className = 'recurrence-label';
      label.textContent = `Start at ${start}`;
      const trace = document.createElement('div');
      trace.className = 'recurrence-trace';
      let value = Math.fround(start);
      for (let step = 0; step <= count; step++) {
        if (step) {
          const arrow = document.createElement('span');
          arrow.className = 'recurrence-arrow';
          arrow.textContent = '→';
          arrow.setAttribute('aria-hidden', 'true');
          trace.append(arrow);
          // Match the example's separate binary32 multiplication and addition.
          value = Math.fround(Math.fround(value * value) + 1);
        }
        const cell = document.createElement('span');
        cell.className = 'recurrence-value';
        cell.textContent = String(value);
        cell.setAttribute('aria-label', `${step} updates: ${value}`);
        trace.append(cell);
      }
      results.push(value);
      lane.append(label, trace);
      output.append(lane);
    }
    const result = document.createElement('p');
    result.className = 'recurrence-result';
    result.textContent = `After ${count} ${count === 1 ? 'update' : 'updates'}: [${results.join(', ')}]`;
    output.append(result);
  }
  slider.addEventListener('input', renderRecurrence);
  renderRecurrence();
}
