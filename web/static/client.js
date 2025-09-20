// web/static/js/client.js
(function () {
  // ===== Socket =====
  const socket = window.socket || io({ transports: ["websocket"] });
  window.socket = socket;

  // ===== Elements =====
  const svg = document.getElementById("board");
  const wrap = document.getElementById("boardWrap");
  const statusEl = document.getElementById("status");
  const btnWhite = document.getElementById("solo-white");
  const btnBlack = document.getElementById("solo-black");
  const btnFinish = document.getElementById("finish");

  const PIECES_BASE = "/static/pieces/";
  const PIECE_TO_FILE = {
    wP: "wp.svg", wR: "wr.svg", wN: "wn.svg", wB: "wb.svg", wQ: "wq.svg", wK: "wk.svg",
    bP: "bp.svg", bR: "br.svg", bN: "bn.svg", bB: "bb.svg", bQ: "bq.svg", bK: "bk.svg",
  };

  // ===== State =====
  let position = startPosition();
  let currentGameId = null;
  let myColor = "white";
  let dragging = null;
  let lastMove = null;

  // ===== Build board (squares + layers + events) =====
  buildSquares();
  renderPieces(position);

  // ===== Buttons =====
  btnWhite.addEventListener("click", () => startSolo("white"));
  btnBlack.addEventListener("click", () => startSolo("black"));
  btnFinish.addEventListener("click", finishGame);

  // ===== Socket events =====
  socket.on("ai_move", (data) => {
    if (!data || data.game_id !== currentGameId) return;
    applyUciOnPosition(position, data.move_uci);
    renderPieces(position);
    statusEl.textContent = `AI moved: ${data.move_uci}`;
  });

  // (اختياري) لو السيرفر بعت explain state في أي وقت
  socket.on("solo_state", (data) => {
    if (data && data.your_color) {
      const isWhite = (data.your_color === "white");
      wrap.classList.toggle("orient-white", isWhite);
      wrap.classList.toggle("orient-black", !isWhite);
    }
    hydrateExplain(data && data.explain);
  });

  // ===== Actions =====
  function startSolo(color) {
  myColor = color;
  wrap.classList.toggle("orient-white", color === "white");
  wrap.classList.toggle("orient-black", color === "black");
  socket.emit("new_game", { color }, (resp) => {
    if (!resp || resp.ok !== true || !resp.game_id) {
      statusEl.textContent = "Error starting game.";
      return;
    }
    currentGameId = resp.game_id;
    position = startPosition();
    lastMove = null;
    renderPieces(position);
    statusEl.textContent = `Game started (${resp.your_color || myColor})`;

    // ✅ لو أنت بتلعب أسود، السيرفر بيرجع أول نقلة للـAI هنا
    if (resp.ai_move) {
      applyUciOnPosition(position, resp.ai_move);
      renderPieces(position);
      statusEl.textContent = `AI moved: ${resp.ai_move}`;
    }
  });
}

  function finishGame() {
    if (!currentGameId) return (statusEl.textContent = "No game.");
    fetch("/api/finish_game", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: currentGameId })
    })
      .then(r => r.json())
      .then(d => {
        if (d.ok) {
          statusEl.textContent = d.mail_error ? `Report ready (email error: ${d.mail_error})` : "Report generated & emailed.";
        } else {
          statusEl.textContent = "Error: " + (d.error || "");
        }
      })
      .catch(e => statusEl.textContent = "Error: " + e);
  }

  // ===== Board building/rendering =====
  function buildSquares() {
    const ns = "http://www.w3.org/2000/svg";
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const rect = document.createElementNS(ns, "rect");
        rect.setAttribute("x", c);
        rect.setAttribute("y", r);
        rect.setAttribute("width", 1);
        rect.setAttribute("height", 1);
        rect.setAttribute("class", "square " + (((r + c) % 2) ? "dark" : "light"));
        svg.appendChild(rect);
      }
    }
    const hl = document.createElementNS(ns, "g");
    hl.setAttribute("id", "last-move");
    svg.appendChild(hl);

    const layer = document.createElementNS(ns, "g");
    layer.setAttribute("id", "pieces");
    svg.appendChild(layer);

    svg.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
  }

  function renderPieces(pos) {
    const layer = document.getElementById("pieces");
    while (layer.firstChild) layer.removeChild(layer.firstChild);

    for (const sq of allSquares()) {
      const p = pos[sq];
      if (!p) continue;
      const { x, y } = squareToXY(sq);
      const href = PIECES_BASE + (PIECE_TO_FILE[p] || "");
      if (!href) continue;

      const img = document.createElementNS("http://www.w3.org/2000/svg", "image");
      img.setAttribute("href", href);
      img.setAttribute("class", "piece");
      img.setAttribute("data-square", sq);
      img.setAttribute("data-piece", p);
      img.setAttribute("x", x);
      img.setAttribute("y", y);
      img.setAttribute("width", 1);
      img.setAttribute("height", 1);
      img.style.cursor = canDragPiece(p) ? "grab" : "default";
      layer.appendChild(img);
    }
    renderLastMove();
  }

  function renderLastMove() {
    const layer = document.getElementById("last-move");
    while (layer.firstChild) layer.removeChild(layer.firstChild);
    if (!lastMove) return;
    const ns = "http://www.w3.org/2000/svg";
    const fromXY = squareToXY(lastMove.from);
    const toXY = squareToXY(lastMove.to);

    const rf = document.createElementNS(ns, "rect");
    rf.setAttribute("x", fromXY.x);
    rf.setAttribute("y", fromXY.y);
    rf.setAttribute("width", 1);
    rf.setAttribute("height", 1);
    rf.setAttribute("class", "last from");
    layer.appendChild(rf);

    const rt = document.createElementNS(ns, "rect");
    rt.setAttribute("x", toXY.x);
    rt.setAttribute("y", toXY.y);
    rt.setAttribute("width", 1);
    rt.setAttribute("height", 1);
    rt.setAttribute("class", "last to");
    layer.appendChild(rt);
  }

  // ===== Drag & drop =====
  function onPointerDown(e) {
    if (e.button !== 0) return;
    const target = e.target;
    if (!(target && target.classList && target.classList.contains("piece"))) return;

    const piece = target.getAttribute("data-piece");
    if (!canDragPiece(piece)) return;

    const from = target.getAttribute("data-square");
    const pt = clientToBoardPoint(e);
    const { x, y } = squareToXY(from);

    dragging = {
      piece,
      from,
      el: target,
      dx: pt.x - x,
      dy: pt.y - y,
      originalX: x,
      originalY: y
    };
    target.style.pointerEvents = "none";
  }

  function onPointerMove(e) {
    if (!dragging) return;
    const pt = clientToBoardPoint(e);
    dragging.el.setAttribute("x", clamp(pt.x - dragging.dx, 0, 7));
    dragging.el.setAttribute("y", clamp(pt.y - dragging.dy, 0, 7));
  }

  function onPointerUp(e) {
    if (!dragging) return;
    const target = dragging.el;
    target.style.pointerEvents = "auto";
    const from = dragging.from;
    const pt = clientToBoardPoint(e);
    const to = xyToSquare(
      Math.round(clamp(pt.x - dragging.dx + 0.5, 0, 7)),
      Math.round(clamp(pt.y - dragging.dy + 0.5, 0, 7))
    );
    const promo = willPromote(dragging.piece, from, to) ? "q" : "";
    const uci = buildUci(from, to, promo);

    const snapshot = clonePosition(position);
    const ok = applyUciOnPosition(position, uci);
    dragging = null;

    if (!ok) {
      position = snapshot;
      renderPieces(position);
      statusEl.textContent = "Illegal (client).";
      return;
    }
    renderPieces(position);
    statusEl.textContent = "You: " + uci;

    // أرسل للسيرفر مع انتظار ACK
    socket.emit("user_move", { game_id: currentGameId, move_uci: uci }, (resp) => {
      if (!resp || resp.ok !== true) {
        position = snapshot;
        renderPieces(position);
        const reason = (resp && resp.reason) || (resp && resp.error) || "Illegal move.";
        statusEl.textContent = reason;
      }
    });
  }

  // ===== Explain panel =====
  function hydrateExplain(ex){
    const $ = id => document.getElementById(id);
    if (!ex) return;
    $('net-value').textContent = (ex.net_value ?? '—');
    $('sf-eval').textContent  = (ex.sf_eval_cp ?? '—');

    const mTop = $('mcts-top'); mTop.innerHTML = '';
    if (ex.mcts && ex.mcts.visits){
      Object.entries(ex.mcts.visits)
        .sort((a,b)=>b[1]-a[1]).slice(0,5)
        .forEach(([uci,n])=>{
          const li = document.createElement('li');
          li.textContent = `${uci}: ${n}`;
          mTop.appendChild(li);
        });
    }

    const pTop = $('policy-top'); pTop.innerHTML = '';
    if (Array.isArray(ex.net_top)){
      ex.net_top.slice(0,5).forEach(([uci,p])=>{
        const li = document.createElement('li');
        li.textContent = `${uci}: ${(p*100).toFixed(1)}%`;
        pTop.appendChild(li);
      });
    }
  }

  // ===== Helpers =====
  function startPosition() {
    const pos = {};
    const back = ["R","N","B","Q","K","B","N","R"];
    for (let i = 0; i < 8; i++) {
      pos[fileChar(i) + "2"] = "wP";
      pos[fileChar(i) + "7"] = "bP";
      pos[fileChar(i) + "1"] = "w" + back[i];
      pos[fileChar(i) + "8"] = "b" + back[i];
    }
    return pos;
  }
  function clonePosition(pos) { const n = {}; for (const k in pos) n[k] = pos[k]; return n; }
  function buildUci(from, to, promo) { return from + to + (promo ? promo : ""); }
  function willPromote(piece, from, to) {
    if (!piece) return false;
    const isPawn = piece[1] === "P";
    if (!isPawn) return false;
    const rankTo = to[1];
    return (piece[0] === "w" && rankTo === "8") || (piece[0] === "b" && rankTo === "1");
  }
  function applyUciOnPosition(pos, uci) {
    if (!uci || uci.length < 4) return false;
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promo = uci.slice(4).toLowerCase();

    const moving = pos[from];
    if (!moving) return false;

    const isWhite = moving[0] === "w";
    const isPawn = moving[1] === "P";
    const isKing = moving[1] === "K";

    let isCastle = false, isEP = false;

    if (isKing && from[0] === "e" && (to[0] === "g" || to[0] === "c")) isCastle = true;
    if (isPawn && from[0] !== to[0] && !pos[to]) isEP = true;

    const capture = !isEP && !!pos[to];

    delete pos[from];

    if (isEP) {
      const dir = isWhite ? -1 : +1;
      const victimRank = String.fromCharCode(to.charCodeAt(1) + dir);
      const victimSq = to[0] + victimRank;
      delete pos[victimSq];
      pos[to] = moving;
    } else if (isCastle) {
      pos[to] = moving;
      if (to[0] === "g") {
        const rookFrom = "h" + from[1];
        const rookTo = "f" + from[1];
        if (pos[rookFrom]) { pos[rookTo] = pos[rookFrom]; delete pos[rookFrom]; }
      } else if (to[0] === "c") {
        const rookFrom = "a" + from[1];
        const rookTo = "d" + from[1];
        if (pos[rookFrom]) { pos[rookTo] = pos[rookFrom]; delete pos[rookFrom]; }
      }
    } else {
      let pieceToPlace = moving;
      if (isPawn && (to[1] === "1" || to[1] === "8") && promo) {
        const map = { q:"Q", r:"R", b:"B", n:"N" };
        pieceToPlace = moving[0] + (map[promo] || "Q");
      }
      pos[to] = pieceToPlace;
    }

    lastMove = { from, to, piece: moving, capture, isEP, isCastle, promo };
    return true;
  }

  // ---- Geometry / orientation ----
  function squareToXY(sq) {
    const file = sq.charCodeAt(0) - 97; // a=0
    const rank = parseInt(sq[1], 10) - 1; // 1..8 -> 0..7
    let x = file, y = 7 - rank; // white at bottom
    if (myColor === "black") { x = 7 - x; y = 7 - y; }
    return { x, y };
  }
  function xyToSquare(x, y) {
    if (myColor === "black") { x = 7 - x; y = 7 - y; }
    const file = String.fromCharCode(97 + x);
    const rank = String(8 - y);
    return file + rank;
  }
  function allSquares() {
    const arr = [];
    for (let r = 8; r >= 1; r--) for (let f = 0; f < 8; f++) arr.push(String.fromCharCode(97 + f) + r);
    return arr;
  }
  function canDragPiece(p) { return (myColor === "white" && p[0] === "w") || (myColor === "black" && p[0] === "b"); }
  function clientToBoardPoint(e) {
    const rect = svg.getBoundingClientRect();
    const s = Math.min(rect.width, rect.height);
    const bx = (e.clientX - rect.left) / (s / 8);
    const by = (e.clientY - rect.top) / (s / 8);
    return { x: bx, y: by };
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function fileChar(i) { return String.fromCharCode(97 + i); }
})();
