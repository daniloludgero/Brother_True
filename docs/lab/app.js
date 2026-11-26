/**
 * BrotherTrue Interactive Biocognitive & Epistemic Workbench (Pure Deterministic JS)
 * Mirrors the exact mathematical contracts of ode.py, farmacologia.py, evidence_gate.py,
 * thought_renderer.py, and cot_parser.py with 7-state UI resilience and zero external deps.
 */
(function () {
  "use strict";

  const HILL_CONSTANTS = {
    dopamina:       { kd: 45.0,  n: 1.35 },
    serotonina:     { kd: 18.0,  n: 1.20 },
    noradrenalina:  { kd: 32.0,  n: 1.40 },
    gaba:           { kd: 60.0,  n: 1.55 }
  };

  const state = {
    turn: 0,
    prevL: 1.00,
    prevQ: 1.00,
    prevF: 0.00,
    status: "default" // default | loading | empty | error | recovery
  };

  function hillOccupancy(concNm, kd, n) {
    if (concNm <= 0) return 0.0;
    const cn = Math.pow(concNm, n);
    const kdn = Math.pow(kd, n);
    return cn / (kdn + cn);
  }

  function sanitizeEvidence(raw) {
    return String(raw || "")
      .replace(/<\/?(?:thought|think)>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>/gi, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function runSimulationStep() {
    const btn = document.getElementById("btn-simulate");
    btn.setAttribute("aria-busy", "true");
    state.status = "loading";

    const driveH = parseFloat(document.getElementById("inp-drive").value);
    const age = parseFloat(document.getElementById("inp-age").value);
    const dopaNm = parseFloat(document.getElementById("inp-dopa").value);
    const gabaNm = parseFloat(document.getElementById("inp-gaba").value);
    const ragScore = parseFloat(document.getElementById("inp-rag-score").value);
    const mode = document.getElementById("inp-mode").value;
    const rawPassage = document.getElementById("inp-passage").value;

    document.getElementById("val-drive").textContent = driveH.toFixed(2);
    document.getElementById("val-age").textContent = age.toFixed(2);
    document.getElementById("val-dopa").textContent = dopaNm.toFixed(0) + " nM";
    document.getElementById("val-gaba").textContent = gabaNm.toFixed(0) + " nM";
    document.getElementById("val-rag-score").textContent = ragScore.toFixed(2);

    // 1. Hill Receptor Occupancies (farmacologia.py)
    const occDopa = hillOccupancy(dopaNm, HILL_CONSTANTS.dopamina.kd, HILL_CONSTANTS.dopamina.n);
    const occGaba = hillOccupancy(gabaNm, HILL_CONSTANTS.gaba.kd, HILL_CONSTANTS.gaba.n);
    const occSero = hillOccupancy(22.0 + 18.0 * occGaba, HILL_CONSTANTS.serotonina.kd, HILL_CONSTANTS.serotonina.n);
    const occNora = hillOccupancy(15.0 + 35.0 * Math.max(0, driveH - 1.2), HILL_CONSTANTS.noradrenalina.kd, HILL_CONSTANTS.noradrenalina.n);

    // 2. QualiaODE Integration & Mandatory Collapse Damping (ode.py)
    const E0 = 2.40;
    const etaE = 1.60;
    const eCrit = E0 + etaE * age;

    let rawL = state.prevL * 0.72 + driveH * (0.95 + 0.55 * occDopa) - 0.30 * occGaba;
    let rawQ = state.prevQ * 0.74 + driveH * (0.85 + 0.40 * occSero) - 0.25 * occGaba;
    let rawF = Math.max(0.0, state.prevF * 0.80 + 0.12 * driveH);
    rawL = Math.max(0.10, rawL);
    rawQ = Math.max(0.10, rawQ);

    const rawE = 0.5 * rawL + 0.5 * rawQ;
    let collapsed = false;
    let finalL = rawL;
    let finalQ = rawQ;
    let finalF = rawF;

    if (rawE > eCrit) {
      collapsed = true;
      const kc = 0.55;
      finalL = rawL * Math.exp(-kc);
      finalQ = rawQ * Math.exp(-kc);
      finalF = rawF + 2.0;
    }
    const finalE = 0.5 * finalL + 0.5 * finalQ;

    // 3. Causal t-1 Introspection (Pensamento) & Epistemic KL Surprise (Palavras)
    const hatL = state.prevL;
    const hatQ = state.prevQ;
    const hatF = state.prevF;
    const lossP = state.turn === 0
      ? 0.0
      : ((Math.pow(finalL - hatL, 2) + Math.pow(finalQ - hatQ, 2) + Math.pow(finalF - hatF, 2)) / 3.0);

    const klDiv = Math.abs(finalL - hatL) * 0.42 + Math.abs(occNora - occGaba) * 0.65;
    const surprise = klDiv > 0.50;

    // Append current state strictly AFTER prediction (Zero Target Leakage)
    state.prevL = finalL;
    state.prevQ = finalQ;
    state.prevF = finalF;
    state.turn += 1;

    // 4. Node 05: RAG Evidence Gate & Tag-Injection Sanitizer
    const cleanPassage = sanitizeEvidence(rawPassage);
    const minConf = 0.35;
    let gateVerdict = "insufficient";
    let gateAccepted = false;

    if (!cleanPassage) {
      state.status = "empty";
      gateVerdict = "insufficient (empty passage)";
    } else if (ragScore >= minConf) {
      gateAccepted = true;
      gateVerdict = mode === "grounded" ? "supported (score >= 0.35)" : mode + "_contextual";
      state.status = "default";
    } else {
      gateAccepted = false;
      gateVerdict = "rejected_below_confidence";
      state.status = "recovery";
    }

    const directiveMap = {
      grounded: gateAccepted
        ? "Responder somente o sustentado pela evidência aprovada, declarando limites."
        : "Evidência insuficiente: abster-se de afirmação documental e declarar lacuna (ζ ≠ 0).",
      general: "Modo geral: distinguir inferência paramétrica de fatos documentados.",
      creative: "Modo criativo: hipótese ou abstração permitida, marcada explicitamente como tal."
    };
    const directive = directiveMap[mode] || directiveMap.grounded;

    // 5. Node 06: Auditable Symbolic <thought> Projection
    const thoughtProjection = [
      "<thought>",
      `[Homeostase Qualia: L=${finalL.toFixed(2)}, Q=${finalQ.toFixed(2)}, F=${finalF.toFixed(2)}, E=${finalE.toFixed(2)}, E_crit=${eCrit.toFixed(2)}]`,
      `[Receptores Hill (nM): Dopa=${occDopa.toFixed(3)}, GABA=${occGaba.toFixed(3)}, Sero=${occSero.toFixed(3)}, Nora=${occNora.toFixed(3)}]`,
      `[Predição Causal t-1 (Pensamento): L_hat=${hatL.toFixed(2)}, Q_hat=${hatQ.toFixed(2)}, loss_p=${lossP.toFixed(4)}]`,
      `[Surpresa Epistêmica (Palavras): KL=${klDiv.toFixed(4)}, surprise=${surprise}]`,
      `[Evidence Gate (${mode}): accepted=${gateAccepted}, verdict=${gateVerdict}]`,
      gateAccepted ? `  - (banco_vetorial, verificada) ${cleanPassage}` : "  - nenhuma evidência aprovada",
      `[Diretriz: ${directive}]`,
      "</thought>"
    ].join("\n");

    // 6. Node 08: Public Clean Verbal Channel (Zero-Leakage Preview)
    let publicSpeech = "";
    if (mode === "grounded" && !gateAccepted) {
      publicSpeech = "Não há evidência documental aprovada pelo Evidence Gate (confiança abaixo de 0.35 ou fonte vazia). Prefiro declarar essa lacuna com transparência a inventar um fato não verificado.";
    } else if (mode === "creative") {
      publicSpeech = `[Hipótese Criativa Declarada] Sob estado interno equilibrado, podemos explorar a analogia proposta: ${cleanPassage || "um cenário aberto de mediação simbólica"}.`;
    } else {
      publicSpeech = `Com base na passagem verificada pelo Gate ("${cleanPassage}"), o sistema mantém consistência contextual sem expor tags internas ou telemetria bruta.`;
    }

    // Update DOM Telemetry
    document.getElementById("m-energy").textContent = `${finalE.toFixed(2)} / ${eCrit.toFixed(2)}`;
    document.getElementById("m-lqf").textContent = `${finalL.toFixed(2)} · ${finalQ.toFixed(2)} · ${finalF.toFixed(2)}`;
    document.getElementById("m-lossp").textContent = lossP.toFixed(4);
    document.getElementById("m-kl").textContent = `${klDiv.toFixed(4)} (${surprise ? "SURPRISE" : "CALM"})`;

    const badge = document.getElementById("m-collapse-badge");
    if (collapsed) {
      badge.className = "status-pill status-collapse";
      badge.textContent = `COLLAPSE DAMPED (raw E=${rawE.toFixed(2)} > E_crit)`;
    } else {
      badge.className = "status-pill status-ok";
      badge.textContent = "HOMEOSTATIC HALO INTACT (E ≤ E_crit)";
    }

    document.getElementById("out-thought").textContent = thoughtProjection;
    document.getElementById("out-speech").textContent = publicSpeech;
    btn.setAttribute("aria-busy", "false");
  }

  document.addEventListener("DOMContentLoaded", function () {
    const ids = ["inp-drive", "inp-age", "inp-dopa", "inp-gaba", "inp-rag-score", "inp-mode", "inp-passage"];
    ids.forEach(function (id) {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", runSimulationStep);
    });
    const btn = document.getElementById("btn-simulate");
    if (btn) btn.addEventListener("click", runSimulationStep);
    runSimulationStep();
  });
})();
