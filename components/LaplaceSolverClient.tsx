"use client";

import { useMemo, useState } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
} from "recharts";

type BoundaryKey = "top" | "right" | "bottom" | "left";
type GridValue = number[][];

interface PointHistory {
  iteration: number;
  P1: number;
  P2: number;
  P3: number;
  P4: number;
}

const GRID_SIZE = 4;

const DEFAULT_BOUNDARIES: Record<BoundaryKey, number> = {
  top: 80,
  right: 50,
  bottom: 40,
  left: 10,
};

const CHART_COLORS = ["#000000", "#525252", "#a3a3a3", "#d4d4d4"];

function makeInitialGrid(boundaries: Record<BoundaryKey, number>): GridValue {
  const grid: GridValue = Array.from({ length: GRID_SIZE }, () =>
    Array.from({ length: GRID_SIZE }, () => 0)
  );
  for (let c = 0; c < GRID_SIZE; c++) {
    grid[0][c] = boundaries.top;
    grid[GRID_SIZE - 1][c] = boundaries.bottom;
  }
  for (let r = 0; r < GRID_SIZE; r++) {
    grid[r][0] = boundaries.left;
    grid[r][GRID_SIZE - 1] = boundaries.right;
  }
  return grid;
}

/* Frio → claro, Quente → escuro (paleta B&W computacional). */
function heatColor(temp: number, min: number, max: number): string {
  const span = Math.max(max - min, 1);
  const t = Math.max(0, Math.min(1, (temp - min) / span));
  const v = Math.round(245 - t * 220);
  return `rgb(${v}, ${v}, ${v})`;
}

function textColorForTemperature(temp: number, min: number, max: number): string {
  const span = Math.max(max - min, 1);
  const t = Math.max(0, Math.min(1, (temp - min) / span));
  return t > 0.55 ? "#ffffff" : "#18181b";
}

function nodeLabel(r: number, c: number): string {
  if (r === 0 && c < 3) return "T";
  if (r === 3 && c < 3) return "B";
  if (c === 0 && r < 3) return "L";
  if (c === 3 && r < 3) return "R";
  if (r === 1 && c === 1) return "P1";
  if (r === 1 && c === 2) return "P2";
  if (r === 2 && c === 1) return "P3";
  if (r === 2 && c === 2) return "P4";
  return "";
}

function isBoundaryNode(r: number, c: number): boolean {
  return r === 0 || c === 0 || r === GRID_SIZE - 1 || c === GRID_SIZE - 1;
}

function reduceToDisplay(history: PointHistory[], maxPoints = 200) {
  if (history.length <= maxPoints) return history.map((h) => ({ ...h }));
  const step = Math.ceil(history.length / maxPoints);
  const reduced: PointHistory[] = [];
  for (let i = 0; i < history.length; i += step) {
    reduced.push({ ...history[i] });
  }
  if (reduced[reduced.length - 1]?.iteration !== history[history.length - 1].iteration) {
    reduced.push({ ...history[history.length - 1] });
  }
  return reduced;
}

export default function LaplaceSolverClient() {
  const [boundaries, setBoundaries] =
    useState<Record<BoundaryKey, number>>(DEFAULT_BOUNDARIES);
  const [grid, setGrid] = useState<GridValue>(() => makeInitialGrid(DEFAULT_BOUNDARIES));
  const [tolerance, setTolerance] = useState(0.001);
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState<PointHistory[]>([
    { iteration: 0, P1: 0, P2: 0, P3: 0, P4: 0 },
  ]);
  const [residual, setResidual] = useState(0);

  const minTemp = useMemo(() => Math.min(...grid.flat()), [grid]);
  const maxTemp = useMemo(() => Math.max(...grid.flat()), [grid]);

  const stepsTaken = Math.max(history.length - 1, 0);
  const converged = residual < tolerance && iteration > 0;

  function refreshFromBoundaries(next: Record<BoundaryKey, number>) {
    const g = makeInitialGrid(next);
    setGrid(g);
    setIteration(0);
    setResidual(0);
    setHistory([{ iteration: 0, P1: g[1][1], P2: g[1][2], P3: g[2][1], P4: g[2][2] }]);
  }

  function handleBoundaryChange(key: BoundaryKey, value: string) {
    const num = Number(value);
    if (Number.isNaN(num)) return;
    const next = { ...boundaries, [key]: num };
    setBoundaries(next);
    refreshFromBoundaries(next);
  }

  function resetToBoundaries() {
    refreshFromBoundaries(boundaries);
  }

  function runSingleStep() {
    const g = grid.map((row) => [...row]);
    let maxDiff = 0;
    for (let r = 1; r < GRID_SIZE - 1; r++) {
      for (let c = 1; c < GRID_SIZE - 1; c++) {
        const newVal = (g[r - 1][c] + g[r + 1][c] + g[r][c - 1] + g[r][c + 1]) / 4;
        maxDiff = Math.max(maxDiff, Math.abs(newVal - g[r][c]));
        g[r][c] = newVal;
      }
    }
    const newIteration = iteration + 1;
    setGrid(g);
    setIteration(newIteration);
    setResidual(maxDiff);
    setHistory((prev) => [
      ...prev,
      { iteration: newIteration, P1: g[1][1], P2: g[1][2], P3: g[2][1], P4: g[2][2] },
    ]);
  }

  function runToConvergence() {
    const g = grid.map((row) => [...row]);
    let localIteration = iteration;
    let localResidual = Number.POSITIVE_INFINITY;
    const newHistory: PointHistory[] = [];
    const guard = 100000;
    let cycle = 0;
    while (localResidual > tolerance && cycle < guard) {
      cycle++;
      let maxDiff = 0;
      for (let r = 1; r < GRID_SIZE - 1; r++) {
        for (let c = 1; c < GRID_SIZE - 1; c++) {
          const newVal = (g[r - 1][c] + g[r + 1][c] + g[r][c - 1] + g[r][c + 1]) / 4;
          maxDiff = Math.max(maxDiff, Math.abs(newVal - g[r][c]));
          g[r][c] = newVal;
        }
      }
      localIteration++;
      localResidual = maxDiff;
      newHistory.push({
        iteration: localIteration,
        P1: g[1][1],
        P2: g[1][2],
        P3: g[2][1],
        P4: g[2][2],
      });
    }
    setGrid(g);
    setIteration(localIteration);
    setResidual(localResidual);
    if (newHistory.length > 0) setHistory((prev) => [...prev, ...newHistory]);
  }

  const displayHistory = useMemo(() => reduceToDisplay(history, 200), [history]);

  const nodeCoords: Record<string, [number, number]> = {
    P1: [1, 1],
    P2: [1, 2],
    P3: [2, 1],
    P4: [2, 2],
  };

  return (
    <main className="min-h-screen bg-white text-zinc-950">
      <div className="mx-auto max-w-7xl px-6 py-10 lg:px-10">
        {/* Header */}
        <header className="mb-10 border-b border-zinc-200 pb-8">
          <div>
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-950 text-xs font-bold text-white">
                ∇²
              </div>
              <span className="font-mono text-xs uppercase tracking-[0.25em] text-zinc-500">
                Física Numérica / Resolvedor de Laplace
              </span>
            </div>

            <div className="flex flex-wrap items-end justify-between gap-6">
              <div>
                <h1 className="text-4xl font-semibold tracking-[-0.04em] md:text-5xl">
                  Transferência de Calor 2D
                </h1>
                <p className="mt-3 max-w-2xl font-mono text-sm leading-6 text-zinc-500">
                  Resolução da Equação de Laplace (∇²T = 0) por diferenças finitas
                  via método de Gauss-Seidel, com atualização sequencial dos nós
                  internos P1–P4.
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* Main grid */}
        <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
          {/* Parameters sidebar */}
          <div className="space-y-6">
            <section className="rounded-2xl border border-zinc-200 bg-zinc-50/60 p-6">
              <div className="mb-7">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  ENTRADA / 01
                </div>
                <h2 className="mt-1 text-lg font-semibold">Condições de contorno</h2>
              </div>

              <div className="space-y-5">
                <NumberInput
                  label="Superior"
                  value={boundaries.top}
                  onChange={(v) => handleBoundaryChange("top", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Direita"
                  value={boundaries.right}
                  onChange={(v) => handleBoundaryChange("right", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Inferior"
                  value={boundaries.bottom}
                  onChange={(v) => handleBoundaryChange("bottom", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Esquerda"
                  value={boundaries.left}
                  onChange={(v) => handleBoundaryChange("left", String(v))}
                  suffix="°C"
                />
              </div>

              <div className="my-7 h-px bg-zinc-200" />

              <div className="mb-5">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  DOMÍNIO NUMÉRICO
                </div>
                <h3 className="mt-1 text-sm font-semibold">Configuração do solver</h3>
              </div>

              <label className="block">
                <span className="mb-2 block font-mono text-[10px] uppercase tracking-wider text-zinc-400">
                  Tolerância (ε)
                </span>
                <select
                  value={tolerance}
                  onChange={(e) => setTolerance(Number(e.target.value))}
                  className="w-full rounded-xl border border-zinc-200 bg-white px-3 py-3 font-mono text-sm outline-none transition focus:border-zinc-950 focus:ring-1 focus:ring-zinc-950"
                >
                  <option value={0.01}>0.01</option>
                  <option value={0.001}>0.001</option>
                  <option value={0.0001}>0.0001</option>
                  <option value={0.00001}>0.00001</option>
                </select>
              </label>

              <button
                onClick={resetToBoundaries}
                className="mt-7 w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider transition hover:border-zinc-950 hover:bg-zinc-950 hover:text-white"
              >
                Redefinir nós
              </button>
            </section>

            {/* Actions */}
            <section className="rounded-2xl border border-zinc-200 bg-zinc-50/60 p-6">
              <div className="mb-6">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  EXECUÇÃO / 02
                </div>
                <h2 className="mt-1 text-lg font-semibold">Método de Gauss-Seidel</h2>
              </div>

              <div className="space-y-3">
                <button
                  onClick={runSingleStep}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-950 px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider text-white transition hover:bg-zinc-800"
                >
                  ▶ Executar passo a passo
                </button>
                <button
                  onClick={runToConvergence}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-white px-4 py-3 font-mono text-xs font-medium uppercase tracking-wider transition hover:border-zinc-950 hover:bg-zinc-950 hover:text-white"
                >
                  ≫ Calcular até convergência
                </button>
              </div>

              <div className="my-6 grid grid-cols-3 gap-px overflow-hidden rounded-xl border border-zinc-200 bg-zinc-200">
                <SummaryStat label="Iterações" value={String(iteration)} />
                <SummaryStat label="Residual" value={residual.toExponential(2)} />
                <SummaryStat
                  label="Status"
                  value={converged ? "✓ ok" : iteration === 0 ? "init" : "…"}
                />
              </div>
            </section>
          </div>

          {/* Main results */}
          <div className="min-w-0 space-y-6">
            {/* Equation */}
            <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-white">
              <div className="border-b border-zinc-200 px-6 py-4">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  EQUAÇÃO / 03
                </div>
              </div>

              <div className="bg-zinc-950 px-6 py-10 text-white md:px-10">
                <div className="flex flex-wrap items-baseline gap-x-3 font-mono text-lg tracking-tight md:text-2xl">
                  <span className="text-zinc-400">P_i</span>
                  <span className="text-zinc-600">=</span>
                  <span className="text-zinc-400">(</span>
                  <span>P_N</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_S</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_L</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_O</span>
                  <span className="text-zinc-400">)</span>
                  <span className="text-zinc-600">/</span>
                  <span>4</span>
                </div>

                <p className="mt-3 max-w-xl font-mono text-sm leading-6 text-zinc-400">
                  Cada nó interno é a média aritmética dos seus quatro vizinhos
                  imediatos, atualizada sequencialmente (Gauss-Seidel).
                </p>

                <div className="mt-8 flex flex-wrap items-center gap-3 font-mono text-xs">
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    ∇²T = 0
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    malha {GRID_SIZE}×{GRID_SIZE}
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    ε = {tolerance}
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    nós internos = 4
                  </span>
                </div>
              </div>
            </section>

            {/* Result + Plate */}
            <section className="grid gap-6 md:grid-cols-2">
              <div className="relative overflow-hidden rounded-2xl bg-zinc-950 p-6 text-white">
                <div className="absolute right-5 top-5 font-mono text-[9px] uppercase tracking-[0.2em] text-zinc-600">
                  SISTEMA LINEAR
                </div>
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500">
                  VALORES CONVERGIDOS
                </div>
                <div className="mt-4 space-y-2 font-mono">
                  {(["P1", "P2", "P3", "P4"] as const).map((p) => {
                    const [r, c] = nodeCoords[p];
                    return (
                      <div key={p} className="flex items-center justify-between">
                        <span className="text-zinc-400">{p}</span>
                        <span className="text-lg font-semibold tracking-tight">
                          {grid[r][c].toFixed(2)} °C
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50 p-6">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  MALHA / 04
                </div>
                <p className="mt-1 font-semibold">Mapa de calor (placa)</p>

                <div className="mt-5 flex items-center gap-3 font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                  <span>frio</span>
                  <div className="h-1.5 flex-1 bg-gradient-to-r from-zinc-200 to-zinc-900" />
                  <span>quente</span>
                </div>

                <div
                  className="mt-3 grid gap-1"
                  style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)` }}
                >
                  {grid.map((row, r) =>
                    row.map((value, c) => {
                      const bg = heatColor(value, minTemp, maxTemp);
                      const fg = textColorForTemperature(value, minTemp, maxTemp);
                      const boundary = isBoundaryNode(r, c);
                      const label = nodeLabel(r, c);
                      return (
                        <div
                          key={`${r}-${c}`}
                          className="relative flex aspect-square items-center justify-center border text-center"
                          style={{
                            backgroundColor: bg,
                            borderColor: boundary ? "#18181b" : "rgba(0,0,0,0.08)",
                            color: fg,
                          }}
                        >
                          <span className="font-mono text-sm font-semibold">
                            {value.toFixed(0)}
                          </span>
                          {label && (
                            <span className="absolute left-1 top-1 font-mono text-[7px] opacity-50">
                              {label}
                            </span>
                          )}
                        </div>
                      );
                    })
                  )}
                </div>

                <p className="mt-3 font-mono text-[10px] text-zinc-400">
                  Escala: {minTemp.toFixed(0)} °C → {maxTemp.toFixed(0)} °C (claro → escuro)
                </p>
              </div>
            </section>

            {/* Solver status */}
            <section className="rounded-2xl border border-zinc-200">
              <div className="flex flex-col gap-4 border-b border-zinc-200 px-6 py-5 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                    RESOLVEDOR / 05
                  </div>
                  <h2 className="mt-1 font-semibold">Gauss-Seidel — Iterações</h2>
                </div>
                <div className="flex items-center gap-2 font-mono text-xs">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      converged ? "bg-zinc-950" : "bg-zinc-300"
                    }`}
                  />
                  {converged ? "CONVERGIU" : iteration === 0 ? "PRONTO" : "ITERANDO"}
                </div>
              </div>

              <div className="grid grid-cols-2 divide-x divide-zinc-200 md:grid-cols-4">
                <Stat label="Iterações" value={String(iteration)} />
                <Stat label="Passos registrados" value={String(stepsTaken)} />
                <Stat label="Tolerância" value={tolerance.toExponential(2)} />
                <Stat label="Residual atual" value={residual.toExponential(4)} />
              </div>
            </section>

            {/* Iterations table */}
            <section className="overflow-hidden rounded-2xl border border-zinc-200">
              <div className="flex items-center justify-between border-b border-zinc-200 px-6 py-5">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                    REGISTRO DE ITERAÇÕES / 06
                  </div>
                  <h2 className="mt-1 font-semibold">Convergência numérica</h2>
                </div>
                <span className="font-mono text-xs text-zinc-400">
                  {stepsTaken} passo(s)
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] border-collapse font-mono text-xs">
                  <thead>
                    <tr className="border-b border-zinc-200 bg-zinc-50 text-left text-[10px] uppercase tracking-wider text-zinc-400">
                      <th className="px-5 py-4">#</th>
                      <th className="px-5 py-4">P1</th>
                      <th className="px-5 py-4">P2</th>
                      <th className="px-5 py-4">P3</th>
                      <th className="px-5 py-4">P4</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayHistory
                      .slice(-15)
                      .map((item) => (
                        <tr
                          key={item.iteration}
                          className="border-b border-zinc-100 transition hover:bg-zinc-50"
                        >
                          <td className="px-5 py-3 text-zinc-400">
                            {String(item.iteration).padStart(3, "0")}
                          </td>
                          <td className="px-5 py-3">{item.P1.toFixed(6)}</td>
                          <td className="px-5 py-3">{item.P2.toFixed(6)}</td>
                          <td className="px-5 py-3">{item.P3.toFixed(6)}</td>
                          <td className="px-5 py-3 font-semibold">{item.P4.toFixed(6)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {displayHistory.length > 15 && (
                <div className="border-t border-zinc-200 px-6 py-3 font-mono text-[10px] text-zinc-400">
                  Mostrando as últimas 15 linhas de {displayHistory.length}.
                </div>
              )}
            </section>

            {/* Chart */}
            <section className="overflow-hidden rounded-2xl border border-zinc-200">
              <div className="border-b border-zinc-200 px-6 py-5">
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                  GRÁFICO / 07
                </div>
                <h2 className="mt-1 font-semibold">Evolução das temperaturas</h2>
              </div>
              <div className="h-80 w-full p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={displayHistory} margin={{ top: 10, right: 18, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(150,150,150,0.25)" />
                    <XAxis
                      dataKey="iteration"
                      type="number"
                      allowDecimals={false}
                      label={{ value: "iteração", position: "insideBottom", offset: -2, fontSize: 11 }}
                      tick={{ fontSize: 11 }}
                      stroke="#71717a"
                    />
                    <YAxis
                      label={{ value: "°C", angle: -90, position: "insideLeft", fontSize: 11 }}
                      tick={{ fontSize: 11 }}
                      stroke="#71717a"
                    />
                    <Tooltip
                      formatter={(value, name) => [`${Number(value).toFixed(4)} °C`, String(name)]}
                      labelFormatter={(label) => `iteração ${label}`}
                      contentStyle={{
                        borderRadius: 10,
                        border: "1px solid #e4e4e7",
                        boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                        fontSize: 12,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    {(["P1", "P2", "P3", "P4"] as const).map((key, i) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={CHART_COLORS[i]}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        </div>

        {/* About */}
        <section className="mt-16">
          <div className="mb-8">
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
              SOBRE / 08
            </div>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight">
              Sobre a aplicação
            </h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <Card
              title="Equação de Laplace"
              description="Resolve ∇²T = 0 no regime estacionário com as condições de contorno informadas nas quatro bordas."
            />
            <Card
              title="Gauss-Seidel"
              description="Atualiza cada nó interno P1–P4 como média sequencial dos quatro vizinhos mais recentes."
            />
            <Card
              title="Contorno editável"
              description="Altera Superior, Direita, Inferior e Esquerda em tempo real, reconstruindo a malha imediatamente."
            />
            <Card
              title="Convergência"
              description="Para quando a maior variação entre iterações fica abaixo da tolerância ε definida."
            />
            <Card
              title="Mapa de calor"
              description="Malha 4×4 com intensidade de temperatura em escala computacional (claro → escuro)."
            />
            <Card
              title="Histórico numérico"
              description="Registro completo das iterações com os valores de P1, P2, P3 e P4 em cada passo."
            />
            <Card
              title="Evolução gráfica"
              description="Gráfico de linhas mostrando a trajetória dos quatro nós internos até a convergência."
            />
            <Card
              title="100% client-side"
              description="Toda a computação acontece no navegador, sem API ou dependências externas."
            />
            <Card
              title="Estética computacional"
              description="Visual em preto e branco, tipografia monoespaçada, bordas finas e nomenclatura de solver."
            />
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-10 flex flex-col justify-between gap-3 border-t border-zinc-200 pt-6 font-mono text-[10px] uppercase tracking-wider text-zinc-400 md:flex-row">
          <span>Interface de computação numérica</span>
          <span>TypeScript / React / Next.js / Tailwind</span>
        </footer>
      </div>
    </main>
  );
}

/* -------------------------------------------------------------------------- */
/* Components                                                                 */
/* -------------------------------------------------------------------------- */

type NumberInputProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  suffix?: string;
};

function NumberInput({ label, value, onChange, suffix }: NumberInputProps) {
  return (
    <label className="block">
      <span className="mb-2 block font-mono text-[10px] uppercase tracking-wider text-zinc-400">
        {label}
      </span>
      <div className="flex items-center overflow-hidden rounded-xl border border-zinc-200 bg-white transition focus-within:border-zinc-950 focus-within:ring-1 focus-within:ring-zinc-950">
        <input
          type="number"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="w-full bg-transparent px-3 py-3 font-mono text-sm outline-none"
        />
        {suffix && (
          <span className="border-l border-zinc-200 px-3 py-3 font-mono text-xs text-zinc-400">
            {suffix}
          </span>
        )}
      </div>
    </label>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="min-w-0 px-5 py-5">
      <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">{label}</div>
      <div className="mt-2 truncate font-mono text-sm font-medium">{value}</div>
    </div>
  );
}

function SummaryStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white p-3 text-center">
      <div className="font-mono text-[8px] uppercase tracking-wider text-zinc-400">{label}</div>
      <div className="mt-1 font-mono text-sm font-semibold">{value}</div>
    </div>
  );
}

function Card({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-2xl border border-zinc-200 bg-zinc-50/60 p-6">
      <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">{title}</div>
      <p className="mt-3 text-sm leading-5 text-zinc-500">{description}</p>
    </div>
  );
}
