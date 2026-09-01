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
import HeatPlate3D from "./HeatPlate3D";
import {
  boundaryExtrema,
  makeInitialField,
  sweepField,
} from "@/lib/laplaceField";

type BoundaryKey = "top" | "bottom" | "north" | "south" | "east" | "west";
type GridValue = number[][][];

const GRID_SIZE = 4;

const INTERIOR_KEYS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"] as const;
type PointKey = (typeof INTERIOR_KEYS)[number];

const NODE_COORDS: Record<PointKey, [number, number, number]> = {
  P1: [1, 1, 1],
  P2: [1, 1, 2],
  P3: [1, 2, 1],
  P4: [1, 2, 2],
  P5: [2, 1, 1],
  P6: [2, 1, 2],
  P7: [2, 2, 1],
  P8: [2, 2, 2],
};

interface PointHistory {
  iteration: number;
  P1: number;
  P2: number;
  P3: number;
  P4: number;
  P5: number;
  P6: number;
  P7: number;
  P8: number;
}

const DEFAULT_BOUNDARIES: Record<BoundaryKey, number> = {
  top: 80,
  east: 50,
  bottom: 40,
  west: 10,
  north: 25,
  south: 65,
};

const CHART_COLORS = [
  "#000000",
  "#3f3f46",
  "#52525b",
  "#71717a",
  "#a1a1aa",
  "#b4b4bc",
  "#d4d4d4",
  "#e4e4e7",
];

function makeInitialGrid(boundaries: Record<BoundaryKey, number>): GridValue {
  const grid: GridValue = Array.from({ length: GRID_SIZE }, () =>
    Array.from({ length: GRID_SIZE }, () =>
      Array.from({ length: GRID_SIZE }, () => 0)
    )
  );
  for (let i = 0; i < GRID_SIZE; i++) {
    for (let j = 0; j < GRID_SIZE; j++) {
      for (let k = 0; k < GRID_SIZE; k++) {
        if (i === 0) grid[i][j][k] = boundaries.west;
        if (i === GRID_SIZE - 1) grid[i][j][k] = boundaries.east;
        if (j === 0) grid[i][j][k] = boundaries.bottom;
        if (j === GRID_SIZE - 1) grid[i][j][k] = boundaries.top;
        if (k === 0) grid[i][j][k] = boundaries.north;
        if (k === GRID_SIZE - 1) grid[i][j][k] = boundaries.south;
      }
    }
  }
  return grid;
}

/* Frio → claro, Quente → escuro (paleta B&W computacional). */

function gridSnapshot(g: GridValue, iteration: number): PointHistory {
  const item = { iteration, P1: 0, P2: 0, P3: 0, P4: 0, P5: 0, P6: 0, P7: 0, P8: 0 };
  for (const key of INTERIOR_KEYS) {
    const [i, j, k] = NODE_COORDS[key];
    item[key] = g[i][j][k];
  }
  return item;
}

function solveOneStep(g: GridValue): number {
  let maxDiff = 0;
  for (let i = 1; i < GRID_SIZE - 1; i++) {
    for (let j = 1; j < GRID_SIZE - 1; j++) {
      for (let k = 1; k < GRID_SIZE - 1; k++) {
        const newVal =
          (g[i - 1][j][k] +
            g[i + 1][j][k] +
            g[i][j - 1][k] +
            g[i][j + 1][k] +
            g[i][j][k - 1] +
            g[i][j][k + 1]) /
          6;
        maxDiff = Math.max(maxDiff, Math.abs(newVal - g[i][j][k]));
        g[i][j][k] = newVal;
      }
    }
  }
  return maxDiff;
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
  const [history, setHistory] = useState<PointHistory[]>([gridSnapshot(
    makeInitialGrid(DEFAULT_BOUNDARIES),
    0
  )]);
  const [residual, setResidual] = useState(0);
  /* Campo denso (RES³) para a visualização 3D, avançado junto com o solver. */
  const [field, setField] = useState<Float32Array>(() =>
    makeInitialField(DEFAULT_BOUNDARIES)
  );
  const { minV, maxV } = useMemo(() => boundaryExtrema(boundaries), [boundaries]);

  const stepsTaken = Math.max(history.length - 1, 0);
  const converged = residual < tolerance && iteration > 0;

  function refreshFromBoundaries(next: Record<BoundaryKey, number>) {
    const g = makeInitialGrid(next);
    setGrid(g);
    setIteration(0);
    setResidual(0);
    setHistory([gridSnapshot(g, 0)]);
    setField(makeInitialField(next));
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
    const g = grid.map((slice) => slice.map((row) => [...row]));
    const maxDiff = solveOneStep(g);
    const newIteration = iteration + 1;
    const f = field.slice();
    sweepField(f);
    setGrid(g);
    setField(f);
    setIteration(newIteration);
    setResidual(maxDiff);
    setHistory((prev) => [...prev, gridSnapshot(g, newIteration)]);
  }

  function runToConvergence() {
    const g = grid.map((slice) => slice.map((row) => [...row]));
    const f = field.slice();
    let localIteration = iteration;
    let localResidual = Number.POSITIVE_INFINITY;
    let localDenseResidual = Number.POSITIVE_INFINITY;
    const newHistory: PointHistory[] = [];
    const guard = 100000;
    let cycle = 0;
    while ((localResidual > tolerance || localDenseResidual > tolerance) && cycle < guard) {
      cycle++;
      localIteration++;
      const maxDiff = solveOneStep(g);
      localResidual = maxDiff;
      localDenseResidual = sweepField(f);
      newHistory.push(gridSnapshot(g, localIteration));
    }
    setGrid(g);
    setField(f);
    setIteration(localIteration);
    setResidual(localResidual);
    if (newHistory.length > 0) setHistory((prev) => [...prev, ...newHistory]);
  }

  const displayHistory = useMemo(() => reduceToDisplay(history, 200), [history]);

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
                Física Numérica / Resolvedor de Laplace 3D
              </span>
            </div>

            <div className="flex flex-wrap items-end justify-between gap-6">
              <div>
                <h1 className="text-4xl font-semibold tracking-[-0.04em] md:text-5xl">
                  Transferência de Calor 3D
                </h1>
                <p className="mt-3 max-w-2xl font-mono text-sm leading-6 text-zinc-500">
                  Resolução da Equação de Laplace (∇²T = 0) em um domínio
                  tridimensional por diferenças finitas via método de Gauss-Seidel,
                  com atualização sequencial dos nós internos P1–P8.
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
                <p className="mt-1 font-mono text-[10px] leading-4 text-zinc-400">
                  Seis faces do sólido: superior, inferior, frente, fundo, esquerda
                  e direita.
                </p>
              </div>

              <div className="space-y-5">
                <NumberInput
                  label="Superior (topo)"
                  value={boundaries.top}
                  onChange={(v) => handleBoundaryChange("top", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Inferior (base)"
                  value={boundaries.bottom}
                  onChange={(v) => handleBoundaryChange("bottom", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Frente (norte)"
                  value={boundaries.north}
                  onChange={(v) => handleBoundaryChange("north", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Fundo (sul)"
                  value={boundaries.south}
                  onChange={(v) => handleBoundaryChange("south", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Esquerda (oeste)"
                  value={boundaries.west}
                  onChange={(v) => handleBoundaryChange("west", String(v))}
                  suffix="°C"
                />
                <NumberInput
                  label="Direita (leste)"
                  value={boundaries.east}
                  onChange={(v) => handleBoundaryChange("east", String(v))}
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
                  <span>P_O</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_L</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_SUP</span>
                  <span className="text-zinc-600">+</span>
                  <span>P_INF</span>
                  <span className="text-zinc-400">)</span>
                  <span className="text-zinc-600">/</span>
                  <span>6</span>
                </div>

                <p className="mt-3 max-w-xl font-mono text-sm leading-6 text-zinc-400">
                  Cada nó interno é a média aritmética dos seus seis vizinhos
                  imediatos (N, S, O, L, superior e inferior), atualizada
                  sequencialmente (Gauss-Seidel).
                </p>

                <div className="mt-8 flex flex-wrap items-center gap-3 font-mono text-xs">
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    ∇²T = 0
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    malha {GRID_SIZE}×{GRID_SIZE}×{GRID_SIZE}
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    ε = {tolerance}
                  </span>
                  <span className="rounded-md border border-zinc-700 px-3 py-1.5 text-zinc-400">
                    nós internos = 8
                  </span>
                </div>
              </div>
            </section>

            {/* Result + Plate */}
            <section className="grid gap-6 md:grid-cols-[300px_1fr]">
              <div className="relative overflow-hidden rounded-2xl bg-zinc-950 p-6 text-white">
                <div className="absolute right-5 top-5 font-mono text-[9px] uppercase tracking-[0.2em] text-zinc-600">
                  SISTEMA LINEAR
                </div>
                <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-500">
                  VALORES CONVERGIDOS
                </div>
                <div className="mt-4 space-y-2.5 font-mono">
                  {INTERIOR_KEYS.map((p) => {
                    const [i, j, k] = NODE_COORDS[p];
                    return (
                      <div key={p} className="flex items-center justify-between">
                        <span className="text-zinc-400">{p}</span>
                        <span className="text-lg font-semibold tracking-tight">
                          {grid[i][j][k].toFixed(2)} °C
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-50 p-6">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-400">
                      MALHA / 04
                    </div>
                    <p className="mt-1 font-semibold">Isotermas & linhas de fluxo (3D)</p>
                  </div>
                  <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">
                    malha {GRID_SIZE}×{GRID_SIZE}×{GRID_SIZE}
                  </span>
                </div>

                <div className="relative mt-3 h-[480px] w-full overflow-hidden rounded-xl border border-zinc-200 bg-white">
                  <HeatPlate3D field={field} minV={minV} maxV={maxV} />
                </div>
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
                <table className="w-full min-w-[900px] border-collapse font-mono text-xs">
                  <thead>
                    <tr className="border-b border-zinc-200 bg-zinc-50 text-left text-[10px] uppercase tracking-wider text-zinc-400">
                      <th className="px-5 py-4">#</th>
                      {INTERIOR_KEYS.map((key) => (
                        <th key={key} className="px-5 py-4">
                          {key}
                        </th>
                      ))}
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
                          {INTERIOR_KEYS.map((key, idx) => (
                            <td
                              key={key}
                              className={`px-5 py-3 ${
                                idx === INTERIOR_KEYS.length - 1 ? "font-semibold" : ""
                              }`}
                            >
                              {item[key].toFixed(6)}
                            </td>
                          ))}
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
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    {INTERIOR_KEYS.map((key, i) => (
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
              title="Equação de Laplace 3D"
              description="Resolve ∇²T = 0 no regime estacionário com condições de contorno em seis faces do sólido tridimensional."
            />
            <Card
              title="Gauss-Seidel"
              description="Atualiza cada nó interno P1–P8 como média sequencial dos seis vizinhos mais recentes (estêncil 6 pontos)."
            />
            <Card
              title="Contorno em seis faces"
              description="Altera Superior, Inferior, Frente, Fundo, Esquerda e Direita em tempo real, reconstruindo a malha 4×4×4 imediatamente."
            />
            <Card
              title="Convergência"
              description="Para quando a maior variação entre iterações fica abaixo da tolerância ε definida."
            />
            <Card
              title="Sala de aula 3D"
              description="Visualização com Three.js: casca translúcida com as faces de contorno e núcleo sólido com os nós internos, orbitável."
            />
            <Card
              title="Histórico numérico"
              description="Registro completo das iterações com os valores de P1 a P8 em cada passo."
            />
            <Card
              title="Evolução gráfica"
              description="Gráfico de linhas mostrando a trajetória dos oito nós internos até a convergência."
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
          <span>TypeScript / React / Three.js / Next.js / Tailwind</span>
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