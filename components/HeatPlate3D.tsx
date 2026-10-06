"use client";

import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { FIELD_RES, trilinear } from "@/lib/laplaceField";

/* Domínio discreto: campo entregue via prop (RES×RES×RES), com o sólido
   (fisicamente) ocupando o intervalo [0, RES-1]³ no espaço da malha. Na cena,
   cada eixo é REDIMENSIONADO para formar um retângulo (cuboide, estilo sala de
   aula: mais comprido que alto) e centralizado em (0,0,0). */
const RES = FIELD_RES;
const EXTENT = RES - 1;

/* Dimensões do retângulo no mundo (u.m.). */
const DIMS = { x: 24, y: 12, z: 17 };
const SCALE = { x: DIMS.x / EXTENT, y: DIMS.y / EXTENT, z: DIMS.z / EXTENT };
const CENTER = { x: DIMS.x / 2, y: DIMS.y / 2, z: DIMS.z / 2 };

/* Mapeia coordenadas de malha (0..EXTENT)³ → mundo centrado (0,0,0). */
function toWorld(x: number, y: number, z: number): [number, number, number] {
  return [x * SCALE.x - CENTER.x, y * SCALE.y - CENTER.y, z * SCALE.z - CENTER.z];
}

const ISO_LEVELS = 6;
const FLUX_MAX_STEPS = 160;
const FLUX_STEP = 0.4;
const FLUX_SEEDS_PER_FACE = 6;

/* Escala térmica frio→quente em RGB [0..255]: azul → ciano → verde →
   amarelo → vermelho. */
function coldHotRGB(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * 4;
  const seg = Math.min(3, Math.floor(x));
  const f = x - seg;
  const stops: [number, number, number][] = [
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
  ];
  const a = stops[seg];
  const b = stops[seg + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

function coldHotCSS(t: number): string {
  const [r, g, b] = coldHotRGB(t);
  return `rgb(${r}, ${g}, ${b})`;
}

/* -------------------------------------------------------------------------- */
/* Superfícies isotérmicas — surface nets (simples, sem tabelas de marching)   */
/* -------------------------------------------------------------------------- */

interface IsoMeshData {
  level: number;
  t: number;
  geometry: THREE.BufferGeometry;
}

function isoSurface(field: Float32Array, n: number, iso: number) {
  const m = n - 1;
  const total = m * m * m;
  const has = new Uint8Array(total);
  const vpos = new Float32Array(total * 3);
  const cid = (i: number, j: number, k: number) => i + j * m + k * m * m;
  const fIdx = (i: number, j: number, k: number) => i + j * n + k * n * n;

  /* Passo 1 — vértice dual de cada célula que cruza o nível (média dos pontos
     de intersecção nas 12 arestas). */
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      for (let k = 0; k < m; k++) {
        const corners: [number, number, number][] = [
          [i, j, k],
          [i + 1, j, k],
          [i, j + 1, k],
          [i + 1, j + 1, k],
          [i, j, k + 1],
          [i + 1, j, k + 1],
          [i, j + 1, k + 1],
          [i + 1, j + 1, k + 1],
        ];
        const vals = corners.map(([x, y, z]) => field[fIdx(x, y, z)]);
        const inside = vals.map((v) => (v < iso ? 1 : 0));
        const sum =
          inside[0] + inside[1] + inside[2] + inside[3] + inside[4] + inside[5] + inside[6] + inside[7];
        if (sum === 0 || sum === 8) continue;

        const edges: [number, number][] = [
          [0, 1], [0, 2], [1, 3], [2, 3],
          [4, 5], [4, 6], [5, 7], [6, 7],
          [0, 4], [1, 5], [2, 6], [3, 7],
        ];
        let ax = 0, ay = 0, az = 0, cnt = 0;
        for (const [a, b] of edges) {
          if (inside[a] === inside[b]) continue;
          const denom = vals[b] - vals[a];
          const tt = denom === 0 ? 0.5 : (iso - vals[a]) / denom;
          ax += corners[a][0] + tt * (corners[b][0] - corners[a][0]);
          ay += corners[a][1] + tt * (corners[b][1] - corners[a][1]);
          az += corners[a][2] + tt * (corners[b][2] - corners[a][2]);
          cnt++;
        }
        if (cnt === 0) continue;
        const c = cid(i, j, k);
        has[c] = 1;
        vpos[c * 3] = ax / cnt;
        vpos[c * 3 + 1] = ay / cnt;
        vpos[c * 3 + 2] = az / cnt;
      }
    }
  }

  /* Passo 2 — quads duais (2 triângulos cada) nas três famílias de planos. */
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const getV = (i: number, j: number, k: number): [number, number, number] | null => {
    if (i < 0 || j < 0 || k < 0 || i >= m || j >= m || k >= m) return null;
    const c = cid(i, j, k);
    if (!has[c]) return null;
    return toWorld(vpos[c * 3], vpos[c * 3 + 1], vpos[c * 3 + 2]);
  };

  const emitQuad = (a: [number, number, number], b: [number, number, number], c: [number, number, number], d: [number, number, number]) => {
    const va = getV(a[0], a[1], a[2]);
    const vb = getV(b[0], b[1], b[2]);
    const vc = getV(c[0], c[1], c[2]);
    const vd = getV(d[0], d[1], d[2]);
    if (!va || !vb || !vc || !vd) return;

    const base = positions.length / 3;
    for (const p of [va, vb, vc, vd]) positions.push(p[0], p[1], p[2]);
    normals.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    const addNormal = (i0: number, j0: number, k0: number) => {
      const ax = positions[i0 * 3], ay = positions[i0 * 3 + 1], az = positions[i0 * 3 + 2];
      const bx = positions[j0 * 3], by = positions[j0 * 3 + 1], bz = positions[j0 * 3 + 2];
      const dx = positions[k0 * 3], dy = positions[k0 * 3 + 1], dz = positions[k0 * 3 + 2];
      const ux = bx - ax, uy = by - ay, uz = bz - az;
      const vx = dx - ax, vy = dy - ay, vz = dz - az;
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1;
      for (const iv of [i0, j0, k0]) {
        normals[iv * 3] += nx / len;
        normals[iv * 3 + 1] += ny / len;
        normals[iv * 3 + 2] += nz / len;
      }
      indices.push(i0, j0, k0);
    };

    addNormal(base, base + 1, base + 2);
    addNormal(base, base + 2, base + 3);
  };

  for (let i = 0; i < m - 1; i++) {
    for (let j = 0; j < m - 1; j++) {
      for (let k = 0; k < m; k++) {
        emitQuad([i, j, k], [i + 1, j, k], [i + 1, j + 1, k], [i, j + 1, k]);
      }
    }
  }
  for (let i = 0; i < m - 1; i++) {
    for (let k = 0; k < m - 1; k++) {
      for (let j = 0; j < m; j++) {
        emitQuad([i, j, k], [i + 1, j, k], [i + 1, j, k + 1], [i, j, k + 1]);
      }
    }
  }
  for (let j = 0; j < m - 1; j++) {
    for (let k = 0; k < m - 1; k++) {
      for (let i = 0; i < m; i++) {
        emitQuad([i, j, k], [i, j + 1, k], [i, j + 1, k + 1], [i, j, k + 1]);
      }
    }
  }

  return { positions, normals, indices };
}

function buildIsoMeshes(field: Float32Array, n: number, minV: number, maxV: number): IsoMeshData[] {
  const out: IsoMeshData[] = [];
  for (let s = 1; s <= ISO_LEVELS; s++) {
    const level = minV + ((maxV - minV) * s) / (ISO_LEVELS + 1);
    const raw = isoSurface(field, n, level);
    if (raw.indices.length === 0) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(raw.positions, 3));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(raw.normals, 3));
    geometry.setIndex(raw.indices);
    out.push({ level, t: (level - minV) / (maxV - minV), geometry });
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Linhas de fluxo de calor (seguem q = −k∇T, do quente para o frio)           */
/* -------------------------------------------------------------------------- */

function traceFluxLines(
  field: Float32Array,
  n: number,
  minV: number,
  maxV: number
): { positions: Float32Array; colors: Float32Array } {
  const directionAt = (x: number, y: number, z: number): [number, number, number] | null => {
    const d = 0.5;
    const gx = (trilinear(field, n, x + d, y, z) - trilinear(field, n, x - d, y, z)) / (2 * d);
    const gy = (trilinear(field, n, x, y + d, z) - trilinear(field, n, x, y - d, z)) / (2 * d);
    const gz = (trilinear(field, n, x, y, z + d) - trilinear(field, n, x, y, z - d)) / (2 * d);
    const mag = Math.hypot(gx, gy, gz);
    if (mag < 1e-6) return null;
    return [-gx / mag, -gy / mag, -gz / mag];
  };

  const dirAt = (x: number, y: number, z: number): [number, number, number] =>
    directionAt(x, y, z) ?? [0, 0, 0];

  const inside = (x: number, y: number, z: number) =>
    x >= 0 && x <= n - 1 && y >= 0 && y <= n - 1 && z >= 0 && z <= n - 1;

  /* Sementes distribuídas nas faces quentes (topo, sul e leste). */
  const spread = (count: number) => {
    const out: number[] = [];
    for (let s = 0; s < count; s++) out.push(1 + ((EXTENT - 2) * s) / Math.max(count - 1, 1));
    return out;
  };
  const xs = spread(FLUX_SEEDS_PER_FACE);
  const ys = spread(FLUX_SEEDS_PER_FACE);
  const zs = spread(FLUX_SEEDS_PER_FACE);
  const inset = 0.35;
  const seeds: [number, number, number][] = [];
  for (const x of xs) for (const z of zs) seeds.push([x, EXTENT - inset, z]);
  for (const x of xs) for (const y of ys) seeds.push([x, y, EXTENT - inset]);
  for (const y of ys) for (const z of zs) seeds.push([EXTENT - inset, y, z]);

  const segs: number[] = [];
  const cols: number[] = [];
  const h = FLUX_STEP;

  for (const seed of seeds) {
    let x = seed[0], y = seed[1], z = seed[2];
    for (let s = 0; s < FLUX_MAX_STEPS; s++) {
      const k1 = directionAt(x, y, z);
      if (!k1) break;
      const k2 = dirAt(x + (h * k1[0]) / 2, y + (h * k1[1]) / 2, z + (h * k1[2]) / 2);
      const k3 = dirAt(x + (h * k2[0]) / 2, y + (h * k2[1]) / 2, z + (h * k2[2]) / 2);
      const k4 = dirAt(x + h * k3[0], y + h * k3[1], z + h * k3[2]);
      const dx = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
      const dy = (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
      const dz = (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6;
      const mag = Math.hypot(dx, dy, dz);
      if (mag < 1e-6) break;
      const nx = x + (dx * h) / mag;
      const ny = y + (dy * h) / mag;
      const nz = z + (dz * h) / mag;
      if (!inside(nx, ny, nz)) break;

      const t0 = trilinear(field, n, x, y, z);
      const t1 = trilinear(field, n, nx, ny, nz);
      const [r0, g0, bl0] = coldHotRGB((t0 - minV) / (maxV - minV));
      const [r1, g1, bl1] = coldHotRGB((t1 - minV) / (maxV - minV));
      const [wx0, wy0, wz0] = toWorld(x, y, z);
      const [wx1, wy1, wz1] = toWorld(nx, ny, nz);
      segs.push(wx0, wy0, wz0, wx1, wy1, wz1);
      cols.push(
        r0 / 255, g0 / 255, bl0 / 255,
        r1 / 255, g1 / 255, bl1 / 255
      );
      x = nx; y = ny; z = nz;
    }
  }
  return { positions: new Float32Array(segs), colors: new Float32Array(cols) };
}

/* -------------------------------------------------------------------------- */
/* Cena Three.js                                                               */
/* -------------------------------------------------------------------------- */

type Mode = "iso" | "flux" | "both";

const MODES: { id: Mode; label: string }[] = [
  { id: "iso", label: "Isotermas" },
  { id: "flux", label: "Linhas de fluxo" },
  { id: "both", label: "Ambos" },
];

function RoomEdges() {
  const e = EXTENT;
  const corners: [number, number, number][] = [
    toWorld(0, 0, 0),
    toWorld(e, 0, 0),
    toWorld(e, e, 0),
    toWorld(0, e, 0),
    toWorld(0, 0, e),
    toWorld(e, 0, e),
    toWorld(e, e, e),
    toWorld(0, e, e),
  ];
  const pairs: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  return (
    <group>
      {pairs.map(([a, b], idx) => (
        <Line key={idx} points={[corners[a], corners[b]]} color="#a1a1aa" lineWidth={1} transparent opacity={0.55} />
      ))}
    </group>
  );
}

function HeatPlate3DScene({
  isoMeshes,
  flux,
  mode,
}: {
  isoMeshes: IsoMeshData[];
  flux: { positions: Float32Array; colors: Float32Array };
  mode: Mode;
}) {
  const fluxGeometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(flux.positions, 3));
    geom.setAttribute("color", new THREE.Float32BufferAttribute(flux.colors, 3));
    return geom;
  }, [flux]);

  /* Material explícito: vertexColors + sem tone mapping garantem que a cor
     azul→vermelho por vértice apareça viva (sem ser lavada pelas superfícies). */
  const fluxMaterial = useMemo(
    () =>
      new THREE.LineBasicMaterial({
        vertexColors: true,
        toneMapped: false,
        transparent: true,
        opacity: 0.95,
      }),
    []
  );

  return (
    <>
      <ambientLight intensity={0.9} />
      <directionalLight position={[30, 40, 25]} intensity={1.3} />
      <directionalLight position={[-30, -20, -25]} intensity={0.35} />

      {(mode === "iso" || mode === "both") &&
        isoMeshes.map((m) => (
          <mesh key={m.level} geometry={m.geometry}>
            <meshStandardMaterial
              color={coldHotCSS(m.t)}
              transparent
              opacity={0.38}
              side={THREE.DoubleSide}
              roughness={0.65}
              metalness={0.05}
              depthWrite={false}
            />
          </mesh>
        ))}

      {(mode === "flux" || mode === "both") && (
        <lineSegments geometry={fluxGeometry} material={fluxMaterial} renderOrder={5} />
      )}

      <RoomEdges />

      <Grid
        position={[0, -CENTER.y - 8, 0]}
        args={[26, 26]}
        cellSize={1}
        cellColor="#e9e9ec"
        sectionSize={4}
        sectionColor="#d4d4d8"
        fadeDistance={55}
        fadeStrength={3}
      />
      <OrbitControls
        makeDefault
        enablePan={false}
        minDistance={6}
        maxDistance={80}
        target={[0, 0, 0]}
      />
    </>
  );
}

export default function HeatPlate3D({
  field,
  minV,
  maxV,
}: {
  field: Float32Array;
  minV: number;
  maxV: number;
}) {
  const [mode, setMode] = useState<Mode>("both");

  const isoMeshes = useMemo(() => buildIsoMeshes(field, RES, minV, maxV), [field, minV, maxV]);
  const flux = useMemo(() => traceFluxLines(field, RES, minV, maxV), [field, minV, maxV]);

  return (
    <div>
       <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
         <div className="flex gap-1 rounded-xl border border-zinc-200 bg-white p-1">
           {MODES.map(({ id, label }) => (
             <button
               key={id}
               onClick={() => setMode(id)}
               className={`rounded-lg px-3 py-1.5 font-mono text-[10px] font-medium uppercase tracking-wider transition ${
                 mode === id ? "bg-zinc-950 text-white" : "text-zinc-500 hover:text-zinc-950"
               }`}
             >
               {label}
             </button>
           ))}
         </div>
         <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-400 text-right">
           campo na iteração atual · malha {RES}³
         </span>
       </div>

       <div className="relative mt-3 h-[420px] w-full overflow-hidden rounded-xl border border-zinc-200 bg-white md:h-[520px]">
         <Canvas
           camera={{ position: [40, 24, 44], fov: 40 }}
           dpr={[1, 2]}
           gl={{ antialias: true, alpha: true }}
         >
           <HeatPlate3DScene isoMeshes={isoMeshes} flux={flux} mode={mode} />
         </Canvas>
       </div>

      <div className="mt-3 flex items-center gap-3 font-mono text-[9px] uppercase tracking-wider text-zinc-400">
        <span>frio</span>
        <div
          className="h-1.5 flex-1 rounded"
          style={{
            background:
              "linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)",
          }}
        />
        <span>quente</span>
      </div>
      <p className="mt-3 font-mono text-[10px] leading-4 text-zinc-400">
        Escala: {minV.toFixed(0)} °C → {maxV.toFixed(0)} °C — azul = frio →
        vermelho = quente. A cor de cada superfície e de cada trecho de linha é
        o valor de T naquele ponto (campo na iteração atual). Linhas seguem o
        fluxo q = −k∇T, do quente para o frio. Arraste para orbitar, role para
        aproximar.
      </p>
    </div>
  );
}