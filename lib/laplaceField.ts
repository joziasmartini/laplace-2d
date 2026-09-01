export type BoundaryKey = "top" | "bottom" | "north" | "south" | "east" | "west";
export type Boundaries = Record<BoundaryKey, number>;

export const FIELD_RES = 24;

export function fieldIndex(n: number, i: number, j: number, k: number): number {
  return i + j * n + k * n * n;
}

/* Campo inicial: seis faces com as condições de contorno, interior em 0
   (espelhando a malha pedagógica 4×4×4). */
export function makeInitialField(b: Boundaries): Float32Array {
  const n = FIELD_RES;
  const T = new Float32Array(n * n * n);
  const idx = (i: number, j: number, k: number) => fieldIndex(n, i, j, k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        if (i === 0) T[idx(i, j, k)] = b.west;
        else if (i === n - 1) T[idx(i, j, k)] = b.east;
        else if (j === 0) T[idx(i, j, k)] = b.bottom;
        else if (j === n - 1) T[idx(i, j, k)] = b.top;
        else if (k === 0) T[idx(i, j, k)] = b.north;
        else if (k === n - 1) T[idx(i, j, k)] = b.south;
      }
    }
  }
  return T;
}

/* Uma varredura Gauss-Seidel (ω = 1) sobre os nós internos. Retorna o maior
   resíduo (variação) registrado e muta `T` in place. O valor permanece limitado
   pelas condições de contorno (monotônico, sem overshoot). */
export function sweepField(T: Float32Array, n: number = FIELD_RES, omega: number = 1): number {
  const idx = (i: number, j: number, k: number) => fieldIndex(n, i, j, k);
  let residual = 0;
  for (let i = 1; i < n - 1; i++) {
    for (let j = 1; j < n - 1; j++) {
      for (let k = 1; k < n - 1; k++) {
        const p = idx(i, j, k);
        const avg =
          (T[idx(i - 1, j, k)] +
            T[idx(i + 1, j, k)] +
            T[idx(i, j - 1, k)] +
            T[idx(i, j + 1, k)] +
            T[idx(i, j, k - 1)] +
            T[idx(i, j, k + 1)]) /
          6;
        const corr = omega * (avg - T[p]);
        residual = Math.max(residual, Math.abs(corr));
        T[p] += corr;
      }
    }
  }
  return residual;
}

export function boundaryExtrema(b: Boundaries): { minV: number; maxV: number } {
  const values = [b.top, b.bottom, b.north, b.south, b.east, b.west];
  return { minV: Math.min(...values), maxV: Math.max(...values) };
}

export function trilinear(
  T: Float32Array,
  n: number,
  x: number,
  y: number,
  z: number
): number {
  const clamp0 = (v: number) => Math.max(0, Math.min(n - 1, v));
  const cx = clamp0(x);
  const cy = clamp0(y);
  const cz = clamp0(z);
  const u0 = Math.floor(cx);
  const v0 = Math.floor(cy);
  const w0 = Math.floor(cz);
  const u1 = Math.min(u0 + 1, n - 1);
  const v1 = Math.min(v0 + 1, n - 1);
  const w1 = Math.min(w0 + 1, n - 1);
  const fu = cx - u0;
  const fv = cy - v0;
  const fw = cz - w0;

  const i000 = T[u0 + v0 * n + w0 * n * n];
  const i100 = T[u1 + v0 * n + w0 * n * n];
  const i010 = T[u0 + v1 * n + w0 * n * n];
  const i110 = T[u1 + v1 * n + w0 * n * n];
  const i001 = T[u0 + v0 * n + w1 * n * n];
  const i101 = T[u1 + v0 * n + w1 * n * n];
  const i011 = T[u0 + v1 * n + w1 * n * n];
  const i111 = T[u1 + v1 * n + w1 * n * n];

  const a = i000 + fu * (i100 - i000);
  const b = i010 + fu * (i110 - i010);
  const c = i001 + fu * (i101 - i001);
  const d = i011 + fu * (i111 - i011);
  return (a + fv * (b - a)) + fw * ((c + fv * (d - c)) - (a + fv * (b - a)));
}