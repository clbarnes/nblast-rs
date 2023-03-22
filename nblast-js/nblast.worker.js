import { expose } from 'comlink';

const wasm = import('./pkg/nblast_wasm_bg.wasm');

/**
 * Calculate tangents and alpha values for given points
 * (as an array of 3-length arrays of numbers),
 * and returned in flat typed array form as can be passed into the NblastArena.
 *
 * Tangents and alphas can be given, in which case they will be returned as
 * flat typed arrays,
 * but it's better to do this elsewhere to save copies between the main and worker threads.
 *
 * @param {number[][]} points
 * @param {number[][]} [tangents]
 * @param {number[]} [alphas]
 * @returns {Object.<string, Float64Array>} { points, tangents, alphas }, in flattened form which can be passed straight into wasm.
 */
export async function makeFlatPointsTangentsAlphas(points, tangents, alphas) {
  const pointsFlat = new Float64Array(points.flat());

  if (tangents != null) {
    if (alphas == null) {
      alphas = new Float64Array(points.length).fill(1);
    }
    tangents = new Float64Array(tangents.flat());
  } else {
    const lib = await wasm;
    const tangentsAlphas = lib.make_flat_tangents_alphas(pointsFlat);
    alphas = tangentsAlphas.slice(pointsFlat.length);
    for (let i = 0; i < alphas.length; i++) {
      tangentsAlphas.pop();
    }
    tangents = tangentsAlphas;
  }

  return {
    points: pointsFlat,
    alphas: alphas,
    tangents: tangents
  }
}

/**
 * Return a Float64Array, which may contain the contents of (flattened) arr,
 * or an array with a particular length and fill value.
 *
 * @param {(number[]|number[][]|Float64Array)} [arr]
 * @param {number} [lengthIfNull]
 * @param {number} [fillIfNull]
 * @returns {Float64Array}
 */
function flatArray64(arr, lengthIfNull, fillIfNull) {
  if (arr == null) {
    return new Float64Array(lengthIfNull).fill(fillIfNull);
  }
  if (arr instanceof Float64Array) {
    return arr;
  }
  if (Array.isArray(arr[0])) {
    return new Float64Array(arr.flat());
  } else {
    return new Float64Array(arr);
  }
}

/**
 * Class containing NBLASTable neurons and a score matrix.
 *
 * After instantiation, must call the async init() method.
 */
export class Nblaster {
  constructor(distThresholds, dotThresholds, cells, k) {
    this.arena = null;

    this.distThresholds = flatArray64(distThresholds),
    this.dotThresholds = flatArray64(dotThresholds),
    this.cells = flatArray64(cells),
    this.k = Math.round(k)
  }

  async init() {
    const lib = await wasm;
    this.arena = new lib.NblastArena(
      flatArray64(distThresholds),
      flatArray64(dotThresholds),
      flatArray64(cells),
      Math.round(k)
    )
    return this;
  }

  /**
   * Add a point cloud to the arena.
   * Points and tangents can be given as an array of 3-arrays of numbers, or a flattened row-major Float64Array (used internally).
   *
   * @param {*} points - coordinates of point cloud as array of 3-arrays of numbers, or a flattened row-major Float64Array.
   * @param {*} [tangents] - calculated from points if not given.
   * @param {*} [alphas] - calculated from points if not given.
   * @returns {number} - Index of this point cloud; used for queries later.
   */
  async addNeuron(points, tangents, alphas) {
    let pointsFlat = flatArray64(points);

    if (tangents == null) {
      return this.arena.add_points(pointsFlat);
    }
    let tangentsFlat = flatArray64(points);
    let alphasFlat = flatArray64(alphas, points.length, 1);

    return this.arena.add_points_tangents_alphas(
      pointsFlat,
      tangentsFlat,
      alphasFlat
    );
  }

  /**
   * NBLAST two point clouds against each other.
   *
   * @param {number} queryIdx - Query neuron index
   * @param {number} targetIdx - Target neuron index
   * @param {boolean} [normalize=false] - whether to normalise against self-hit score
   * @param {string|null} [symmetry=undefined] - whether and how to combine with the reverse score. Should be "arithmetic_mean", "geometric_mean", "harmonic_mean", "min", "max", or undefined (do not use reverse score).
   * @param {boolean} [useAlpha=false] - whether to scale dotprops by colinearity values.
   * @returns {number} - NBLAST score
   */
  async queryTarget(queryIdx, targetIdx, normalize, symmetry, useAlpha) {
    const sym = symmetry ? symmetry.toString() : undefined;
    return this.arena.query_target(
      Math.round(queryIdx),
      Math.round(targetIdx),
      !!normalize,
      sym,
      !!useAlpha
    );
  }

  /**
   * NBLAST multiple point clouds against each other (as a cartesian product).
   *
   * @param {number[]} queryIdxs - Query neuron indices
   * @param {number[]} targetIdxs - Target neuron indices
   * @param {boolean} [normalize=false] - whether to normalise against self-hit score
   * @param {string|null} [symmetry] - whether and how to combine with the reverse score. Should be "arithmetic_mean", "geometric_mean", "harmonic_mean", "min", "max", or undefined (do not use reverse score).
   * @param {boolean} [useAlpha=false] - whether to scale dotprops by colinearity values.
   * @param {number} [maxCentroidDist] - skip query if point clouds' centroids are further than this distance; if not given, run all queries.
   * @returns {Map<number, Map<number, number>>} - NBLAST scores as map of query index to map of target index to score.
   */
  async queriesTargets(
    queryIdxs,
    targetIdxs,
    normalize,
    symmetry,
    useAlpha,
    maxCentroidDist
  ) {
    const sym = symmetry ? symmetry.toString() : undefined;
    const mcd =
      Number(maxCentroidDist) > 0 ? Number(maxCentroidDist) : undefined;
    return this.arena.queries_targets(
      new BigUint64Array(queryIdxs),
      new BigUint64Array(targetIdxs),
      !!normalize,
      sym,
      !!useAlpha,
      mcd
    );
  }

  /**
   * NBLAST multiple point clouds against each other (as a cartesian product).
   *
   * @param {number[]} queryIdxs - Query neuron indices
   * @param {number[]} targetIdxs - Target neuron indices
   * @param {boolean} [normalize=false] - whether to normalise against self-hit score
   * @param {string|null} [symmetry] - whether and how to combine with the reverse score. Should be "arithmetic_mean", "geometric_mean", "harmonic_mean", "min", "max", or undefined (do not use reverse score).
   * @param {boolean} [useAlpha=false] - whether to scale dotprops by colinearity values.
   * @param {number} [maxCentroidDist] - skip query if point clouds' centroids are further than this distance; if not given, run all queries.
   * @returns {Map<number, Map<number, number>>} - NBLAST scores as map of query index to map of target index to score.
   */
  async allVsAll(normalize, symmetry, useAlpha, maxCentroidDist) {
    const sym = symmetry ? symmetry.toString() : undefined;
    const mcd =
      Number(maxCentroidDist) > 0 ? Number(maxCentroidDist) : undefined;
    return this.arena.all_v_all(!!normalize, sym, !!useAlpha, mcd);
  }
}

expose(Nblaster);


// function parsePoints(str) {
//   const points = str
//     .split("\n")
//     .slice(1)
//     .map((r) => r.split(",").slice(1).map(parseFloat));
//   return points;
// }

// function parseSmat(str) {
//   const rows = str
//     .split("\n")
//     .filter((r) => !!r.trim())
//     .map((r) => r.split(","));
//   const dotBoundaries = [0];
//   for (let idx = 2; idx < rows[0].length; idx += 2) {
//     dotBoundaries.push(parseFloat(rows[0][idx].slice(0, -2)));
//   }

//   const distBoundaries = [0];
//   const values = [];
//   for (const row of rows.slice(1)) {
//     if (!row.length) {
//       continue;
//     }
//     distBoundaries.push(parseFloat(row[1].slice(0, -2)));
//     row.slice(2).forEach((el) => values.push(parseFloat(el)));
//   }
//   return {
//     distThresholds: distBoundaries,
//     dotThresholds: dotBoundaries,
//     cells: values,
//   };
// }

// Comlink.expose(NblastArena);
