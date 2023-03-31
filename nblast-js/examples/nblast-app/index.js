import init, { NblastArena, makeFlatTangentsAlphas } from "./node_modules/nblast-js/nblast_js.js";

const CACHE = {};

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
function makeFlatPointsTangentsAlphas(points, tangents, alphas) {
  const pointsFlat = flatArray64(points);
  let tangentsFlat;
  let alphasFlat;

  if (tangents != null) {
    tangentsFlat = flatArray64(tangents);
    alphasFlat = flatArray64(alphas, points.length, 1);
  } else {
    const tangentsAlphas = makeFlatTangentsAlphas(pointsFlat);
    tangentsFlat = tangentsAlphas.slice(0, pointsFlat.length);
    alphasFlat = tangentsAlphas.slice(pointsFlat.length);
  }

  return {
    points: pointsFlat,
    tangents: tangentsFlat,
    alphas: alphasFlat
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
    this.arena = new NblastArena(
      flatArray64(distThresholds),
      flatArray64(dotThresholds),
      flatArray64(cells),
      Math.round(k)
    )
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
  addNeuron(points, tangents, alphas) {
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
  queryTarget(queryIdx, targetIdx, normalize, symmetry, useAlpha) {
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
  queriesTargets(
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
   * NBLAST all point clouds against each other (as a cartesian product).
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

function parsePoints(str) {
  const points = str
    .split("\n")
    .slice(1)
    .map((r) => r.split(",").slice(1).map(parseFloat));
  return points;
}

function parseSmat(str) {
  const rows = str
    .split("\n")
    .filter((r) => !!r.trim())
    .map((r) => r.split(","));
  const dotBoundaries = [0];
  for (let idx = 2; idx < rows[0].length; idx += 2) {
    dotBoundaries.push(parseFloat(rows[0][idx].slice(0, -2)));
  }

  const distBoundaries = [0];
  const values = [];
  for (const row of rows.slice(1)) {
    if (!row.length) {
      continue;
    }
    distBoundaries.push(parseFloat(row[1].slice(0, -2)));
    row.slice(2).forEach((el) => values.push(parseFloat(el)));
  }
  return {
    distThresholds: distBoundaries,
    dotThresholds: dotBoundaries,
    cells: values,
  };
}

function parseFile(file, strParser) {
  return new Promise(function (resolve, reject) {
    const reader = new FileReader();
    reader.onload = function (ev) {
      const result = strParser(ev.target.result);
      return resolve(result);
    };

    reader.onerror = reject;
    reader.onabort = reject;
    reader.readAsText(file);
  });
}

function tableFromCellMap(mom, rowNames, columnNames) {
  const table = document.createElement("table");
  const header = document.createElement("tr");
  header.appendChild(document.createElement("th"));
  for (let name of columnNames) {
    const cell = document.createElement("th");
    cell.innerText = name;
    header.appendChild(cell);
  }
  table.appendChild(header);

  for (const [ridx, rname] of rowNames.entries()) {
    const row = document.createElement("tr");
    const label = document.createElement("td");
    label.innerText = rname;
    row.appendChild(label);
    for (const [cidx, cname] of columnNames.entries()) {
      const cell = document.createElement("td");
      cell.innerText = "n/a";
      const inner = mom.get(ridx);
      if (inner !== undefined) {
        const val = inner.get(cidx);
        if (val !== undefined) {
          cell.innerText = val.toFixed(3);
        }
      }
      row.appendChild(cell);
    }
    table.append(row);
  }
  return table;
}

async function onButtonClick(ev) {
  const progress = document.getElementById("progress");
  progress.hidden = false;
  progress.innerText = "Creating NBlaster...";
  const arena = new NBlaster(
    CACHE.smatArgs.distThresholds,
    CACHE.smatArgs.dotThresholds,
    CACHE.smatArgs.cells,
    parseInt(document.getElementById("kInput").value)
  );

  CACHE.idxs = [];
  let idx = 0;
  console.log("creating dotprops");
  for (let p of CACHE.points) {
    progress.innerText = "Creating dotprops for cloud " + idx;
    CACHE.idxs.push(arena.addNeuron(p));
    console.log("dotprops created for " + idx);
    idx++;
  }

  console.log("running nblast");
  progress.innerText = `Running NBLAST query for ${idx}x${idx}`;
  const result = arena.allVAll(
    document.getElementById("normalizeInput").checked,
    document.getElementById("symmetryInput").value,
    document.getElementById("alphaInput").checked,
    document.getElementById("maxCentroidDistInput").value
  );
  CACHE.result = result;

  console.log("generating table");
  progress.innerText = "Generating output table";
  const tableDiv = document.getElementById("resultsDiv");
  tableDiv.textContent = "";
  const table = tableFromCellMap(result, CACHE.filenames, CACHE.filenames);
  tableDiv.appendChild(table);
  progress.innerText = "Done";
}

async function onCsvChange(ev) {
  const table = document.getElementById("indexTable");
  while (table.children.length > 1) {
    table.removeChild(table.lastElementChild);
  }
  const progress = document.getElementById("progress");
  CACHE.points = [];
  CACHE.filenames = [];
  let idx = 0;
  console.log("parsing points");
  for (const f of ev.target.files) {
    progress.innerText = "Parsing points from " + f.name;
    CACHE.points.push(await parseFile(f, parsePoints));
    CACHE.filenames.push(f.name);
    const td1 = document.createElement("td");
    td1.textContent = idx;
    const td2 = document.createElement("td");
    td2.textContent = f.name;
    const tr = document.createElement("tr");
    tr.appendChild(td1);
    tr.appendChild(td2);
    table.appendChild(tr);
    idx++;
  }
  console.log("points parsed");
  progress.innerText = "Points parsed";

  document.getElementById("button").disabled = false;
}

async function onSmatChange(ev) {
  console.log("parsing smat");
  CACHE.smatArgs = await parseFile(ev.target.files[0], parseSmat);
  document.getElementById("csvInput").disabled = false;
}

function onClearClick(ev) {
  console.log("clearing");
  Object.keys(CACHE).forEach((key) => delete CACHE[key]);
  document.getElementById("smatInput").value = null;
  const csvInput = document.getElementById("csvInput");
  csvInput.value = null;
  csvInput.disabled = true;

  const button = document.getElementById("button");
  button.disabled = true;

  document.getElementById("resultsDiv").innerHTML = "";
  const indexTable = document.getElementById("indexTable");
  while (indexTable.children.length > 1) {
    indexTable.removeChild(indexTable.lastElementChild);
  }
}

init().then(() => {
  let smatInput = document.getElementById("smatInput");
  smatInput.onchange = onSmatChange;

  const csvInput = document.getElementById("csvInput");
  csvInput.onchange = onCsvChange;

  const button = document.getElementById("button");
  button.onclick = onButtonClick;

  const clearButton = document.getElementById("clearButton");
  clearButton.onclick = onClearClick;

  console.log("ready");
  window.CACHE = CACHE;
});
