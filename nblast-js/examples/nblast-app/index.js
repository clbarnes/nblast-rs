
import init, { NblastArena } from "./node_modules/nblast-js/nblast_js.js";

const CACHE = {};

class NBlaster {
  constructor(distThresholds, dotThresholds, cells, k) {
    this.arena = new NblastArena(
      new Float64Array(distThresholds),
      new Float64Array(dotThresholds),
      new Float64Array(cells),
      Math.round(k),
    );
  }

  addPoints(points) {
    return this.arena.add_points(new Float64Array(points.flat()));
  }

  addPointsTangentsAlphas(points, tangents, alphas) {
    return this.arena.add_points_tangents_alphas(
      new Float64Array(points.flat()),
      new Float64Array(tangents.flat()),
      new Float64Array(alphas),
    )
  }

  queryTarget(queryIdx, targetIdx, normalize, symmetry, useAlpha) {
    const sym = symmetry ? symmetry.toString() : undefined;
    return this.arena.query_target(
      Math.round(queryIdx),
      Math.round(targetIdx),
      !!normalize,
      sym,
      !!useAlpha,
    );
  }

  queriesTargets(queryIdxs, targetIdxs, normalize, symmetry, useAlpha) {
    const sym = symmetry ? symmetry.toString() : undefined;
    return this.arena.queries_targets(
      new BigUint64Array(queryIdxs),
      new BigUint64Array(targetIdxs),
      !!normalize,
      sym,
      !!useAlpha,
    );
  }

  allVsAll(normalize, symmetry, useAlpha) {
    const sym = symmetry ? symmetry.toString() : undefined;
    return this.arena.all_v_all(!!normalize, sym, !!useAlpha);
  }
}

function parsePoints(str) {
  const points = str.split("\n")
    .slice(1)
    .map((r) => r.split(",").slice(1).map(parseFloat));
  return points;
}

function parseSmat(str) {
  const rows = str.split("\n").filter((r) => !!r.trim()).map((r) => r.split(","));
  const dotBoundaries = [0];
  for (let idx = 2; idx < rows[0].length; idx += 2) {
    dotBoundaries.push(parseFloat(rows[0][idx].slice(0, -2)))
  }

  const distBoundaries = [0];
  const values = [];
  for (const row of rows.slice(1)) {
    if (!row.length) {
      continue
    }
    distBoundaries.push(parseFloat(row[1].slice(0, -2)));
    row.slice(2).forEach((el) => values.push(parseFloat(el)));
  }
  return {
    distThresholds: distBoundaries,
    dotThresholds: dotBoundaries,
    cells: values,
  }
}

function parseFile(file, strParser) {
  const reader = new FileReader();
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
    row.appendChild(label)
    for (const [cidx, cname] of columnNames.entries()) {
      const cell = document.createElement("td");
      const inner = mom.get(ridx);
      if (inner !== undefined) {
        const val = inner.get(cidx);
        if (val !== undefined) {
          cell.innerText = val;
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
  progress.innerText = "Creating NBlaster..."
  const arena = new NBlaster(
    CACHE.smatArgs.distThresholds,
    CACHE.smatArgs.dotThresholds,
    CACHE.smatArgs.cells,
    parseInt(document.getElementById("kInput").value),
  );

  CACHE.idxs = [];
  let idx = 0;
  console.log("creating dotprops");
  for (let p of CACHE.points) {
    progress.innerText = "Creating dotprops for cloud " + idx;
    CACHE.idxs.push(arena.addPoints(p));
    console.log("dotprops created for " + idx);
    idx++;
  }

  console.log("running nblast");
  progress.innerText = `Running NBLAST query for ${idx}x${idx}`;
  const result = arena.allVsAll(
    document.getElementById("normalizeInput").checked,
    document.getElementById("symmetryInput").value,
    document.getElementById("alphaInput").checked,
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
  progress.innerText = "Points parsed"

  document.getElementById("button").disabled = false;
}

async function onSmatChange(ev) {
  console.log("parsing smat");
  CACHE.smatArgs = await parseFile(ev.target.files[0], parseSmat);
  document.getElementById("csvInput").disabled = false;
}

function onClearClick(ev) {
  console.log("clearing")
  Object.keys(CACHE).forEach(key => delete CACHE[key]);
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
  let smatInput = document.getElementById("smatInput")
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
