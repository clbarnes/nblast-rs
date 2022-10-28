
import init, { NblastArena } from "./node_modules/nblast-js/nblast_js.js";

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

async function onButtonClick(ev) {
  const smat = document.getElementById("smatInput");
  const csv1 = document.getElementById("csvInput1");
  const csv2 = document.getElementById("csvInput2");
  if (smatInput.files.length != 1 || csv1.files.length != 1 || csv1.files.length != 1) {
    alert("Must be exactly one file in smat, csv1, csv2")
    return;
  }
  const smatArgs = await parseFile(smat.files[0], parseSmat);
  const points1 = await parseFile(csv1.files[0], parsePoints);
  const points2 = await parseFile(csv2.files[0], parsePoints);

  const arena = new NBlaster(
    smatArgs.distThresholds,
    smatArgs.dotThresholds,
    smatArgs.cells,
    5, // k
  )
  const idx1 = arena.addPoints(points1);
  const idx2 = arena.addPoints(points2);

  const result = arena.queryTarget(idx1, idx2, false, undefined, false);
  alert("NBLAST score is " + result);
}

init().then(() => {
  const button = document.getElementById("button");
  button.disabled = false;
  button.onclick = onButtonClick;
  console.log("ready");
});
