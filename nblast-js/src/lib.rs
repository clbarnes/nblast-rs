use std::error::Error;
use std::fmt::Display;

use js_sys::JsString;
use nblast::{NeuronIdx, RStarTangentsAlphas, RangeTable, ScoreCalc, Symmetry};
use wasm_bindgen::prelude::*;

fn flat_to_array3<T: Sized + Copy>(v: &[T]) -> Vec<[T; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

// fn vec_to_array3<T: Sized + Copy>(v: &Vec<T>) -> [T; 3] {
//     [v[0], v[1], v[2]]
// }

// fn vec_to_unitvector3<T: 'static + Sized + Copy + PartialEq + Debug>(
//     v: &Vec<T>,
// ) -> Unit<Vector3<T>> {
//     Unit::new_unchecked(Vector3::new(v[0], v[1], v[2]))
// }

fn str_to_sym(s: &str) -> Result<Symmetry, &str> {
    let msg = "Did not recognise symmetry type";
    match s {
        "arithmetic_mean" => Ok(Symmetry::ArithmeticMean),
        "geometric_mean" => Ok(Symmetry::GeometricMean),
        "harmonic_mean" => Ok(Symmetry::HarmonicMean),
        "min" => Ok(Symmetry::Min),
        "max" => Ok(Symmetry::Max),
        _ => Err(msg),
    }
}

#[wasm_bindgen]
pub struct NblastArena {
    arena: nblast::NblastArena<RStarTangentsAlphas>,
    k: usize,
}

fn to_js_err<E: Display + Error>(e: E) -> JsError {
    JsError::new(&format!("{}", e))
}

type JsResult<T> = Result<T, JsError>;

#[wasm_bindgen]
impl NblastArena {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dist_thresholds: &[f64],
        dot_thresholds: &[f64],
        cells: &[f64],
        k: usize,
    ) -> JsResult<NblastArena> {
        let rtable = RangeTable::new_from_bins(
            vec![dist_thresholds.to_vec(), dot_thresholds.to_vec()],
            cells.to_vec(),
        )
        .map_err(to_js_err)?;
        let score_calc = ScoreCalc::Table(rtable);
        Ok(Self {
            arena: nblast::NblastArena::new(score_calc),
            k,
        })
    }

    pub fn add_points(&mut self, flat_points: &[f64]) -> JsResult<usize> {
        let points = flat_to_array3(flat_points);
        let neuron = RStarTangentsAlphas::new(points, self.k).map_err(JsError::new)?;
        Ok(self.arena.add_neuron(neuron))
    }

    pub fn query_target(
        &self,
        query_idx: NeuronIdx,
        target_idx: NeuronIdx,
        normalize: bool,
        symmetry: Option<JsString>,
        use_alpha: bool,
    ) -> JsResult<Option<f64>> {
        let sym = match symmetry {
            Some(s) => {
                let sym_str = s
                    .as_string()
                    .ok_or_else(|| JsError::new("Invalid symmetry"))?;
                let sym_enum = str_to_sym(&sym_str).map_err(JsError::new)?;
                Some(sym_enum)
            }
            _ => None,
        };
        Ok(self
            .arena
            .query_target(query_idx, target_idx, normalize, &sym, use_alpha))
    }
}

#[wasm_bindgen]
extern "C" {
    pub fn alert(s: &str);
}

#[wasm_bindgen]
pub fn nblast(name: &str) {
    alert(&format!("Hello, {}!", name));
}
