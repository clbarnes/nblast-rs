use std::fmt::Display;
use std::{collections::HashMap, error::Error};

use js_sys::JsString;
use nblast::Precision;
use nblast::{
    nalgebra::{Unit, Vector3},
    NeuronIdx, RStarTangentsAlphas, RangeTable, ScoreCalc, Symmetry, TangentAlpha,
};
use wasm_bindgen::prelude::*;

fn flat_to_array3<T: Sized + Copy>(v: &[T]) -> Vec<[T; 3]> {
    v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
}

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

fn parse_symmetry(symmetry: Option<JsString>) -> JsResult<Option<Symmetry>> {
    let s = match symmetry {
        Some(s) => {
            let sym_str = s
                .as_string()
                .ok_or_else(|| JsError::new("Invalid symmetry"))?;
            let sym_enum = str_to_sym(&sym_str).map_err(JsError::new)?;
            Some(sym_enum)
        }
        _ => None,
    };
    Ok(s)
}

fn convert_multi_output(
    mut result: HashMap<(NeuronIdx, NeuronIdx), Precision>,
) -> HashMap<NeuronIdx, HashMap<NeuronIdx, Precision>> {
    let mut out: HashMap<NeuronIdx, HashMap<NeuronIdx, f64>> = HashMap::default();
    for ((q, t), v) in result.drain() {
        out.entry(q).or_insert_with(HashMap::default).insert(t, v);
    }
    out
}

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

    pub fn add_points_tangents_alphas(
        &mut self,
        flat_points: &[f64],
        flat_tangents: &[f64],
        alphas: &[f64],
    ) -> JsResult<usize> {
        let tangents_alphas = flat_tangents
            .chunks(3)
            .zip(alphas.iter())
            .map(|(t, a)| TangentAlpha {
                tangent: Unit::new_unchecked(Vector3::new(t[0], t[1], t[2])),
                alpha: *a,
            })
            .collect();
        let points = flat_to_array3(flat_points);
        let neuron = RStarTangentsAlphas::new_with_tangents_alphas(points, tangents_alphas)
            .map_err(JsError::new)?;
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
        let sym = parse_symmetry(symmetry)?;
        Ok(self
            .arena
            .query_target(query_idx, target_idx, normalize, &sym, use_alpha))
    }

    pub fn queries_targets(
        &self,
        query_idxs: &[NeuronIdx],
        target_idxs: &[NeuronIdx],
        normalize: bool,
        symmetry: Option<JsString>,
        use_alpha: bool,
        max_centroid_dist: Option<Precision>,
    ) -> JsResult<JsValue> {
        let sym = parse_symmetry(symmetry)?;
        let out = convert_multi_output(self.arena.queries_targets(
            query_idxs,
            target_idxs,
            normalize,
            &sym,
            use_alpha,
            None,
            max_centroid_dist,
        ));
        Ok(serde_wasm_bindgen::to_value(&out)?)
    }

    pub fn all_v_all(
        &self,
        normalize: bool,
        symmetry: Option<JsString>,
        use_alpha: bool,
        max_centroid_dist: Option<Precision>,
    ) -> JsResult<JsValue> {
        let sym = parse_symmetry(symmetry)?;
        let out = convert_multi_output(self.arena.all_v_all(
            normalize,
            &sym,
            use_alpha,
            None,
            max_centroid_dist,
        ));
        Ok(serde_wasm_bindgen::to_value(&out)?)
    }
}
