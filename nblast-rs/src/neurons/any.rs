#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Enum for serializing/ deserializing any kind of neuron.
///
/// Get neurons in and out of it with `From` and `TryFrom` respectively.
#[derive(Serialize, Deserialize)]
pub enum AnyNeuron {
    #[cfg(feature = "bosque")]
    Bosque(super::bosque::BosqueNeuron),
    #[cfg(feature = "kiddo")]
    Kiddo(super::kiddo::KiddoNeuron),
    #[cfg(feature = "kiddo")]
    ApproxKiddo(super::kiddo::ApproxKiddoNeuron),
    #[cfg(feature = "rstar")]
    Rstar(super::rstar::RstarNeuron),
    // nabo doesn't implement Ser/Deser because it uses the Point trait rather than a specific type internally.

    // can't implement neuron traits because iterators have different types,
    // but could Box them
}

#[derive(Debug, thiserror::Error)]
#[error("Wrong neuron backend: expected '{expected}', got '{got}'")]
pub struct WrongNeuronType {
    expected: &'static str,
    got: &'static str,
}

impl WrongNeuronType {
    fn new(expected: &'static str, got: &'static str) -> Self {
        Self { expected, got }
    }
}

#[cfg(feature = "rstar")]
impl From<super::rstar::RstarNeuron> for AnyNeuron {
    fn from(value: super::rstar::RstarNeuron) -> Self {
        Self::Rstar(value)
    }
}

#[cfg(feature = "rstar")]
impl TryFrom<AnyNeuron> for super::rstar::RstarNeuron {
    type Error = WrongNeuronType;

    fn try_from(value: AnyNeuron) -> Result<Self, Self::Error> {
        use AnyNeuron::*;
        let expected = "rstar";
        match value {
            #[cfg(feature = "rstar")]
            Rstar(n) => Ok(n),
            #[cfg(feature = "kiddo")]
            Kiddo(_) => Err(WrongNeuronType::new(expected, "kiddo")),
            #[cfg(feature = "kiddo")]
            ApproxKiddo(_) => Err(WrongNeuronType::new(expected, "approx_kiddo")),
            #[cfg(feature = "bosque")]
            Bosque(_) => Err(WrongNeuronType::new(expected, "bosque")),
        }
    }
}

#[cfg(feature = "kiddo")]

impl From<super::kiddo::KiddoNeuron> for AnyNeuron {
    fn from(value: super::kiddo::KiddoNeuron) -> Self {
        Self::Kiddo(value)
    }
}
#[cfg(feature = "kiddo")]
impl TryFrom<AnyNeuron> for super::kiddo::KiddoNeuron {
    type Error = WrongNeuronType;

    fn try_from(value: AnyNeuron) -> Result<Self, Self::Error> {
        use AnyNeuron::*;
        let expected = "kiddo";
        match value {
            #[cfg(feature = "rstar")]
            Rstar(_) => Err(WrongNeuronType::new(expected, "rstar")),
            #[cfg(feature = "kiddo")]
            Kiddo(n) => Ok(n),
            #[cfg(feature = "kiddo")]
            ApproxKiddo(_) => Err(WrongNeuronType::new(expected, "approx_kiddo")),
            #[cfg(feature = "bosque")]
            Bosque(_) => Err(WrongNeuronType::new(expected, "bosque")),
        }
    }
}

#[cfg(feature = "kiddo")]

impl From<super::kiddo::ApproxKiddoNeuron> for AnyNeuron {
    fn from(value: super::kiddo::ApproxKiddoNeuron) -> Self {
        Self::ApproxKiddo(value)
    }
}

#[cfg(feature = "kiddo")]
impl TryFrom<AnyNeuron> for super::kiddo::ApproxKiddoNeuron {
    type Error = WrongNeuronType;

    fn try_from(value: AnyNeuron) -> Result<Self, Self::Error> {
        use AnyNeuron::*;
        let expected = "approx_kiddo";
        match value {
            #[cfg(feature = "rstar")]
            Rstar(_) => Err(WrongNeuronType::new(expected, "rstar")),
            #[cfg(feature = "kiddo")]
            Kiddo(_) => Err(WrongNeuronType::new(expected, "kiddo")),
            #[cfg(feature = "kiddo")]
            ApproxKiddo(n) => Ok(n),
            #[cfg(feature = "bosque")]
            Bosque(_) => Err(WrongNeuronType::new(expected, "bosque")),
        }
    }
}

#[cfg(feature = "bosque")]

impl From<super::bosque::BosqueNeuron> for AnyNeuron {
    fn from(value: super::bosque::BosqueNeuron) -> Self {
        Self::Bosque(value)
    }
}

#[cfg(feature = "bosque")]
impl TryFrom<AnyNeuron> for super::bosque::BosqueNeuron {
    type Error = WrongNeuronType;

    fn try_from(value: AnyNeuron) -> Result<Self, Self::Error> {
        use AnyNeuron::*;
        let expected = "bosque";
        match value {
            #[cfg(feature = "rstar")]
            Rstar(_) => Err(WrongNeuronType::new(expected, "rstar")),
            #[cfg(feature = "kiddo")]
            Kiddo(_) => Err(WrongNeuronType::new(expected, "kiddo")),
            #[cfg(feature = "kiddo")]
            ApproxKiddo(_) => Err(WrongNeuronType::new(expected, "approx_kiddo")),
            #[cfg(feature = "bosque")]
            Bosque(n) => Ok(n),
        }
    }
}
