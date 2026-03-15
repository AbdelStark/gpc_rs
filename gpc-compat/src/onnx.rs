//! ONNX model inspection and inference via Tract.

use std::path::Path;

use tract_onnx::prelude::*;

type TypedSimplePlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// ONNX model inspector and inference runner.
pub struct OnnxInspector {
    model: TypedSimplePlan,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl OnnxInspector {
    /// Load an ONNX model from file.
    pub fn load(path: &Path) -> gpc_core::Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| gpc_core::GpcError::Model(format!("Failed to load ONNX: {e}")))?
            .into_optimized()
            .map_err(|e| gpc_core::GpcError::Model(format!("Failed to optimize: {e}")))?
            .into_runnable()
            .map_err(|e| gpc_core::GpcError::Model(format!("Failed to make runnable: {e}")))?;

        let input_shapes: Vec<Vec<usize>> = model
            .model()
            .input_outlets()
            .iter()
            .filter_map(|outlets| {
                // input_outlets returns &[OutletId]
                outlets.iter().next().and_then(|o| {
                    model
                        .model()
                        .outlet_fact(*o)
                        .ok()
                        .and_then(|f| f.shape.as_concrete().map(|s| s.to_vec()))
                })
            })
            .collect();

        let output_shapes: Vec<Vec<usize>> = model
            .model()
            .output_outlets()
            .iter()
            .filter_map(|outlets| {
                outlets.iter().next().and_then(|o| {
                    model
                        .model()
                        .outlet_fact(*o)
                        .ok()
                        .and_then(|f| f.shape.as_concrete().map(|s| s.to_vec()))
                })
            })
            .collect();

        Ok(Self {
            model,
            input_shapes,
            output_shapes,
        })
    }

    /// Get input shapes of the model.
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }

    /// Get output shapes of the model.
    pub fn output_shapes(&self) -> &[Vec<usize>] {
        &self.output_shapes
    }

    /// Run inference with f32 input tensors.
    pub fn run(&self, inputs: Vec<Tensor>) -> gpc_core::Result<Vec<Vec<f32>>> {
        let result = self
            .model
            .run(inputs.into_iter().map(|t| t.into()).collect())
            .map_err(|e| gpc_core::GpcError::Model(format!("Inference failed: {e}")))?;

        let outputs: Vec<Vec<f32>> = result
            .iter()
            .enumerate()
            .map(|(i, t)| {
                t.as_slice::<f32>().map(|s| s.to_vec()).map_err(|e| {
                    gpc_core::GpcError::Model(format!(
                        "failed to convert output tensor {i} to f32: {e}"
                    ))
                })
            })
            .collect::<gpc_core::Result<Vec<_>>>()?;

        Ok(outputs)
    }

    /// Get a summary of the model.
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("ONNX Model Summary\n");
        summary.push_str(&format!("  Inputs: {:?}\n", self.input_shapes));
        summary.push_str(&format!("  Outputs: {:?}\n", self.output_shapes));
        summary.push_str(&format!("  Nodes: {}\n", self.model.model().nodes().len()));
        summary
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_onnx_inspector_api_exists() {
        fn _assert_methods(inspector: &super::OnnxInspector) {
            let _ = inspector.input_shapes();
            let _ = inspector.output_shapes();
            let _ = inspector.summary();
        }
    }
}
