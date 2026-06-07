use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;
use tokenizers::Tokenizer;

#[allow(dead_code)]
pub struct RerankerModel {
    session: Session,
    tokenizer: Tokenizer,
    pub name: String,
}

impl RerankerModel {
    pub fn load(model_dir: &Path, name: &str) -> Result<Self> {
        let onnx_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
            .commit_from_file(onnx_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {e}"))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            session,
            tokenizer,
            name: name.to_string(),
        })
    }
}
