use anyhow::{Context, Result};
use ndarray::Array2;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;
use tokenizers::EncodeInput;
use tokenizers::Tokenizer;

const MAX_DOCUMENTS: usize = 500;
const MAX_SEQ_LEN: usize = 8192;

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

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        Ok(Self {
            session,
            tokenizer,
            name: name.to_string(),
        })
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        if documents.len() > MAX_DOCUMENTS {
            return Err(anyhow::anyhow!(
                "Too many documents: {}. Max: {}",
                documents.len(),
                MAX_DOCUMENTS
            ));
        }

        let pairs: Vec<EncodeInput> = documents
            .iter()
            .map(|doc| EncodeInput::Dual(query.to_string().into(), doc.clone().into()))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let batch_size = documents.len();
        let mut max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        if max_len > MAX_SEQ_LEN {
            tracing::warn!(
                "Sequence length {} exceeds model max {}. Truncating.",
                max_len,
                MAX_SEQ_LEN
            );
            max_len = MAX_SEQ_LEN;
        }

        let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            for j in 0..len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                token_type_ids[[i, j]] = type_ids[j] as i64;
            }
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids.clone().into_dyn())
                .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?,
            "attention_mask" => Value::from_array(attention_mask.clone().into_dyn())
                .map_err(|e| anyhow::anyhow!("Failed to create attention mask tensor: {e}"))?,
            "token_type_ids" => Value::from_array(token_type_ids.into_dyn())
                .map_err(|e| anyhow::anyhow!("Failed to create token type ids tensor: {e}"))?,
        ])?;

        let (_name, value) = outputs.iter().next().context("Model produced no outputs")?;
        let (_shape, data) = value
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let mut results: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, sigmoid(logit)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    pub fn count_tokens(&self, query: &str, doc: &str) -> Result<u32> {
        let encoding = self
            .tokenizer
            .encode(
                EncodeInput::Dual(query.to_string().into(), doc.to_string().into()),
                true,
            )
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        Ok(encoding.len() as u32)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
