use crate::models::{FlashQwen3Model, Model, Pplx1Config};
use candle::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use text_embeddings_backend_core::{Batch, ModelType, Pool};

pub struct FlashPplx1Model {
    inner: FlashQwen3Model,
}

impl FlashPplx1Model {
    pub fn load(vb: VarBuilder, config: &Pplx1Config, model_type: ModelType) -> Result<Self> {
        match model_type {
            ModelType::Classifier => {
                candle::bail!("`classifier` model type is not supported for Pplx1")
            }
            ModelType::Embedding(ref pool) => {
                if pool != &Pool::Mean {
                    candle::bail!("Pplx1 only supports mean pooling, got {:?}", pool);
                }
            }
        };

        // NOTE: Pplx1 applies an INT8 quantization head (`tanh()*127`) to the pooled
        // embeddings; fp16 precision is insufficient and yields incorrect results, so
        // this wrapper requires BF16. Routing in `CandleBackend::new` enforces this as
        // well, but the guard is repeated here so the constraint is colocated with the
        // quantization logic.
        if vb.dtype() != DType::BF16 {
            candle::bail!(
                "FlashPplx1 requires DType::BF16 because of the INT8 quantization head, got {:?}",
                vb.dtype()
            );
        }

        // NOTE: Qwen3 but the `config` contains `use_bidirectional_attention=true`
        let inner = FlashQwen3Model::load(vb, config, model_type)?;

        Ok(Self { inner })
    }

    pub fn forward(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (pooled, raw) = self.inner.forward(batch)?;

        // NOTE: Apply Pplx1-specific quantization to pooled embeddings
        let pooled = pooled
            .map(|embeddings| {
                embeddings
                    .tanh() // Apply tanh: [-1, 1]
                    // NOTE: To benefit from the INT8 quantization / scaling, the `normalize`
                    // parameter when generating embeddings should be set to `false`, otherwise the
                    // quantization is "lost"
                    .and_then(|t| t.affine(127.0, 0.0)) // INT8 scale: [-127, 127]
                    .and_then(|t| t.round()) // Round to integers
            })
            .transpose()?;

        Ok((pooled, raw))
    }
}

impl Model for FlashPplx1Model {
    fn is_padded(&self) -> bool {
        self.inner.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.forward(batch)
    }
}
