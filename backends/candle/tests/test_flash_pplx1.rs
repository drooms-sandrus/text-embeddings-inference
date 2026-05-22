#![allow(dead_code, unused_imports)]
mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_pplx1_rejects_fp16() -> Result<()> {
    let (model_root, _) = download_artifacts("perplexity-ai/pplx-embed-v1-0.6b", None, None)?;

    // Pplx1 with flash attention requires BF16 because the INT8 quantization head
    // (`tanh()*127`) yields incorrect embeddings in fp16. Loading the backend with
    // `float16` must surface a clear error rather than silently producing bad output.
    let err = CandleBackend::new(
        &model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )
    .err()
    .expect("expected float16 to be rejected for Pplx1");
    let msg = err.to_string();
    assert!(
        msg.contains("F16") || msg.contains("float16") || msg.contains("BF16"),
        "unexpected error message: {msg}"
    );

    Ok(())
}


#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_pplx1embed_bf16() -> Result<()> {
    let (model_root, _) = download_artifacts("perplexity-ai/pplx-embed-v1-0.6b", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "bfloat16".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("pplx1_bf16_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("pplx1_bf16_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_pplx1_quantization_bf16() -> Result<()> {
    let (model_root, _) = download_artifacts("perplexity-ai/pplx-embed-v1-0.6b", None, None)?;
    let tokenizer = load_tokenizer(&model_root)?;

    let backend = CandleBackend::new(
        &model_root,
        "bfloat16".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let input = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let embeddings_map = backend.embed(input)?;
    let (pooled_embeddings, _) = sort_embeddings(embeddings_map);
    let embeddings = &pooled_embeddings[0];

    for value in embeddings.iter() {
        assert!(
            *value >= -127.0 && *value <= 127.0,
            "Value {} is outside [-127, 127] range",
            value
        );
        let rounded = value.round();
        assert!(
            (value - rounded).abs() < 0.01,
            "Value {} is not close to an integer (rounded: {})",
            value,
            rounded
        );
    }

    Ok(())
}
