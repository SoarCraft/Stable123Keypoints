# Zero123Plus v1.2 模型结构分析报告

**模型**: sudo-ai/zero123plus-v1.2

## Pipeline 组件概览

| 组件名称               | 组件类型                        |
| ---------------------- | ------------------------------- |
| vae                    | AutoencoderKL                   |
| text_encoder           | CLIPTextModel                   |
| tokenizer              | CLIPTokenizer                   |
| unet                   | UNet2DConditionModel            |
| scheduler              | EulerAncestralDiscreteScheduler |
| vision_encoder         | CLIPVisionModelWithProjection   |
| feature_extractor_clip | CLIPImageProcessor              |
| feature_extractor_vae  | CLIPImageProcessor              |

## 模型参数统计

| 组件名称       | 参数数量          |
| -------------- | ----------------- |
| vae            | 83,653,863        |
| text_encoder   | 340,387,840       |
| unet           | 865,910,724       |
| vision_encoder | 632,076,800       |
| **总计**       | **1,922,029,227** |

## 详细模型结构

### vae

```cs
AutoencoderKL(
  (encoder): Encoder(
    (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (down_blocks): ModuleList(
      (0): DownEncoderBlock2D(
        (resnets): ModuleList(
          (0-1): 2 x ResnetBlock2D(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
        (downsamplers): ModuleList(
          (0): Downsample2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
          )
        )
      )
      (1): DownEncoderBlock2D(
        (resnets): ModuleList(
          (0): ResnetBlock2D(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
            (conv_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock2D(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
        (downsamplers): ModuleList(
          (0): Downsample2D(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
          )
        )
      )
      (2): DownEncoderBlock2D(
        (resnets): ModuleList(
          (0): ResnetBlock2D(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
            (conv_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock2D(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
        (downsamplers): ModuleList(
          (0): Downsample2D(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
          )
        )
      )
      (3): DownEncoderBlock2D(
        (resnets): ModuleList(
          (0-1): 2 x ResnetBlock2D(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
      )
    )
    (mid_block): UNetMidBlock2D(
      (attentions): ModuleList(
        (0): Attention(
          (group_norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (to_q): Linear(in_features=512, out_features=512, bias=True)
          (to_k): Linear(in_features=512, out_features=512, bias=True)
          (to_v): Linear(in_features=512, out_features=512, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
    (conv_norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
    (conv_act): SiLU()
    (conv_out): Conv2d(512, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (decoder): Decoder(
    (conv_in): Conv2d(4, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (up_blocks): ModuleList(
      (0-1): 2 x UpDecoderBlock2D(
        (resnets): ModuleList(
          (0-2): 3 x ResnetBlock2D(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
        (upsamplers): ModuleList(
          (0): Upsample2D(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (2): UpDecoderBlock2D(
        (resnets): ModuleList(
          (0): ResnetBlock2D(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
            (conv_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1-2): 2 x ResnetBlock2D(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
        (upsamplers): ModuleList(
          (0): Upsample2D(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (3): UpDecoderBlock2D(
        (resnets): ModuleList(
          (0): ResnetBlock2D(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
            (conv_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          )
          (1-2): 2 x ResnetBlock2D(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nonlinearity): SiLU()
          )
        )
      )
    )
    (mid_block): UNetMidBlock2D(
      (attentions): ModuleList(
        (0): Attention(
          (group_norm): GroupNorm(32, 512, eps=1e-06, affine=True)
          (to_q): Linear(in_features=512, out_features=512, bias=True)
          (to_k): Linear(in_features=512, out_features=512, bias=True)
          (to_v): Linear(in_features=512, out_features=512, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=512, out_features=512, bias=True)
            (1): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
    (conv_norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
    (conv_act): SiLU()
    (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (quant_conv): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
  (post_quant_conv): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
)
```

### text_encoder

```cs
CLIPTextModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(49408, 1024)
      (position_embedding): Embedding(77, 1024)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-22): 23 x CLIPEncoderLayer(
          (self_attn): CLIPAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
  )
)
```

### tokenizer

```cs
CLIPTokenizer(
  vocab_size=49408,
  model_max_length=77,
  is_fast=False,
  padding_side='right',
  truncation_side='right',
  special_tokens={'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>', 'pad_token': '!'},
  clean_up_tokenization_spaces=True,
  added_tokens_decoder={
  	0: AddedToken("!", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
  	49406: AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
  	49407: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
  }
)
```

### unet

```cs
UNet2DConditionModel(
  (conv_in): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (time_proj): Timesteps()
  (time_embedding): TimestepEmbedding(
    (linear_1): Linear(in_features=320, out_features=1280, bias=True)
    (act): SiLU()
    (linear_2): Linear(in_features=1280, out_features=1280, bias=True)
  )
  (down_blocks): ModuleList(
    (0): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=320, out_features=320, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=1024, out_features=320, bias=False)
                (to_v): Linear(in_features=1024, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=320, out_features=320, bias=True)
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=640, out_features=640, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=1024, out_features=640, bias=False)
                (to_v): Linear(in_features=1024, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=640, out_features=640, bias=True)
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
          (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnDownBlock2D(
      (attentions): ModuleList(
        (0-1): 2 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
      (downsamplers): ModuleList(
        (0): Downsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
      )
    )
    (3): DownBlock2D(
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
        )
      )
    )
  )
  (up_blocks): ModuleList(
    (0): UpBlock2D(
      (resnets): ModuleList(
        (0-2): 3 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (1): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
        )
      )
      (resnets): ModuleList(
        (0-1): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
          (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
          (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (2): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=640, out_features=640, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=640, out_features=640, bias=False)
                (to_v): Linear(in_features=640, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=640, out_features=640, bias=False)
                (to_k): Linear(in_features=1024, out_features=640, bias=False)
                (to_v): Linear(in_features=1024, out_features=640, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=640, out_features=640, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=640, out_features=5120, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=2560, out_features=640, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=640, out_features=640, bias=True)
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
          (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): ResnetBlock2D(
          (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
          (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
          (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (upsamplers): ModuleList(
        (0): Upsample2D(
          (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (3): CrossAttnUpBlock2D(
      (attentions): ModuleList(
        (0-2): 3 x Transformer2DModel(
          (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=320, out_features=320, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn1): Attention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=320, out_features=320, bias=False)
                (to_v): Linear(in_features=320, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (attn2): Attention(
                (to_q): Linear(in_features=320, out_features=320, bias=False)
                (to_k): Linear(in_features=1024, out_features=320, bias=False)
                (to_v): Linear(in_features=1024, out_features=320, bias=False)
                (to_out): ModuleList(
                  (0): Linear(in_features=320, out_features=320, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              (ff): FeedForward(
                (net): ModuleList(
                  (0): GEGLU(
                    (proj): Linear(in_features=320, out_features=2560, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=1280, out_features=320, bias=True)
                )
              )
            )
          )
          (proj_out): Linear(in_features=320, out_features=320, bias=True)
        )
      )
      (resnets): ModuleList(
        (0): ResnetBlock2D(
          (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
          (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
        )
        (1-2): 2 x ResnetBlock2D(
          (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
          (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
          (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
          (dropout): Dropout(p=0.0, inplace=False)
          (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (nonlinearity): SiLU()
          (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
        )
      )
    )
  )
  (mid_block): UNetMidBlock2DCrossAttn(
    (attentions): ModuleList(
      (0): Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
          (0): BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1280, out_features=1280, bias=False)
              (to_v): Linear(in_features=1280, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
              (to_q): Linear(in_features=1280, out_features=1280, bias=False)
              (to_k): Linear(in_features=1024, out_features=1280, bias=False)
              (to_v): Linear(in_features=1024, out_features=1280, bias=False)
              (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
              )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
              (net): ModuleList(
                (0): GEGLU(
                  (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
              )
            )
          )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
      )
    )
    (resnets): ModuleList(
      (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
      )
    )
  )
  (conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
  (conv_act): SiLU()
  (conv_out): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
```

### scheduler

```cs
EulerAncestralDiscreteScheduler {
  "_class_name": "EulerAncestralDiscreteScheduler",
  "_diffusers_version": "0.35.1",
  "beta_end": 0.012,
  "beta_schedule": "linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "prediction_type": "v_prediction",
  "rescale_betas_zero_snr": false,
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "timestep_spacing": "linspace",
  "trained_betas": null
}
```

### vision_encoder

```cs
CLIPVisionModelWithProjection(
  (vision_model): CLIPVisionTransformer(
    (embeddings): CLIPVisionEmbeddings(
      (patch_embedding): Conv2d(3, 1280, kernel_size=(14, 14), stride=(14, 14), bias=False)
      (position_embedding): Embedding(257, 1280)
    )
    (pre_layrnorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-31): 32 x CLIPEncoderLayer(
          (self_attn): CLIPAttention(
            (k_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (v_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (q_proj): Linear(in_features=1280, out_features=1280, bias=True)
            (out_proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (layer_norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=1280, out_features=5120, bias=True)
            (fc2): Linear(in_features=5120, out_features=1280, bias=True)
          )
          (layer_norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  )
  (visual_projection): Linear(in_features=1280, out_features=1024, bias=False)
)
```

### feature_extractor_clip

```cs
CLIPImageProcessor {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}
```

### feature_extractor_vae

```cs
CLIPImageProcessor {
  "crop_size": {
    "height": 512,
    "width": 512
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": 0.5,
  "image_processor_type": "CLIPImageProcessor",
  "image_std": 0.8,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 512
  }
}
```

## Pipeline 配置

| 配置项                 | 值                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| vae                    | ('diffusers', 'AutoencoderKL')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| text_encoder           | ('transformers', 'CLIPTextModel')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| tokenizer              | ('transformers', 'CLIPTokenizer')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| unet                   | ('diffusers', 'UNet2DConditionModel')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| scheduler              | ('diffusers', 'EulerAncestralDiscreteScheduler')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| safety_checker         | (None, None)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| vision_encoder         | ('transformers', 'CLIPVisionModelWithProjection')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| feature_extractor_clip | ('transformers', 'CLIPImageProcessor')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| feature_extractor_vae  | ('transformers', 'CLIPImageProcessor')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ramping_coefficients   | [0.00301829120144248, 0.2204633206129074, 0.21527841687202454, 0.23498539626598358, 0.1914631873369217, 0.20188239216804504, 0.19352824985980988, 0.17249998450279236, 0.15826298296451569, 0.15236389636993408, 0.13444548845291138, 0.12044154852628708, 0.12808501720428467, 0.1271015852689743, 0.13629068434238434, 0.14516159892082214, 0.15645112097263336, 0.16885493695735931, 0.18022602796554565, 0.1958882212638855, 0.21415705978870392, 0.23056700825691223, 0.2505834102630615, 0.2574525773525238, 0.275470107793808, 0.2808215022087097, 0.29953837394714355, 0.2967497408390045, 0.2883710563182831, 0.3023308515548706, 0.3054688572883606, 0.32596179842948914, 0.3225354254245758, 0.3140765428543091, 0.3288663625717163, 0.3435625731945038, 0.3342442810535431, 0.32937031984329224, 0.35734811425209045, 0.3601177930831909, 0.3517529368400574, 0.3810708224773407, 0.40007662773132324, 0.4264647364616394, 0.3977527916431427, 0.4314143657684326, 0.49558719992637634, 0.4665665030479431, 0.48960328102111816, 0.5141982436180115, 0.5230164527893066, 0.5266074538230896, 0.5456079840660095, 0.5737904906272888, 0.5882097482681274, 0.6210350394248962, 0.6530380845069885, 0.6383244395256042, 0.6792004704475403, 0.6567418575286865, 0.7517656683921814, 0.736494243144989, 0.7586457133293152, 0.8130561709403992, 0.9578766226768494, 1.001284122467041, 0.9404520988464355, 1.004292368888855, 0.9145274758338928, 0.9771682620048523, 1.0350638628005981, 1.0265849828720093, 1.0594775676727295, 0.980824887752533, 1.0715670585632324, 1.0140161514282227, 1.1983819007873535] |
| \_name_or_path         | sudo-ai/zero123plus-v1.2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## 调度器信息

**类型**: EulerAncestralDiscreteScheduler

### 调度器配置

| 配置项                 | 值                              |
| ---------------------- | ------------------------------- |
| num_train_timesteps    | 1000                            |
| beta_start             | 0.00085                         |
| beta_end               | 0.012                           |
| beta_schedule          | linear                          |
| trained_betas          | None                            |
| prediction_type        | v_prediction                    |
| timestep_spacing       | linspace                        |
| steps_offset           | 1                               |
| rescale_betas_zero_snr | False                           |
| \_use_default_values   | ['rescale_betas_zero_snr']      |
| \_class_name           | EulerAncestralDiscreteScheduler |
| \_diffusers_version    | 0.35.1                          |
| clip_sample            | False                           |
| set_alpha_to_one       | False                           |
| skip_prk_steps         | True                            |
