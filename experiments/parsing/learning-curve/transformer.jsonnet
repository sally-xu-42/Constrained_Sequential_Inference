local train_path = std.extVar("TRAIN_PATH");
local valid_path = std.extVar("VALID_PATH");

local transformer_embedding_dim = 128;
local use_constraints = false;

{
  "dataset_reader": {
    "type": "parsing_reader"
  },
  "train_data_path": train_path,
  "validation_data_path": valid_path,
  "datasets_for_vocab_creation": ["train"],
  "vocabulary": {
    "only_include_pretrained_words": true,
    "pretrained_files": {
      "tokens": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
    }
  },
  "model": {
    "type": "parsing",
    "token_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "embedding_dim": 100,
        "trainable": true,
      }
    },
    "nonterminal_embedder": {
      "type": "embedding",
      "embedding_dim": 20
    },
    "encoder_type": 'Transformer',
    "transformer_embedding_dim": transformer_embedding_dim,
    "transformer_hidden_size": 256,
    "transformer_num_layers": 1,
    "transformer_dropout": 0.1,
    "attention": {
      "type": "mlp",
      "encoder_dim": transformer_embedding_dim * 2,
      "decoder_dim": transformer_embedding_dim * 2,
      "attention_dim": transformer_embedding_dim * 2
    },
    "constraint_set": {
      "namespace": "nonterminals",
      "constraints": [
        {"type": "num-tokens"},
        {"type": "non-empty-phrase"},
        {"type": "balanced-parens", "max_length": 200}
      ]
    },
    "num_layers": 1,
    "dropout": 0.3,
    // During training, we just use unconstrained beam search
    "beam_search": {
      "type": "unconstrained",
      "beam_size": 1,
      "namespace": "nonterminals"
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8,
  },
  "trainer": {
    "num_epochs": 15,
    "cuda_device": 0,
    "shuffle": true,
    "optimizer": {
      "type": "adam",
      "lr": 0.001,
    },
    "learning_rate_scheduler": {
      "type": "step",
      "step_size": 2,
      "gamma": 0.75
    }
  },
  "random_seed": 0,
  "numpy_seed": 0,
  "pytorch_seed": 0,
}
