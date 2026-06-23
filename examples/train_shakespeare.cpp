#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/autograd/ops.hpp"
#include "tiramisu/core/cuda_common.hpp"
#include "tiramisu/core/cuda_memory.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/loss.hpp"
#include "tiramisu/optim/adamw.hpp"
#include "tiramisu/optim/grad_clip.hpp"
#include "tiramisu/optim/lr_scheduler.hpp"
#include "tiramisu/serialize/checkpoint.hpp"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::optim;
using namespace tiramisu::autograd;

struct CharVocab {
  std::unordered_map<char, int64_t> char_to_id;
  std::vector<char> id_to_char;

  int64_t size() const { return static_cast<int64_t>(id_to_char.size()); }

  void build(const std::string& text) {
    std::vector<char> chars;
    for (char c : text) {
      if (char_to_id.find(c) == char_to_id.end()) {
        char_to_id[c] = static_cast<int64_t>(chars.size());
        chars.push_back(c);
      }
    }
    id_to_char = std::move(chars);
  }

  std::vector<int64_t> encode(const std::string& text) const {
    std::vector<int64_t> ids;
    ids.reserve(text.size());
    for (char c : text) {
      ids.push_back(char_to_id.at(c));
    }
    return ids;
  }

  std::string decode(const std::vector<int64_t>& ids) const {
    std::string out;
    out.reserve(ids.size());
    for (int64_t id : ids) {
      out.push_back(id_to_char[static_cast<size_t>(id)]);
    }
    return out;
  }
};

struct TrainConfig {
  std::string data_path = "data/tiny_shakespeare.txt";
  std::string checkpoint_path;
  std::string preset = "tiny";
  int epochs = 3;
  int batch_size = 8;
  int seq_len = 64;
  float lr = 3e-4f;
  float weight_decay = 0.1f;
  float grad_clip = 1.0f;
  int eval_interval = 50;
  int checkpoint_interval = 200;
  int max_batches = -1;
  bool generate = true;
  bool generate_only = false;
  std::string prompt = "First Citizen:\n";
  int sample_chars = 400;
  float temperature = 0.8f;
  int sample_seed = 42;
  bool use_cuda = false;
};

std::string read_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Cannot open: " + path);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

GPTConfig gpt_config_for_preset(const std::string& preset, int64_t vocab_size,
                                int seq_len) {
  if (preset == "10m") {
    return GPTConfig{
        .vocab_size = vocab_size,
        .d_model = 384,
        .num_heads = 6,
        .num_layers = 6,
        .max_seq_len = seq_len,
        .tie_weights = true,
    };
  }
  if (preset == "2m") {
    return GPTConfig{
        .vocab_size = vocab_size,
        .d_model = 200,
        .num_heads = 4,
        .num_layers = 4,
        .max_seq_len = seq_len,
        .tie_weights = true,
    };
  }
  return GPTConfig{
      .vocab_size = vocab_size,
      .d_model = 64,
      .num_heads = 2,
      .num_layers = 2,
      .max_seq_len = seq_len,
      .tie_weights = true,
  };
}

void init_parameters(std::vector<Tensor*>& params, std::mt19937& gen) {
  for (Tensor* p : params) {
    std::vector<float> host(static_cast<size_t>(p->numel()));
    if (p->shape().size() == 2) {
      const int64_t fan_in = p->shape()[0];
      std::normal_distribution<float> dist(
          0.0f, 0.02f / std::sqrt(static_cast<float>(fan_in)));
      for (float& v : host) {
        v = dist(gen);
      }
    } else if (p->shape().size() == 1) {
      std::fill(host.begin(), host.end(), 0.0f);
    }
    if (p->device() == Device::CPU) {
      std::copy(host.begin(), host.end(), p->data<float>());
    } else {
      cuda_mem::copy_bytes(host.data(), p->data<float>(),
                           host.size() * sizeof(float), Device::CPU,
                           p->device());
    }
  }
}

Tensor make_batch_tensor(const std::vector<int64_t>& flat_ids, int64_t batch,
                         int64_t seq, Device device) {
  std::vector<float> host(static_cast<size_t>(batch * seq));
  for (int64_t i = 0; i < batch * seq; ++i) {
    host[static_cast<size_t>(i)] =
        static_cast<float>(flat_ids[static_cast<size_t>(i)]);
  }
  Tensor t({batch, seq}, DType::Float32, device);
  cuda_mem::copy_bytes(host.data(), t.data<float>(), host.size() * sizeof(float),
                       Device::CPU, device);
  return t;
}

float tensor_scalar(const Tensor& t) {
  if (t.device() == Device::CUDA) {
    return cuda_mem::read_scalar_f32(t);
  }
  return t.data<float>()[0];
}

float evaluate_loss(GPT& model, const std::vector<int64_t>& ids, int64_t seq_len,
                    int64_t batch_size, Device device) {
  const int64_t vocab = model.config().vocab_size;
  float total = 0.0f;
  int64_t batches = 0;

  for (size_t start = 0; start + static_cast<size_t>(seq_len) + 1 <= ids.size();
       start += static_cast<size_t>(batch_size * seq_len)) {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> target_ids;
    input_ids.reserve(static_cast<size_t>(batch_size * seq_len));
    target_ids.reserve(static_cast<size_t>(batch_size * seq_len));

    for (int64_t b = 0; b < batch_size; ++b) {
      size_t offset = start + static_cast<size_t>(b * seq_len);
      if (offset + static_cast<size_t>(seq_len) + 1 > ids.size()) {
        break;
      }
      for (int64_t s = 0; s < seq_len; ++s) {
        input_ids.push_back(ids[offset + static_cast<size_t>(s)]);
        target_ids.push_back(ids[offset + static_cast<size_t>(s) + 1]);
      }
    }

    if (input_ids.empty()) break;
    const int64_t actual_batch = static_cast<int64_t>(input_ids.size()) / seq_len;

    Tensor batch_x = make_batch_tensor(input_ids, actual_batch, seq_len, device);
    Tensor batch_y = make_batch_tensor(target_ids, actual_batch, seq_len, device);

    Tensor logits = model.forward(batch_x);
    Tensor flat_logits =
        reshape(logits, {actual_batch * seq_len, vocab});
    Tensor flat_targets = batch_y.reshape({actual_batch * seq_len});
    Tensor loss = cross_entropy_loss(flat_logits, flat_targets);
    total += tensor_scalar(loss);
    batches++;
  }

  return batches > 0 ? total / static_cast<float>(batches) : 0.0f;
}

int64_t sample_next_token(const float* logits, int64_t vocab, float temperature,
                          std::mt19937& rng) {
  if (temperature <= 0.0f) {
    int64_t best = 0;
    for (int64_t i = 1; i < vocab; ++i) {
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    return best;
  }

  float max_logit = logits[0];
  for (int64_t i = 1; i < vocab; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  std::vector<float> probs(static_cast<size_t>(vocab));
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab; ++i) {
    probs[static_cast<size_t>(i)] =
        std::exp((logits[i] - max_logit) / temperature);
    sum += probs[static_cast<size_t>(i)];
  }

  std::uniform_real_distribution<float> dist(0.0f, sum);
  const float target = dist(rng);
  float cumulative = 0.0f;
  for (int64_t i = 0; i < vocab; ++i) {
    cumulative += probs[static_cast<size_t>(i)];
    if (target <= cumulative) {
      return i;
    }
  }
  return vocab - 1;
}

std::string generate_text(GPT& model, const CharVocab& vocab,
                          const std::string& prompt, int64_t max_new_tokens,
                          float temperature, std::mt19937& rng,
                          Device device) {
  NoGradGuard guard;
  std::vector<int64_t> context = vocab.encode(prompt);
  const size_t prompt_len = context.size();

  for (int64_t t = 0; t < max_new_tokens; ++t) {
    Tensor ids =
        make_batch_tensor(context, 1, static_cast<int64_t>(context.size()), device);
    Tensor logits = model.forward(ids);
    const int64_t vocab_size = model.config().vocab_size;
    const int64_t last = logits.shape()[1] - 1;

    std::vector<float> row(static_cast<size_t>(vocab_size));
    if (device == Device::CUDA) {
      cuda_mem::copy_bytes(
          logits.data<float>() + last * vocab_size, row.data(),
          static_cast<std::size_t>(vocab_size) * sizeof(float), Device::CUDA,
          Device::CPU);
    } else {
      std::copy_n(logits.data<float>() + last * vocab_size, vocab_size,
                  row.data());
    }

    const int64_t next_id =
        sample_next_token(row.data(), vocab_size, temperature, rng);
    context.push_back(next_id);
    if (context.size() > static_cast<size_t>(model.config().max_seq_len)) {
      context.erase(context.begin());
    }
  }

  std::vector<int64_t> generated(context.begin() + static_cast<long>(prompt_len),
                                 context.end());
  return vocab.decode(generated);
}

TrainConfig parse_args(int argc, char** argv) {
  TrainConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--data") {
      cfg.data_path = next();
    } else if (arg == "--preset") {
      cfg.preset = next();
    } else if (arg == "--epochs") {
      cfg.epochs = std::stoi(next());
    } else if (arg == "--batch-size") {
      cfg.batch_size = std::stoi(next());
    } else if (arg == "--seq-len") {
      cfg.seq_len = std::stoi(next());
    } else if (arg == "--lr") {
      cfg.lr = std::stof(next());
    } else if (arg == "--checkpoint") {
      cfg.checkpoint_path = next();
    } else if (arg == "--max-batches") {
      cfg.max_batches = std::stoi(next());
    } else if (arg == "--no-generate") {
      cfg.generate = false;
    } else if (arg == "--generate-only") {
      cfg.generate_only = true;
      cfg.generate = true;
      cfg.epochs = 0;
    } else if (arg == "--prompt") {
      cfg.prompt = next();
    } else if (arg == "--sample-chars") {
      cfg.sample_chars = std::stoi(next());
    } else if (arg == "--temperature") {
      cfg.temperature = std::stof(next());
    } else if (arg == "--sample-seed") {
      cfg.sample_seed = std::stoi(next());
    } else if (arg == "--cuda") {
      cfg.use_cuda = true;
    } else if (arg == "--help") {
      std::printf(
          "Usage: train_shakespeare [options]\n"
          "  --data PATH          Corpus file (default: data/tiny_shakespeare.txt)\n"
          "  --preset tiny|2m|10m Model size preset (default: tiny)\n"
          "  --epochs N           Training epochs (default: 3)\n"
          "  --batch-size N       Batch size (default: 8)\n"
          "  --seq-len N          Sequence length (default: 64, use 256 for 10m)\n"
          "  --lr F               Base learning rate (default: 3e-4)\n"
          "  --checkpoint PATH    Save/load checkpoint path\n"
          "  --max-batches N      Stop after N optimizer steps\n"
          "  --no-generate        Skip sampling after training\n"
          "  --generate-only      Load checkpoint and sample (implies --epochs 0)\n"
          "  --prompt TEXT        Generation prompt (default: \"First Citizen:\\n\")\n"
          "  --sample-chars N     Characters to generate (default: 400)\n"
          "  --temperature F      Sampling temperature, 0=greedy (default: 0.8)\n"
          "  --sample-seed N      RNG seed for sampling (default: 42)\n"
          "  --cuda               Train on GPU (requires CUDA build)\n");
      std::exit(0);
    }
  }
  if (cfg.preset == "10m" && cfg.seq_len == 64) {
    cfg.seq_len = 256;
    cfg.batch_size = 16;
    cfg.epochs = 5;
  }
  if (cfg.preset == "2m" && cfg.seq_len == 64) {
    cfg.seq_len = 128;
    cfg.batch_size = 16;
  }
  return cfg;
}

int main(int argc, char** argv) {
  const TrainConfig cfg = parse_args(argc, argv);

  Device device = Device::CPU;
  if (cfg.use_cuda) {
#ifdef TIRAMISU_CUDA_ENABLED
    if (!cuda_available()) {
      throw std::runtime_error("--cuda requested but no CUDA device is available");
    }
    set_device(Device::CUDA);
    device = Device::CUDA;
#else
    throw std::runtime_error("--cuda requested but tiramisu was built without CUDA");
#endif
  }

  const std::string text = read_file(cfg.data_path);
  CharVocab vocab;
  vocab.build(text);
  const std::vector<int64_t> ids = vocab.encode(text);

  const size_t split = ids.size() * 9 / 10;
  std::vector<int64_t> train_ids(ids.begin(), ids.begin() + static_cast<long>(split));
  std::vector<int64_t> val_ids(ids.begin() + static_cast<long>(split), ids.end());

  GPTConfig model_cfg =
      gpt_config_for_preset(cfg.preset, vocab.size(), cfg.seq_len);
  GPT model(gpt_config_for_preset(cfg.preset, vocab.size(), cfg.seq_len), device);
  std::vector<Tensor*> params = model.parameters();
  std::mt19937 gen(1234);
  init_parameters(params, gen);

  int64_t resume_step = 0;
  int64_t resume_epoch = 0;
  if (!cfg.checkpoint_path.empty()) {
    std::ifstream ck(cfg.checkpoint_path);
    if (ck.good()) {
      serialize::load_gpt_model(cfg.checkpoint_path, model, &resume_step,
                                &resume_epoch);
      std::printf("Loaded checkpoint from %s (step %lld, epoch %lld)\n",
                  cfg.checkpoint_path.c_str(),
                  static_cast<long long>(resume_step),
                  static_cast<long long>(resume_epoch));
    }
  }

  AdamW optimizer(params, cfg.lr, 0.9f, 0.999f, 1e-8f, cfg.weight_decay);

  const int64_t steps_per_epoch =
      std::max<int64_t>(1, static_cast<int64_t>(train_ids.size()) /
                               (cfg.batch_size * cfg.seq_len));
  const int64_t total_steps = steps_per_epoch * cfg.epochs;
  CosineAnnealingLR scheduler(cfg.lr, total_steps, cfg.lr * 0.1f);

  std::printf("Preset: %s | device=%s | vocab=%lld d_model=%lld layers=%lld heads=%lld\n",
              cfg.preset.c_str(), device == Device::CUDA ? "cuda" : "cpu",
              static_cast<long long>(model_cfg.vocab_size),
              static_cast<long long>(model_cfg.d_model),
              static_cast<long long>(model_cfg.num_layers),
              static_cast<long long>(model_cfg.num_heads));
  std::printf("Parameters: %lld | train tokens=%zu | val tokens=%zu\n",
              static_cast<long long>(model.count_parameters()), train_ids.size(),
              val_ids.size());

  int64_t global_step = resume_step;
  if (!cfg.generate_only) {
  for (int epoch = resume_epoch; epoch < cfg.epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int64_t batch_count = 0;

    for (size_t start = 0;
         start + static_cast<size_t>(cfg.seq_len) + 1 <= train_ids.size();
         start += static_cast<size_t>(cfg.batch_size * cfg.seq_len)) {
      if (cfg.max_batches >= 0 && global_step >= cfg.max_batches) {
        break;
      }

      std::vector<int64_t> input_ids;
      std::vector<int64_t> target_ids;
      input_ids.reserve(static_cast<size_t>(cfg.batch_size * cfg.seq_len));
      target_ids.reserve(static_cast<size_t>(cfg.batch_size * cfg.seq_len));

      for (int b = 0; b < cfg.batch_size; ++b) {
        size_t offset = start + static_cast<size_t>(b * cfg.seq_len);
        if (offset + static_cast<size_t>(cfg.seq_len) + 1 > train_ids.size()) {
          break;
        }
        for (int s = 0; s < cfg.seq_len; ++s) {
          input_ids.push_back(train_ids[offset + static_cast<size_t>(s)]);
          target_ids.push_back(train_ids[offset + static_cast<size_t>(s) + 1]);
        }
      }
      if (input_ids.empty()) break;

      const int64_t actual_batch =
          static_cast<int64_t>(input_ids.size()) / cfg.seq_len;
      Tensor batch_x =
          make_batch_tensor(input_ids, actual_batch, cfg.seq_len, device);
      Tensor batch_y =
          make_batch_tensor(target_ids, actual_batch, cfg.seq_len, device);

      optimizer.zero_grad();
      Tensor logits = model.forward(batch_x);
      Tensor flat_logits =
          reshape(logits, {actual_batch * cfg.seq_len, model_cfg.vocab_size});
      Tensor flat_targets = batch_y.reshape({actual_batch * cfg.seq_len});
      Tensor loss = cross_entropy_loss(flat_logits, flat_targets);
      backward(loss);

      clip_grad_norm(params, cfg.grad_clip);
      optimizer.step();
      global_step++;

      const float lr = scheduler.step();
      optimizer.set_lr(lr);

      epoch_loss += tensor_scalar(loss);
      batch_count++;

      if (global_step % cfg.eval_interval == 0) {
        const float val_loss =
            evaluate_loss(model, val_ids, cfg.seq_len, cfg.batch_size, device);
        std::printf(
            "step %lld | train_loss %.4f | val_loss %.4f | lr %.6f\n",
            static_cast<long long>(global_step), tensor_scalar(loss), val_loss,
            lr);
      }

      if (!cfg.checkpoint_path.empty() &&
          global_step % cfg.checkpoint_interval == 0) {
        serialize::save_gpt_model(cfg.checkpoint_path, model, global_step, epoch);
        std::printf("Saved checkpoint to %s\n", cfg.checkpoint_path.c_str());
      }
    }

    const float avg_loss =
        batch_count > 0 ? epoch_loss / static_cast<float>(batch_count) : 0.0f;
    const float val_loss =
        evaluate_loss(model, val_ids, cfg.seq_len, cfg.batch_size, device);
    std::printf("epoch %d | avg_train_loss %.4f | val_loss %.4f\n", epoch + 1,
                avg_loss, val_loss);

    if (cfg.max_batches >= 0 && global_step >= cfg.max_batches) {
      break;
    }
  }
  }

  if (!cfg.generate_only && !cfg.checkpoint_path.empty()) {
    serialize::save_gpt_model(cfg.checkpoint_path, model, global_step,
                              cfg.epochs);
    std::printf("Final checkpoint saved to %s\n", cfg.checkpoint_path.c_str());
  }

  if (cfg.generate) {
    std::mt19937 sample_rng(cfg.sample_seed);
    const std::string sample = generate_text(
        model, vocab, cfg.prompt, cfg.sample_chars, cfg.temperature, sample_rng,
        device);
    std::printf("\n--- sample (prompt: %s) ---\n%s\n", cfg.prompt.c_str(),
                sample.c_str());
  }

  return 0;
}
