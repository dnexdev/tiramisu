#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "tiramisu/autograd/grad_mode.hpp"
#include "tiramisu/core/tensor.hpp"
#include "tiramisu/nn/gpt.hpp"
#include "tiramisu/nn/kv_cache.hpp"
#include "tiramisu/serialize/checkpoint.hpp"

#include "shakespeare_vocab.inc"

using namespace tiramisu;
using namespace tiramisu::nn;
using namespace tiramisu::autograd;
using namespace tiramisu::serialize;

namespace {

struct CharVocab {
  std::unordered_map<char, int64_t> char_to_id;
  std::vector<char> id_to_char;

  void build_from_table(const char* table, int size) {
    id_to_char.assign(table, table + size);
    for (int64_t i = 0; i < size; ++i) {
      char_to_id[id_to_char[static_cast<size_t>(i)]] = i;
    }
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

std::unique_ptr<GPT> g_model;
CharVocab g_vocab;

struct GenerateState {
  std::vector<int64_t> context;
  size_t prompt_len = 0;
  int64_t max_new_tokens = 0;
  int64_t produced = 0;
  float temperature = 0.8f;
  std::mt19937 rng;
  GPTKVCache cache;
  std::optional<Tensor> logits;
};

std::unique_ptr<GenerateState> g_gen;

Tensor make_batch_tensor(const std::vector<int64_t>& flat_ids, int64_t batch,
                         int64_t seq) {
  std::vector<float> host(static_cast<size_t>(batch * seq));
  for (int64_t i = 0; i < batch * seq; ++i) {
    host[static_cast<size_t>(i)] =
        static_cast<float>(flat_ids[static_cast<size_t>(i)]);
  }
  Tensor t({batch, seq});
  std::copy_n(host.begin(), host.size(), t.data<float>());
  return t;
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
                          float temperature, std::mt19937& rng) {
  NoGradGuard guard;
  std::vector<int64_t> context = vocab.encode(prompt);
  const size_t prompt_len = context.size();
  const int64_t vocab_size = model.config().vocab_size;

  GPTKVCache cache;
  Tensor prompt_ids =
      make_batch_tensor(context, 1, static_cast<int64_t>(context.size()));
  Tensor logits = model.prefill(prompt_ids, cache);

  std::vector<float> row(static_cast<size_t>(vocab_size));
  for (int64_t t = 0; t < max_new_tokens; ++t) {
    Tensor flat = logits.reshape({vocab_size}).contiguous();
    std::copy_n(flat.data<float>(), vocab_size, row.data());
    const int64_t next_id =
        sample_next_token(row.data(), vocab_size, temperature, rng);
    context.push_back(next_id);
    if (context.size() > static_cast<size_t>(model.config().max_seq_len)) {
      context.erase(context.begin());
      cache.reset();
      cache.layers.resize(static_cast<size_t>(model.config().num_layers));
      Tensor ctx_ids = make_batch_tensor(
          context, 1, static_cast<int64_t>(context.size()));
      logits = model.prefill(ctx_ids, cache);
    } else {
      logits = model.decode_step(next_id, cache);
    }
  }

  std::vector<int64_t> generated(context.begin() + static_cast<long>(prompt_len),
                                 context.end());
  return vocab.decode(generated);
}

void advance_generation(GPT& model, GenerateState& state) {
  const int64_t vocab_size = model.config().vocab_size;
  std::vector<float> row(static_cast<size_t>(vocab_size));
  Tensor flat = state.logits->reshape({vocab_size}).contiguous();
  std::copy_n(flat.data<float>(), vocab_size, row.data());
  const int64_t next_id =
      sample_next_token(row.data(), vocab_size, state.temperature, state.rng);
  state.context.push_back(next_id);
  if (state.context.size() > static_cast<size_t>(model.config().max_seq_len)) {
    state.context.erase(state.context.begin());
    state.cache.reset();
    state.cache.layers.resize(static_cast<size_t>(model.config().num_layers));
    Tensor ctx_ids = make_batch_tensor(
        state.context, 1, static_cast<int64_t>(state.context.size()));
    state.logits = model.prefill(ctx_ids, state.cache);
  } else {
    state.logits = model.decode_step(next_id, state.cache);
  }
  state.produced++;
}

}  // namespace

extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int tiramisu_init(const uint8_t* ckpt, size_t ckpt_len) {
  try {
    g_model = std::make_unique<GPT>(
        create_gpt_from_checkpoint(std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(ckpt), ckpt_len)));
    g_vocab = CharVocab{};
    g_vocab.build_from_table(kShakespeareIdToChar, kShakespeareVocabSize);
    return 0;
  } catch (...) {
    g_model.reset();
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
char* tiramisu_generate(const char* prompt, int max_chars, float temperature,
                        uint32_t seed) {
  if (!g_model) {
    return nullptr;
  }
  try {
    std::mt19937 rng(seed);
    const std::string text = generate_text(
        *g_model, g_vocab, prompt ? prompt : "", max_chars, temperature, rng);
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (!out) {
      return nullptr;
    }
    std::memcpy(out, text.data(), text.size());
    out[text.size()] = '\0';
    return out;
  } catch (...) {
    return nullptr;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int tiramisu_generate_begin(const char* prompt, int max_chars, float temperature,
                            uint32_t seed) {
  if (!g_model) {
    return -1;
  }
  try {
    g_gen = std::make_unique<GenerateState>();
    g_gen->context = g_vocab.encode(prompt ? prompt : "");
    g_gen->prompt_len = g_gen->context.size();
    g_gen->max_new_tokens = max_chars;
    g_gen->produced = 0;
    g_gen->temperature = temperature;
    g_gen->rng = std::mt19937(seed);
    Tensor prompt_ids = make_batch_tensor(
        g_gen->context, 1, static_cast<int64_t>(g_gen->context.size()));
    g_gen->logits = g_model->prefill(prompt_ids, g_gen->cache);
    return 0;
  } catch (...) {
    g_gen.reset();
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int tiramisu_generate_step() {
  if (!g_model || !g_gen || g_gen->produced >= g_gen->max_new_tokens) {
    return -1;
  }
  try {
    advance_generation(*g_model, *g_gen);
    const int64_t token_id = g_gen->context.back();
    return static_cast<unsigned char>(
        g_vocab.id_to_char[static_cast<size_t>(token_id)]);
  } catch (...) {
    return -1;
  }
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void tiramisu_generate_end() { g_gen.reset(); }

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void tiramisu_free(void* ptr) { std::free(ptr); }

}  // extern "C"

#ifndef __EMSCRIPTEN__
int main() { return 0; }
#endif
