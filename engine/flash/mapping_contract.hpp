#pragma once
#ifndef FLASH_GGUF_DIRECT_DECAY
#define FLASH_GGUF_DIRECT_DECAY 0
#endif
namespace chlorine_flash {
inline constexpr bool direct_decay_requested=FLASH_GGUF_DIRECT_DECAY!=0;
inline const char* decay_semantics(bool direct) {
  return direct ? "gguf-negative-coefficient-v1" : "hgn-log-exp-v1";
}
}
