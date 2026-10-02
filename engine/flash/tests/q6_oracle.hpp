// GGML authors, MIT. Standalone adaptation; see ../quant/SOURCES.json.
#pragma once
#include <cassert>
constexpr int QK_K_Q6=256;
struct block_q6_K { uint8_t ql[128], qh[64]; int8_t scales[16]; uint16_t d; };
static_assert(sizeof(block_q6_K)==210);
void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k) {
    assert(k % QK_K_Q6 == 0);
    const int64_t nb = k / QK_K_Q6;

    for (int i = 0; i < nb; i++) {
        const float d = gguf::h2f(x[i].d);

        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;

        for (int n = 0; n < QK_K_Q6; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
