// tests/mm3_beam_test.cu - maya-mm3's Reflection Beam kernels on one GPU against F64 host references (no model):
//   beam_attn - 6 query heads a KV group, a global layer (every position) and a sliding one (window 512, its cache a
//   ring of 512 + a chunk) for a prompt chunk's queries and a decode token's, one block vs split 1 .. 32 ways, the K / V
//   written by qk_norm_rope (RoPE on the sliding layer, none on the global one) in BF16 and in INT8 rotated;
//   beam_route (group-limited: the 4 of 8 groups by their two best, then the top 8) - the same experts and weights;
//   beam_qscale, beam_gate, beam_resid, beam_swiglu, softcap.
//     nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 -Isrc/mm3 tests/mm3_beam_test.cu src/mm3/mm3_kernels.cu \
//          -o build/mm3_beam_test && build/mm3_beam_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "mm3_kernels.cuh"

namespace k = mm3::k;

static float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
template <class T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

static int fails = 0;
static void check(bool ok, const char* what, double got = 0, double lim = 0) {
    std::printf("%s %-62s %.3g (limit %.3g)\n", ok ? "PASS" : "FAIL", what, got, lim);
    if (!ok) ++fails;
}

constexpr int HD = 128, HQ = 12, HKV = 2, G = HQ / HKV, WIN = 512, CHUNK = 64;

// the reference: query t's heads over keys lo .. p of the cache rows (row of position kp: kp % ring, ring 0: kp)
static std::vector<double> ref_attn(const std::vector<float>& q, const std::vector<uint16_t>& kc,
                                    const std::vector<uint16_t>& vc, const std::vector<int>& pos, int window, int ring) {
    const int T = (int) pos.size();
    std::vector<double> out((size_t) T * HQ * HD);
    const double scale = 1.0 / std::sqrt((double) HD);
    for (int t = 0; t < T; ++t) {
        const int p = pos[(size_t) t], lo = window > 0 ? std::max(0, p - window + 1) : 0;
        for (int h = 0; h < HQ; ++h) {
            const int g = h / G;
            std::vector<double> s;
            double m = -1e300;
            for (int kp = lo; kp <= p; ++kp) {
                const size_t row = (size_t) (ring > 0 ? kp % ring : kp);
                double d = 0;
                for (int i = 0; i < HD; ++i)
                    d += (double) q[((size_t) t * HQ + h) * HD + i] * bf2f(kc[(row * HKV + g) * HD + i]);
                s.push_back(d * scale);
                m = std::max(m, d * scale);
            }
            double l = 0;
            for (double& x : s) l += (x = std::exp(x - m));
            for (int i = 0; i < HD; ++i) {
                double a = 0;
                for (int kp = lo; kp <= p; ++kp) {
                    const size_t row = (size_t) (ring > 0 ? kp % ring : kp);
                    a += s[(size_t) (kp - lo)] * bf2f(vc[(row * HKV + g) * HD + i]);
                }
                out[((size_t) t * HQ + h) * HD + i] = a / l;
            }
        }
    }
    return out;
}

static double max_diff(const std::vector<float>& a, const std::vector<double>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - b[i]));
    return m;
}

// one layer's cache filled position by position (chunks of CHUNK through qk_norm_rope, as the engine writes it), then
// the last chunk's queries and a decode token's attention checked
static void attn_case(bool sliding, int kf, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int ctx = 1100, ring = sliding ? WIN + CHUNK : 0, rows = sliding ? ring : ctx;
    const int n_rot = sliding ? HD : 0;
    std::vector<float> inv(HD / 2);
    k::rope_inv_freq(10000.0f, HD, inv.data());
    float* inv_d = dev(inv);
    std::vector<float> ones(HD, 1.0f);
    float* w1 = dev(ones);
    // the BF16 cache (the reference reads it) and, for kf, the quantized one
    uint16_t *kc = nullptr, *vc = nullptr;
    cudaMalloc(&kc, (size_t) rows * HKV * HD * 2);
    cudaMalloc(&vc, (size_t) rows * HKV * HD * 2);
    k::KvQ qq;
    uint8_t* qb = nullptr;
    uint16_t *kq16 = nullptr, *vq16 = nullptr;
    if (kf != k::kKvBf16) {
        const size_t nb = k::kv_bytes(kf, kf, (size_t) rows * HKV);
        cudaMalloc(&qb, nb);
        k::kv_carve(qb, (size_t) rows * HKV, kf, kf, 3, kq16, vq16, qq);
    }
    float *q = nullptr, *kk = nullptr, *vv = nullptr, *out = nullptr, *part = nullptr;
    int* pos_d = nullptr;
    cudaMalloc(&q, (size_t) CHUNK * HQ * HD * 4);
    cudaMalloc(&kk, (size_t) CHUNK * HKV * HD * 4);
    cudaMalloc(&vv, (size_t) CHUNK * HKV * HD * 4);
    cudaMalloc(&out, (size_t) CHUNK * HQ * HD * 4);
    cudaMalloc(&part, k::beam_part_floats(CHUNK, HKV, G, k::kBeamMaxSplits) * 4);
    cudaMalloc(&pos_d, CHUNK * 4);
    std::vector<float> qh, kh, vh;
    std::vector<int> pos;
    const int last0 = ctx - CHUNK - 1;   // (the last chunk, then one decode token at ctx - 1)
    auto step = [&](int p0, int T) {
        qh.resize((size_t) T * HQ * HD);
        kh.resize((size_t) T * HKV * HD);
        vh.resize((size_t) T * HKV * HD);
        for (float& x : qh) x = nd(rng);
        for (float& x : kh) x = nd(rng);
        for (float& x : vh) x = nd(rng);
        pos.resize((size_t) T);
        std::iota(pos.begin(), pos.end(), p0);
        cudaMemcpy(q, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(kk, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(vv, vh.data(), vh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pos_d, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice);
        k::qk_norm_rope(q, kk, vv, w1, w1, pos_d, T, HQ, HKV, HD, n_rot, inv_d, 1e-6f, kc, vc, 0, ring);
        if (kf != k::kKvBf16) {   // (the same K / V again into the quantized cache; the queries normed twice: harmless)
            cudaMemcpy(kk, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice);
            k::qk_norm_rope(nullptr, kk, vv, w1, w1, pos_d, T, 0, HKV, HD, n_rot, inv_d, 1e-6f, nullptr, nullptr, 0, ring,
                            qq);
        }
        if (!sliding) k::beam_qscale(q, pos_d, T, HQ * HD, 0.1f, 2048.0f, 0);
    };
    for (int p0 = 0; p0 < last0; p0 += CHUNK) step(p0, std::min(CHUNK, last0 - p0));
    const char* lay = sliding ? "sliding" : "global";
    const char* fmt = kf == k::kKvBf16 ? "bf16" : k::kv_name(kf);
    char what[128];
    for (int phase = 0; phase < 2; ++phase) {
        const int T = phase == 0 ? CHUNK : 1, p0 = phase == 0 ? last0 : ctx - 1;
        step(p0, T);
        std::vector<float> qn = host(q, (size_t) T * HQ * HD);   // (normed, rotated / scaled: the kernel's queries)
        const auto ref = ref_attn(qn, host(kc, (size_t) rows * HKV * HD), host(vc, (size_t) rows * HKV * HD), pos,
                                  sliding ? WIN : 0, ring);
        const double lim = kf == k::kKvBf16 ? 2e-4 : kf == k::kKvI8 ? 3e-2 : 1.5e-1;
        for (int S : {1, 3, 8, 32}) {
            if (phase == 0 && S > 1) continue;
            const bool ok = k::beam_attn(q, kc, vc, kf == k::kKvBf16 ? k::KvQ{} : qq, pos_d, T, HQ, HKV,
                                         sliding ? WIN : 0, ring, 1.0f / std::sqrt((float) HD), out, 0, S,
                                         S > 1 ? part : nullptr);
            cudaDeviceSynchronize();
            const double d = ok ? max_diff(host(out, (size_t) T * HQ * HD), ref) : 1e9;
            std::snprintf(what, sizeof what, "attn %s %s T=%d S=%d vs F64", lay, fmt, T, S);
            check(ok && d < lim && cudaGetLastError() == cudaSuccess, what, d, lim);
        }
    }
    for (void* p : {(void*) kc, (void*) vc, (void*) qb, (void*) q, (void*) kk, (void*) vv, (void*) out, (void*) part,
                    (void*) pos_d, (void*) inv_d, (void*) w1})
        cudaFree(p);
}

static void route_case(std::mt19937& rng) {
    const int T = 37, E = 256, K = 8, NG = 8, TG = 4;
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> lg((size_t) T * E), bias(E);
    for (float& x : lg) x = nd(rng);
    for (float& x : bias) x = 0.05f * nd(rng);
    float *lg_d = dev(lg), *b_d = dev(bias), *w_d = nullptr;
    int* top_d = nullptr;
    cudaMalloc(&w_d, (size_t) T * K * 4);
    cudaMalloc(&top_d, (size_t) T * K * 4);
    k::beam_route(lg_d, b_d, T, E, K, NG, TG, top_d, w_d, 0);
    cudaDeviceSynchronize();
    const auto top = host(top_d, (size_t) T * K);
    const auto w = host(w_d, (size_t) T * K);
    int bad = 0;
    double wd = 0;
    for (int t = 0; t < T; ++t) {   // transformers' BeamTopkRouter, in F64
        std::vector<double> s(E), ch(E);
        for (int e = 0; e < E; ++e) {
            s[(size_t) e] = 1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + e]));
            ch[(size_t) e] = s[(size_t) e] + bias[(size_t) e];
        }
        const int gs = E / NG;
        std::vector<std::pair<double, int>> gscore;
        for (int gi = 0; gi < NG; ++gi) {
            std::vector<double> v(ch.begin() + gi * gs, ch.begin() + (gi + 1) * gs);
            std::partial_sort(v.begin(), v.begin() + 2, v.end(), std::greater<double>());
            gscore.push_back({v[0] + v[1], gi});
        }
        std::stable_sort(gscore.begin(), gscore.end(), [](auto a, auto b) { return a.first > b.first; });
        std::vector<char> keep(NG, 0);
        for (int r = 0; r < TG; ++r) keep[(size_t) gscore[(size_t) r].second] = 1;
        std::vector<std::pair<double, int>> cand;
        for (int e = 0; e < E; ++e)
            if (keep[(size_t) (e / gs)]) cand.push_back({ch[(size_t) e], e});
        std::stable_sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first > b.first; });
        double sum = 0;
        for (int i = 0; i < K; ++i) sum += s[(size_t) cand[(size_t) i].second];
        for (int i = 0; i < K; ++i) {
            if (top[(size_t) t * K + i] != cand[(size_t) i].second) ++bad;
            wd = std::max(wd, std::fabs(w[(size_t) t * K + i] - s[(size_t) cand[(size_t) i].second] / sum));
        }
    }
    check(bad == 0, "beam_route: the reference's experts, in order (mismatches)", bad, 0);
    check(wd < 1e-6, "beam_route: the weights vs F64", wd, 1e-6);
    // ng 1: every expert a candidate (llama.cpp's routing)
    k::beam_route(lg_d, b_d, T, E, K, 1, 1, top_d, w_d, 0);
    cudaDeviceSynchronize();
    const auto top1 = host(top_d, (size_t) T * K);
    int bad1 = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<std::pair<double, int>> cand;
        for (int e = 0; e < E; ++e)
            cand.push_back({1.0 / (1.0 + std::exp(-(double) lg[(size_t) t * E + e])) + bias[(size_t) e], e});
        std::stable_sort(cand.begin(), cand.end(), [](auto a, auto b) { return a.first > b.first; });
        for (int i = 0; i < K; ++i) bad1 += top1[(size_t) t * K + i] != cand[(size_t) i].second;
    }
    check(bad1 == 0, "beam_route ng 1: every expert (mismatches)", bad1, 0);
    for (void* p : {(void*) lg_d, (void*) b_d, (void*) w_d, (void*) top_d}) cudaFree(p);
}

static void elementwise(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int T = 5, n = T * HQ * HD;
    std::vector<float> a(n), b(n), gb(HQ);
    for (float& x : a) x = nd(rng);
    for (float& x : b) x = nd(rng);
    for (float& x : gb) x = nd(rng);
    std::vector<int> pos = {0, 7, 2047, 100000, 1048575};
    // qscale
    {
        float* x = dev(a);
        int* p = dev(pos);
        k::beam_qscale(x, p, T, HQ * HD, 0.1f, 2048.0f, 0);
        const auto y = host(x, n);
        double d = 0;
        for (int i = 0; i < n; ++i) {
            const double f = std::pow(1.0 + (pos[(size_t) (i / (HQ * HD))] + 1.0) / 2048.0, 0.1);
            d = std::max(d, std::fabs(y[(size_t) i] - a[(size_t) i] * f) / (1.0 + std::fabs(a[(size_t) i] * f)));
        }
        check(d < 1e-6, "beam_qscale vs F64 (relative)", d, 1e-6);
        cudaFree(x);
        cudaFree(p);
    }
    // gate
    {
        float *x = dev(a), *gl = dev(b), *g = dev(gb);
        k::beam_gate(x, gl, g, T, HQ, HD, 0);
        const auto y = host(x, n);
        double d = 0;
        for (int i = 0; i < n; ++i) {
            const int h = (i / HD) % HQ;
            const double r = a[(size_t) i] / (1.0 + std::exp(-((double) b[(size_t) i] + gb[(size_t) h])));
            d = std::max(d, std::fabs(y[(size_t) i] - r));
        }
        check(d < 1e-5, "beam_gate vs F64", d, 1e-5);
        cudaFree(x);
        cudaFree(gl);
        cudaFree(g);
    }
    // resid (layer 7's MLP: i = 17)
    {
        float *h = dev(a), *x = dev(b);
        const double i = 17;
        const float ca = (float) (1.0 / std::sqrt(i)), cb = (float) std::sqrt((i - 1) / i);
        k::beam_resid(h, x, ca, cb, (size_t) n, 0);
        const auto y = host(h, n);
        double d = 0;
        for (int j = 0; j < n; ++j)
            d = std::max(d, std::fabs(y[(size_t) j] - (b[(size_t) j] / std::sqrt(i) + a[(size_t) j] * std::sqrt((i - 1) / i))));
        check(d < 1e-5, "beam_resid vs F64", d, 1e-5);
        cudaFree(h);
        cudaFree(x);
    }
    // swiglu
    {
        float *g = dev(a), *u = dev(b), *hf = nullptr;
        cudaMalloc(&hf, (size_t) n * 4);
        k::beam_swiglu(g, u, (size_t) n, hf, nullptr, 0);
        const auto y = host(hf, n);
        double d = 0;
        for (int j = 0; j < n; ++j) {
            const double gv = a[(size_t) j];
            d = std::max(d, std::fabs(y[(size_t) j] - gv / (1.0 + std::exp(-gv)) * b[(size_t) j]));
        }
        check(d < 1e-4, "beam_swiglu vs F64", d, 1e-4);
        cudaFree(g);
        cudaFree(u);
        cudaFree(hf);
    }
    // softcap
    {
        std::vector<float> lg(n);
        for (int j = 0; j < n; ++j) lg[(size_t) j] = 20.0f * a[(size_t) j];
        float* x = dev(lg);
        k::softcap(x, (size_t) n, 30.0f, 0);
        const auto y = host(x, n);
        double d = 0, mx = 0;
        for (int j = 0; j < n; ++j) {
            d = std::max(d, std::fabs(y[(size_t) j] - 30.0 * std::tanh(lg[(size_t) j] / 30.0)));
            mx = std::max(mx, (double) std::fabs(y[(size_t) j]));
        }
        check(d < 1e-4 && mx <= 30.0, "softcap 30 vs F64 (and |x| <= 30)", d, 1e-4);
        cudaFree(x);
    }
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device\n");
        return 2;
    }
    std::mt19937 rng(42);
    for (bool sliding : {false, true})
        for (int kf : {k::kKvBf16, k::kKvI8, k::kKvFp8}) attn_case(sliding, kf, rng);
    route_case(rng);
    elementwise(rng);
    std::printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
