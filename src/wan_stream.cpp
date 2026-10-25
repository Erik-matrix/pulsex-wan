// wan-stream: the Wan2.2-TI2V-5B DiT on our own ggml stack (2026-10-09: the blocks made to fit our NPU
// instead, as with FLUX and Z-Image). Step 1: ONE block on ggml's CPU backend, checked against the torch
// reference (wan_block.WanBlock on the same dequantised GGUF weights) by wan_ggml_check.py. Step 2 puts the same graph
// on HTP0 with packed weights streamed through two slots, the flux-stream recipe.
//
// The block (wan_block.py, WanAttentionBlock), x [L x 3072] f32 on the host, in and out:
//   m = modulation + e0 (host): shift1, scale1 (+1), gate1, shift2, scale2 (+1), gate2
//   h = LN(x) * scale1 + shift1                                  LN without affine, eps 1e-6
//   q = RMS(Wq h + bq) * nq, k = RMS(Wk h + bk) * nk             RMS over all 3072 channels (not per head)
//   q, k: 3D RoPE, adjacent pairs, 22 / 21 / 21 pairs for frame / row / column, theta 10000 restarting per axis
//   x += (Wo FA(q, k, Wv h + bv) + bo) * gate1                   24 heads x 128, scale 1/sqrt(128)
//   h = LN3(x) (affine);  x += Wco FA(RMS(Wcq h + bcq) * ncq, RMS(Wck ctx + bck) * nck, Wcv ctx + bcv) + bco
//   h = LN(x) * scale2 + shift2;  x += (W2 gelu_tanh(W0 h + b0) + b2) * gate2
// RoPE: GGML_ROPE_TYPE_MSECT (this fork: contiguous sections, theta restart per section, adjacent pairs) computes
// p * 10000^(-2j/128) / ff[i]; Wan wants p * 10000^(-2j/k) with k = 44 / 42 / 42 per axis, so ff[i] = 10000^(2j (1/k - 1/128)).
//
//   wan-stream block <gguf> <dir> <block> [CPU]
//     dir/meta.txt "F H W Lc" (token grid F x H x W, text length Lc), x.f32 [L x 3072], mods.f32 [6 x 3072] (scales with
//     the +1), ctx.f32 [Lc x 3072] (text_embedding output)  ->  dir/y.f32 [L x 3072]
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <array>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }

template <typename T> static std::vector<T> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("kan inte lasa " + p);
    _fseeki64(f, 0, SEEK_END);
    const long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<T> v((size_t) n / sizeof(T));
    if (fread(v.data(), sizeof(T), v.size(), f) != v.size()) { fclose(f); throw std::runtime_error("kort lasning " + p); }
    fclose(f);
    return v;
}
static void write_bin(const std::string & p, const float * d, size_t n) {
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("kan inte skriva " + p);
    fwrite(d, sizeof(float), n, f);
    fclose(f);
}

// the per-block tensors, in this order
enum { SQ_W, SQ_B, SK_W, SK_B, SV_W, SV_B, SO_W, SO_B, SNQ, SNK, N3_W, N3_B, CQ_W, CQ_B, CK_W, CK_B, CV_W, CV_B, CO_W, CO_B,
       CNQ, CNK, F0_W, F0_B, F2_W, F2_B, F0A_W, F0A_B, F0B_W, F0B_B, F2A_W, F2B_W, N_BT };
static const char * BT[N_BT] = {
    "self_attn.q.weight", "self_attn.q.bias", "self_attn.k.weight", "self_attn.k.bias", "self_attn.v.weight", "self_attn.v.bias",
    "self_attn.o.weight", "self_attn.o.bias", "self_attn.norm_q.weight", "self_attn.norm_k.weight", "norm3.weight", "norm3.bias",
    "cross_attn.q.weight", "cross_attn.q.bias", "cross_attn.k.weight", "cross_attn.k.bias", "cross_attn.v.weight", "cross_attn.v.bias",
    "cross_attn.o.weight", "cross_attn.o.bias", "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
    "ffn.0.weight", "ffn.0.bias", "ffn.2.weight", "ffn.2.bias",
    "ffn.0a.weight", "ffn.0a.bias", "ffn.0b.weight", "ffn.0b.bias", "ffn.2a.weight", "ffn.2b.weight" };   // split FFN (make_ffn_split.py)

struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };   // flux-stream's layout
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

struct Engine {
    gguf_context * g = nullptr;
    ggml_context * meta = nullptr;         // every tensor's type and shape, no data
    FILE * gf = nullptr;
    size_t data_base = 0;
    ggml_backend_t npu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool packed = false;                   // HTP0: weights from the .hexpack through set_packed
    bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
    bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    FILE * pk = nullptr;
    FILE * pk2 = nullptr;                  // 10-11: the prefetch thread's own handle on the pack
    std::string pack_file;
    std::vector<PackEnt> ents;             // [ib * N_BT + k]
    ggml_backend_buffer_t slot[2] = {};
    ggml_context * wctx[2] = {};
    size_t slot_size = 0;
    std::vector<uint8_t> stage;
    double t_read = 0, t_copy = 0, t_run = 0;
    int n_nodes = 0, n_splits = 0;
    bool split_ffn = false;
    std::string tap_dir = ".";
    // 10-15 S2V: the resident audio injectors (block index -> its 10 tensors) and their NPU buffer
    enum { IQ_W, IQ_B, IK_W, IK_B, IV_W, IV_B, IO_W, IO_B, INQ, INK, N_INJ };
    std::map<int, std::array<ggml_tensor *, N_INJ>> inj;
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    int dim = 3072, heads = 24, ffn = 14336, n_blocks = 30;              // 10-15: set from the file in open()
    const int hd = 128;
    const float eps = 1e-6f, theta = 10000.0f;

    static ggml_type held_type(const ggml_tensor * m) { return ggml_n_dims(m) == 1 ? GGML_TYPE_F32 : m->type; }
    static std::string tname(int ib, int k) { return "blocks." + std::to_string(ib) + "." + BT[k]; }

    // the bytes of a file tensor as the engine holds it (F16 vectors -> F32)
    bool host_bytes(const ggml_tensor * m, std::vector<uint8_t> & out) {
        const int64_t id = gguf_find_tensor(g, m->name);
        const size_t nb = gguf_get_tensor_size(g, id);
        std::vector<uint8_t> raw(nb);
        _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(raw.data(), 1, nb, gf) != nb) return false;
        if (held_type(m) == m->type) { out.swap(raw); return true; }
        if (m->type != GGML_TYPE_F16) { fprintf(stderr, "[wan] %s: typ %s ovantad for en vektor\n", m->name, ggml_type_name(m->type)); return false; }
        out.resize((size_t) ggml_nelements(m) * sizeof(float));
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), (float *) out.data(), ggml_nelements(m));
        return true;
    }

    bool open(const std::string & path, const std::string & device) {
        gguf_init_params gp = { true, &meta };
        g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return false;
        gf = fopen(path.c_str(), "rb");
        data_base = gguf_get_data_offset(g);
        // 10-15: the model sizes from the file - the 5B (3072 / 24 / 14336 / 30) or the 14B S2V (5120 / 40 / 13824 / 40)
        if (ggml_tensor * q = ggml_get_tensor(meta, "blocks.0.self_attn.q.weight")) { dim = (int) q->ne[0]; heads = dim / hd; }
        if (ggml_tensor * f0 = ggml_get_tensor(meta, "blocks.0.ffn.0.weight")) ffn = (int) f0->ne[1];
        for (n_blocks = 0; ggml_get_tensor(meta, ("blocks." + std::to_string(n_blocks) + ".modulation").c_str()); n_blocks++) {}
        fprintf(stderr, "[wan] model: dim %d, %d heads x %d, ffn %d, %d blocks\n", dim, heads, hd, ffn, n_blocks);
        cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        if (!cpu) return false;
        ggml_backend_cpu_set_n_threads(cpu, env_int("WAN_CPU_THREADS", 6));   // 6 of 8 cores (heavy jobs leave two free)
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(cpu);
        const char * tag = "cpu";
        packed = device != "CPU";
        if (packed) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev) { fprintf(stderr, "[wan] ingen enhet %s\n", device.c_str()); return false; }
            npu = ggml_backend_dev_init(dev, nullptr);
            if (!npu) return false;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
            set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
            auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
            if (!get_packed || !set_packed || !tag_fn) { fprintf(stderr, "[wan] backenden saknar packad-API\n"); return false; }
            tag = tag_fn();
            bt = ggml_backend_get_default_buffer_type(npu);
        }
        const size_t al = ggml_backend_buft_get_alignment(bt);
        // 10-10: a SLIM file (flux_dit/slim_gguf.py) holds the metadata and the modulations; its blocks exist in the .hexpack only
        const bool slim = gguf_find_key(g, "pulsex.slim") >= 0;
        const std::string pack_path = path.substr(0, path.rfind('.')) + ".hexpack";
        if (slim) {
            if (!packed) { fprintf(stderr, "[wan] en smal fil kan bara koras pa NPU:n (CPU-laget laser raa tensorer ur den fullstandiga GGUF:en)\n"); return false; }
            struct _stat64 st;
            if (_stat64(path.c_str(), &st) != 0) return false;
            const int64_t kk = gguf_find_key(g, "pulsex.pack_key");
            const uint64_t pack_key = kk >= 0 ? gguf_get_val_u64(g, kk) : (uint64_t) st.st_mtime;
            const size_t n_want = (size_t) n_blocks * N_BT;
            PackHdr h{};
            FILE * f = fopen(pack_path.c_str(), "rb");
            bool ok = f && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "PXRP0001", 8) && !strncmp(h.tag, tag, sizeof(h.tag)) &&
                      h.gsize == (uint64_t) st.st_size && h.gmtime == pack_key && h.n == n_want;
            if (ok) { ents.resize(n_want); ok = fread(ents.data(), sizeof(PackEnt), n_want, f) == n_want; }
            if (f) fclose(f);
            split_ffn = ok && ents[F0A_W].size != 0;
            for (int ib = 0; ok && ib < n_blocks; ib++) {
                size_t lb = 0;
                for (int k = 0; k < N_BT; k++) {
                    const PackEnt & e = ents[(size_t) ib * N_BT + k];
                    ok = ok && tname(ib, k) == e.name;
                    lb += GGML_PAD(e.size, al);
                }
                slot_size = lb > slot_size ? lb : slot_size;
            }
            if (!ok) {
                fprintf(stderr, "[wan] %s ar en smal fil och packfilen bredvid saknas eller passar inte (backendens tagg: %s).\n"
                                "[wan] Den kan bara byggas om fran den fullstandiga GGUF:en (kor motorn en gang pa den, sedan slim_gguf.py).\n", path.c_str(), tag);
                return false;
            }
            fprintf(stderr, "[wan] smal fil: %zu poster ur %s\n", ents.size(), pack_path.c_str());
        }
        if (!slim) split_ffn = ggml_get_tensor(meta, "blocks.0.ffn.2a.weight") != nullptr;
        for (int ib = 0; !slim && ib < n_blocks; ib++) {                // the entries and the largest block
            size_t lb = 0;
            for (int k = 0; k < N_BT; k++) {
                const std::string nm = tname(ib, k);
                const bool want = split_ffn ? !(k == F0_W || k == F0_B || k == F2_W) : k < F0A_W;
                if (!want) {                                             // not in this layout: an empty entry
                    PackEnt e{};
                    strncpy(e.name, nm.c_str(), sizeof(e.name) - 1);
                    ents.push_back(e);
                    continue;
                }
                ggml_tensor * m = ggml_get_tensor(meta, nm.c_str());
                if (!m) { fprintf(stderr, "[wan] saknar %s\n", nm.c_str()); return false; }
                ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
                ggml_context * c = ggml_init(p);
                ggml_tensor * t = ggml_new_tensor(c, held_type(m), GGML_MAX_DIMS, m->ne);
                PackEnt e{};
                strncpy(e.name, nm.c_str(), sizeof(e.name) - 1);
                e.type = (uint32_t) t->type;
                for (int d = 0; d < 4; d++) e.ne[d] = t->ne[d];
                e.size = ggml_backend_buft_get_alloc_size(bt, t);
                lb += GGML_PAD(e.size, al);
                ggml_free(c);
                ents.push_back(e);
            }
            slot_size = lb > slot_size ? lb : slot_size;
        }
        if (packed && slim) {                                            // checked above, entry table read from it
            pk = fopen(pack_path.c_str(), "rb");
            if (!pk) return false;
        } else if (packed) {
            uint64_t off = GGML_PAD(sizeof(PackHdr) + ents.size() * sizeof(PackEnt), 4096);
            for (auto & e : ents) { e.off = off; off += GGML_PAD(e.size, 4096); }
            struct _stat64 st;
            if (_stat64(path.c_str(), &st) != 0) return false;
            PackHdr want{};
            memcpy(want.magic, "PXRP0001", 8);
            strncpy(want.tag, tag, sizeof(want.tag) - 1);
            want.gsize = (uint64_t) st.st_size; want.gmtime = (uint64_t) st.st_mtime;
            bool ok = false;
            if (FILE * f = fopen(pack_path.c_str(), "rb")) {
                PackHdr h{};
                std::vector<PackEnt> got(ents.size());
                ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, offsetof(PackHdr, n)) && h.n == ents.size() &&
                     fread(got.data(), sizeof(PackEnt), got.size(), f) == got.size() && !memcmp(got.data(), ents.data(), got.size() * sizeof(PackEnt));
                fclose(f);
            }
            if (!ok && !build_pack(pack_path, want, bt)) return false;
            pk = fopen(pack_path.c_str(), "rb");
            if (!pk) return false;
        }
        if (packed) pack_file = pack_path;
        for (int s = 0; s < 2; s++) {
            slot[s] = ggml_backend_buft_alloc_buffer(bt, slot_size);
            if (!slot[s]) return false;
            ggml_backend_buffer_set_usage(slot[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_t bk[2] = { npu, cpu };
        sched = packed ? ggml_backend_sched_new(bk, nullptr, 2, 4096, false, true) : ggml_backend_sched_new(&cpu, nullptr, 1, 4096, false, true);
        fprintf(stderr, "[wan] %s: plats %.1f MB x2, %zu tensorer, FFN %s\n", packed ? device.c_str() : "CPU", slot_size / 1048576.0,
                ents.size(), split_ffn ? "delad (2 x 7168)" : "hel");
        return sched != nullptr && load_injectors(bt);
    }

    // 10-15 S2V: s2v.inj.<b>.* -> one resident buffer on the engine's device (none in a 5B file: nothing to do)
    bool load_injectors(ggml_backend_buffer_type_t bt) {
        static const char * INJ_NAME[N_INJ] = { "q.weight", "q.bias", "k.weight", "k.bias", "v.weight", "v.bias", "o.weight", "o.bias",
                                          "norm_q.weight", "norm_k.weight" };
        std::vector<int> blocks;
        for (int b = 0; b < n_blocks; b++)
            if (ggml_get_tensor(meta, ("s2v.inj." + std::to_string(b) + ".q.weight").c_str())) blocks.push_back(b);
        if (blocks.empty()) return true;
        ggml_init_params p = { ggml_tensor_overhead() * (blocks.size() * N_INJ + 4), nullptr, true };
        ictx = ggml_init(p);
        for (int b : blocks) {
            std::array<ggml_tensor *, N_INJ> a{};
            for (int k = 0; k < N_INJ; k++) {
                const std::string nm = "s2v.inj." + std::to_string(b) + "." + INJ_NAME[k];
                ggml_tensor * m = ggml_get_tensor(meta, nm.c_str());
                if (!m) { fprintf(stderr, "[wan] saknar %s\n", nm.c_str()); return false; }
                a[k] = ggml_new_tensor(ictx, held_type(m), GGML_MAX_DIMS, m->ne);
                ggml_set_name(a[k], nm.c_str());
            }
            inj[b] = a;
        }
        ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, bt);
        if (!ibuf) { fprintf(stderr, "[wan] ljudinjektorerna fick ingen buffert\n"); return false; }
        ggml_backend_buffer_set_usage(ibuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> raw;
        for (auto & kv : inj)
            for (ggml_tensor * t : kv.second) {
                if (!host_bytes(ggml_get_tensor(meta, ggml_get_name(t)), raw)) return false;
                ggml_backend_tensor_set(t, raw.data(), 0, raw.size());       // the backend repacks on set
            }
        fprintf(stderr, "[wan] S2V: %zu ljudinjektorer bofasta (%.0f MB)\n", inj.size(), ggml_backend_buffer_get_size(ibuf) / 1048576.0);
        return true;
    }

    // 10-15 S2V: the audio injection after block b - rows [n0, L) of xs are the noisy tokens, nf frames x hw (frame-major);
    // actx [nf * na x dim] the frames' audio tokens; ada [nf x 2 x dim] (shift, 1 + scale) of this injector. One small
    // graph PER FRAME (all frames in one graph exceeded HTP0's batch limits at 480p).
    bool run_inject(int b, std::vector<float> & xs, int L, int n0, int nf, int na, const float * actx, const float * ada) {
        auto it = inj.find(b);
        if (it == inj.end()) return true;
        ggml_tensor ** I = it->second.data();
        const int N = L - n0, hw = N / nf;
        if (nf <= 0 || hw * nf != N) { fprintf(stderr, "[wan] inject: %d token passar inte %d bildrutor\n", N, nf); return false; }
        static const int dump = env_int("WAN_INJ_DUMP", 0);
        if (dump) write_bin(tap_dir + "\\inj_in.f32", xs.data() + (size_t) n0 * dim, (size_t) N * dim);
        for (int f = 0; f < nf; f++) {
            ggml_init_params p = { ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(128, false), nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, hw);
            ggml_tensor * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, na);
            ggml_tensor * sh = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim);
            ggml_tensor * sc = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim);
            for (ggml_tensor * i2 : { x, a, sh, sc }) ggml_set_input(i2);
            ggml_tensor * h = ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), sc), sh);
            ggml_tensor * q = ggml_reshape_3d(c, rms(c, lin(c, I[IQ_W], I[IQ_B], h), I[INQ]), hd, heads, hw);
            ggml_tensor * k = ggml_reshape_3d(c, rms(c, lin(c, I[IK_W], I[IK_B], a), I[INK]), hd, heads, na);
            ggml_tensor * v = ggml_reshape_3d(c, lin(c, I[IV_W], I[IV_B], a), hd, heads, na);
            ggml_tensor * qp = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));            // [hd, hw, heads]
            ggml_tensor * kp = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));            // [hd, na, heads]
            ggml_tensor * pr = ggml_soft_max_ext(c, ggml_mul_mat(c, kp, qp), nullptr, 1.0f / sqrtf((float) hd), 0.0f);   // [na, hw, heads]
            ggml_tensor * vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));            // [na, hd, heads]
            ggml_tensor * o3 = ggml_mul_mat(c, vt, pr);                                  // [hd, hw, heads]
            ggml_tensor * at = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, o3, 0, 2, 1, 3)), dim, hw);
            ggml_tensor * xo = ggml_add(c, x, lin(c, I[IO_W], I[IO_B], at));
            ggml_set_output(xo);
            ggml_cgraph * gr = ggml_new_graph_custom(c, 128, false);
            ggml_build_forward_expand(gr, xo);
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[wan] inject: alloc misslyckades\n"); return false; }
            float * xf = xs.data() + ((size_t) n0 + (size_t) f * hw) * dim;
            ggml_backend_tensor_set(x, xf, 0, (size_t) hw * dim * sizeof(float));
            ggml_backend_tensor_set(a, actx + (size_t) f * na * dim, 0, (size_t) na * dim * sizeof(float));
            ggml_backend_tensor_set(sh, ada + (size_t) f * 2 * dim, 0, (size_t) dim * sizeof(float));
            ggml_backend_tensor_set(sc, ada + ((size_t) f * 2 + 1) * dim, 0, (size_t) dim * sizeof(float));
            const double t0 = now_s();
            const bool ok = ggml_backend_sched_graph_compute(sched, gr) == GGML_STATUS_SUCCESS;
            t_run += now_s() - t0;
            if (ok) ggml_backend_tensor_get(xo, xf, 0, (size_t) hw * dim * sizeof(float));
            ggml_free(c);
            if (!ok) return false;
        }
        if (dump) write_bin(tap_dir + "\\inj_out.f32", xs.data() + (size_t) n0 * dim, (size_t) N * dim);
        return true;
    }

    // every block tensor in the backend's own layout, once: set (the backend repacks), get_packed, write
    bool build_pack(const std::string & path, PackHdr h, ggml_backend_buffer_type_t bt) {
        const double t0 = now_s();
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        uint64_t maxsz = 0;
        for (auto & e : ents) maxsz = e.size > maxsz ? e.size : maxsz;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, (size_t) maxsz);
        if (!b) { fclose(f); remove(tmp.c_str()); return false; }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        h.n = (uint32_t) ents.size();
        fwrite(&h, sizeof h, 1, f);
        fwrite(ents.data(), sizeof(PackEnt), ents.size(), f);
        std::vector<uint8_t> raw, pkd;
        bool good = true;
        for (size_t i = 0; i < ents.size() && good; i++) {
            if (!ents[i].size) continue;                                // not in this layout
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) ents[i].type, GGML_MAX_DIMS, ents[i].ne);
            ggml_tallocr ta = ggml_tallocr_new(b);
            good = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
            good = good && host_bytes(ggml_get_tensor(meta, ents[i].name), raw);
            if (good) ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
            pkd.assign((size_t) GGML_PAD(ents[i].size, 4096), 0);
            good = good && get_packed(t, pkd.data(), (size_t) ents[i].size);
            _fseeki64(f, (long long) ents[i].off, SEEK_SET);
            good = good && fwrite(pkd.data(), 1, pkd.size(), f) == pkd.size();
            ggml_free(c);
        }
        ggml_backend_buffer_free(b);
        good = fclose(f) == 0 && good;
        if (good) remove(path.c_str());
        if (!good || rename(tmp.c_str(), path.c_str()) != 0) { remove(tmp.c_str()); fprintf(stderr, "[wan] packfilen kunde inte byggas\n"); return false; }
        fprintf(stderr, "[wan] pack byggd: %zu tensorer, %.1f s -> %s\n", ents.size(), now_s() - t0, path.c_str());
        return true;
    }

    // block ib -> slot s, W[N_BT]
    // 10-11: block ib's packed bytes in ONE read (its entries lie in order in the pack), through pk2 - for the prefetch
    // thread while the previous block runs on the NPU
    bool read_block(int ib, std::vector<uint8_t> & buf, uint64_t & base) {
        if (!pk2 && !(pk2 = fopen(pack_file.c_str(), "rb"))) return false;
        uint64_t lo = UINT64_MAX, hi = 0;
        for (int k = 0; k < N_BT; k++) {
            const PackEnt & e = ents[(size_t) ib * N_BT + k];
            if (!e.size) continue;
            lo = e.off < lo ? e.off : lo;
            hi = e.off + e.size > hi ? e.off + e.size : hi;
        }
        if (hi <= lo) return false;
        buf.resize((size_t) (hi - lo));
        base = lo;
        _fseeki64(pk2, (long long) lo, SEEK_SET);
        return fread(buf.data(), 1, buf.size(), pk2) == buf.size();
    }

    // block ib -> slot s from bytes read_block already holds (packed mode only)
    bool upload_block(int ib, int s, ggml_tensor ** W, const std::vector<uint8_t> & buf, uint64_t base) {
        if (wctx[s]) { ggml_free(wctx[s]); wctx[s] = nullptr; }
        ggml_init_params p = { ggml_tensor_overhead() * (N_BT + 2), nullptr, true };
        ggml_context * c = wctx[s] = ggml_init(p);
        ggml_tallocr ta = ggml_tallocr_new(slot[s]);
        for (int k = 0; k < N_BT; k++) {
            const PackEnt & e = ents[(size_t) ib * N_BT + k];
            if (!e.size) { W[k] = nullptr; continue; }
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) e.type, GGML_MAX_DIMS, e.ne);
            ggml_set_name(t, e.name);
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) return false;
            if (e.off < base || e.off - base + e.size > buf.size()) return false;
            const double t1 = now_s();
            if (!set_packed(t, buf.data() + (e.off - base), (size_t) e.size)) return false;
            t_copy += now_s() - t1;
            W[k] = t;
        }
        return true;
    }

    bool load_block(int ib, int s, ggml_tensor ** W) {
        if (wctx[s]) { ggml_free(wctx[s]); wctx[s] = nullptr; }
        ggml_init_params p = { ggml_tensor_overhead() * (N_BT + 2), nullptr, true };
        ggml_context * c = wctx[s] = ggml_init(p);
        ggml_tallocr ta = ggml_tallocr_new(slot[s]);
        for (int k = 0; k < N_BT; k++) {
            const PackEnt & e = ents[(size_t) ib * N_BT + k];
            if (!e.size) { W[k] = nullptr; continue; }                     // not in this layout
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) e.type, GGML_MAX_DIMS, e.ne);
            ggml_set_name(t, e.name);
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) return false;
            const double t0 = now_s();
            if (packed) {
                stage.resize((size_t) e.size);
                _fseeki64(pk, (long long) e.off, SEEK_SET);
                if (fread(stage.data(), 1, (size_t) e.size, pk) != e.size) return false;
                const double t1 = now_s();
                if (!set_packed(t, stage.data(), (size_t) e.size)) return false;
                t_read += t1 - t0; t_copy += now_s() - t1;
            } else {
                if (!host_bytes(ggml_get_tensor(meta, e.name), stage)) return false;
                const double t1 = now_s();
                ggml_backend_tensor_set(t, stage.data(), 0, stage.size());
                t_read += t1 - t0; t_copy += now_s() - t1;
            }
            W[k] = t;
        }
        return true;
    }

    // the FFN's tanh GELU; WAN_GELU_F16=1 (op_params[4]) puts its sigmoid in fp16 (patch_ggml_gelu_f16sig.py), 0 = exact f32
    ggml_tensor * gelu(ggml_context * c, ggml_tensor * x) {
        static const int f16 = env_int("WAN_GELU_F16", 1);   // 10-09: 31.7 -> 19.5 ms per block, block 0 vs torch 1.298e-3 -> 1.327e-3
        ggml_tensor * g = ggml_gelu(c, x);
        if (f16) g->op_params[4] = 1;
        return g;
    }
    ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
        return ggml_add(c, ggml_mul_mat(c, w, x), b);
    }
    ggml_tensor * rms(ggml_context * c, ggml_tensor * x, ggml_tensor * w) {   // over the 3072 channels
        return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
    }
    ggml_tensor * rope(ggml_context * c, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * ff) {   // x [dim, L] -> [hd, heads, L]
        x = ggml_reshape_3d(c, x, hd, heads, x->ne[1]);
        int sections[4] = { 22, 21, 21, 0 };
        return ggml_rope_multi(c, x, ids, ff, hd, sections, GGML_ROPE_TYPE_MSECT, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    ggml_tensor * attn(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {   // [hd, heads, L*] each -> [dim, Lq]
        const int64_t Lq = q->ne[2];
        q = ggml_permute(c, q, 0, 2, 1, 3);
        k = ggml_permute(c, k, 0, 2, 1, 3);
        v = ggml_permute(c, v, 0, 2, 1, 3);
        // WAN_KV_F16=1: K/V as f16 into the flash attention (FLUX_KV_F16's idea) - half the VTCM per key row, so longer
        // KV blocks (Bc) for the same budget; 0 = f32 K/V views (10-09 tiling: Bc 1088, 5 blocks at 5070)
        static const int kv_f16 = env_int("WAN_KV_F16", 0);
        if (kv_f16) { k = ggml_cast(c, k, GGML_TYPE_F16); v = ggml_cast(c, v, GGML_TYPE_F16); }
        ggml_tensor * a = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        return ggml_reshape_2d(c, a, dim, Lq);
    }

    // one block on the CPU backend
    bool run_block(ggml_tensor ** W, std::vector<float> & xs, int L, const std::vector<float> & mods, const std::vector<float> & ctx,
                   int Lc, const std::vector<int32_t> & ids_v, const std::vector<float> & ffv,
                   const std::vector<float> * mods_b = nullptr, int n0 = 0) {
        if (!mods_b || n0 <= 0 || n0 >= L) n0 = 0;                   // image to video: rows [0, n0) take mods_b (t = 0)
        ggml_init_params p = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, L);
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, Lc);
        ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * L);
        ggml_tensor * ff = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd / 2);
        ggml_tensor * m[6];
        for (int k = 0; k < 6; k++) { m[k] = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim); ggml_set_input(m[k]); }
        ggml_tensor * mb[6] = {};
        if (n0) for (int k = 0; k < 6; k++) { mb[k] = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim); ggml_set_input(mb[k]); }
        // a * scale + shift, and a * gate - per row group when n0 > 0 (the first frame's tokens [0, n0) with mb)
        auto rows = [&](ggml_tensor * a, int64_t r0, int64_t nr) { return ggml_view_2d(c, a, a->ne[0], nr, a->nb[1], (size_t) r0 * a->nb[1]); };
        auto modsh = [&](ggml_tensor * a, int ks, int kh) {
            if (!n0) return ggml_add(c, ggml_mul(c, a, m[ks]), m[kh]);
            ggml_tensor * h0 = ggml_add(c, ggml_mul(c, rows(a, 0, n0), mb[ks]), mb[kh]);
            ggml_tensor * h1 = ggml_add(c, ggml_mul(c, rows(a, n0, L - n0), m[ks]), m[kh]);
            return ggml_concat(c, h0, h1, 1);
        };
        auto gate = [&](ggml_tensor * a, int kg) {
            if (!n0) return ggml_mul(c, a, m[kg]);
            return ggml_concat(c, ggml_mul(c, rows(a, 0, n0), mb[kg]), ggml_mul(c, rows(a, n0, L - n0), m[kg]), 1);
        };
        for (ggml_tensor * i : { x, t, ids, ff }) ggml_set_input(i);

        // self attention
        std::vector<std::pair<const char *, ggml_tensor *>> taps;
        auto T = [&](const char * n, ggml_tensor * v) { taps.push_back({ n, v }); return v; };
        ggml_tensor * x_in = x;
        // self attention
        ggml_tensor * h = T("h1", modsh(T("ln1", ggml_norm(c, x, eps)), 1, 0));
        ggml_tensor * qlin = T("qlin", lin(c, W[SQ_W], W[SQ_B], h));
        ggml_tensor * qrms = T("qrms", rms(c, qlin, W[SNQ]));
        ggml_tensor * q = T("qrope", rope(c, qrms, ids, ff));
        ggml_tensor * k = T("krope", rope(c, rms(c, lin(c, W[SK_W], W[SK_B], h), W[SNK]), ids, ff));
        ggml_tensor * v = T("v", ggml_reshape_3d(c, lin(c, W[SV_W], W[SV_B], h), hd, heads, L));
        ggml_tensor * a = T("attn", attn(c, q, k, v));
        x = T("x1", ggml_add(c, x, gate(T("o", lin(c, W[SO_W], W[SO_B], a)), 2)));
        // cross attention to the text
        h = T("h3", ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), W[N3_W]), W[N3_B]));
        q = T("cq", ggml_reshape_3d(c, rms(c, lin(c, W[CQ_W], W[CQ_B], h), W[CNQ]), hd, heads, L));
        k = T("ck", ggml_reshape_3d(c, rms(c, lin(c, W[CK_W], W[CK_B], t), W[CNK]), hd, heads, Lc));
        v = T("cv", ggml_reshape_3d(c, lin(c, W[CV_W], W[CV_B], t), hd, heads, Lc));
        x = T("x2", ggml_add(c, x, T("cattn", lin(c, W[CO_W], W[CO_B], attn(c, q, k, v)))));
        // feed forward
        h = T("h4", modsh(ggml_norm(c, x, eps), 4, 3));
        auto ffn_of = [&](ggml_tensor * hh, bool tap_it) -> ggml_tensor * {
            if (W[F2A_W]) {                                              // split FFN: two independent halves of the hidden dim
                ggml_tensor * f0a = lin(c, W[F0A_W], W[F0A_B], hh);
                ggml_tensor * fa = gelu(c, tap_it ? T("f0", f0a) : f0a);
                if (tap_it) fa = T("gelu", fa);
                ggml_tensor * fb = gelu(c, lin(c, W[F0B_W], W[F0B_B], hh));
                // (W2a fa + b2) is the MUL_MAT+ADD the backend fuses; W2b fb + that may fuse as well (env WAN_FFN_SUM=0: plain sum)
                static const int fsum = env_int("WAN_FFN_SUM", 1);
                return fsum ? ggml_add(c, ggml_mul_mat(c, W[F2B_W], fb), lin(c, W[F2A_W], W[F2_B], fa))
                            : ggml_add(c, ggml_add(c, ggml_mul_mat(c, W[F2A_W], fa), ggml_mul_mat(c, W[F2B_W], fb)), W[F2_B]);
            }
            ggml_tensor * f0 = lin(c, W[F0_W], W[F0_B], hh);
            ggml_tensor * f = gelu(c, tap_it ? T("f0", f0) : f0);
            if (tap_it) f = T("gelu", f);
            return lin(c, W[F2_W], W[F2_B], f);
        };
        // 10-11: the FFN in pieces along the tokens when one piece's GELU tensor would pass the 1 GiB HTP buffer limit
        static const int ffn_tokens = env_int("WAN_FFN_TOKENS", 12800);
        const int n_pieces = (int) ((L + ffn_tokens - 1) / ffn_tokens);
        ggml_tensor * f2 = nullptr;
        if (n_pieces <= 1) {
            f2 = ffn_of(h, true);
        } else {
            static bool told = false;
            if (!told) { fprintf(stderr, "[wan] FFN i %d bitar om hogst %d token (L %d)\n", n_pieces, ffn_tokens, (int) L); told = true; }
            const int64_t per = (L + n_pieces - 1) / n_pieces;
            for (int64_t s0 = 0; s0 < L; s0 += per) {
                const int64_t len = s0 + per <= L ? per : L - s0;
                ggml_tensor * hp = ggml_view_2d(c, h, h->ne[0], len, h->nb[1], (size_t) s0 * h->nb[1]);
                ggml_tensor * fp = ffn_of(hp, false);
                f2 = f2 ? ggml_concat(c, f2, fp, 1) : fp;
            }
        }
        ggml_tensor * xo = T("out", ggml_add(c, x, gate(T("f2", f2), 5)));
        static const char * tap = getenv("WAN_TAP");
        ggml_tensor * tap_t = nullptr;
        if (tap && *tap) {
            for (auto & tp : taps) if (!strcmp(tp.first, tap)) tap_t = ggml_cont(c, tp.second);
            if (!tap_t) { ggml_free(c); fprintf(stderr, "[wan] okand WAN_TAP %s\n", tap); return false; }
            ggml_set_output(tap_t);
        }
        ggml_set_output(xo);

        ggml_cgraph * gr = ggml_new_graph_custom(c, 512, false);
        ggml_build_forward_expand(gr, tap_t ? tap_t : xo);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[wan] alloc misslyckades\n"); return false; }
        // 10-09: x is reassigned through the graph - the INPUT is x_in. Writing x (the last x node) worked on the CPU only
        // because the allocator put that add in place over the input; on HTP0 the input stayed garbage (block rel 0.80).
        // An input a cut-off tap graph does not use has no buffer: skip it.
        auto set_in = [](ggml_tensor * tt, const void * d, size_t n) { if (tt->buffer) ggml_backend_tensor_set(tt, d, 0, n); };
        set_in(x_in, xs.data(), xs.size() * sizeof(float));
        set_in(t, ctx.data(), ctx.size() * sizeof(float));
        set_in(ids, ids_v.data(), ids_v.size() * sizeof(int32_t));
        set_in(ff, ffv.data(), ffv.size() * sizeof(float));
        for (int k2 = 0; k2 < 6; k2++) set_in(m[k2], mods.data() + (size_t) k2 * dim, (size_t) dim * sizeof(float));
        if (n0) for (int k2 = 0; k2 < 6; k2++) set_in(mb[k2], mods_b->data() + (size_t) k2 * dim, (size_t) dim * sizeof(float));
        const double t0 = now_s();
        const bool ok = ggml_backend_sched_graph_compute(sched, gr) == GGML_STATUS_SUCCESS;
        t_run += now_s() - t0;
        if (!n_nodes) {
            n_nodes = ggml_graph_n_nodes(gr);
            n_splits = ggml_backend_sched_get_n_splits(sched);
            fprintf(stderr, "[wan] block: %d noder, %d delningar, %.2f s\n", n_nodes, n_splits, now_s() - t0);
        }
        if (ok && tap_t) {
            std::vector<float> tv((size_t) ggml_nelements(tap_t));
            ggml_backend_tensor_get(tap_t, tv.data(), 0, tv.size() * sizeof(float));
            write_bin(tap_dir + "\\tap.f32", tv.data(), tv.size());
        } else if (ok) ggml_backend_tensor_get(xo, xs.data(), 0, xs.size() * sizeof(float));
        ggml_free(c);
        return ok;
    }

    // every way out goes through here (the destructor): the NPU session is closed whichever return or exception ends the run
    void close() {
        for (auto & w : wctx) if (w) { ggml_free(w); w = nullptr; }
        if (ibuf) { ggml_backend_buffer_free(ibuf); ibuf = nullptr; }    // 10-15 S2V
        if (ictx) { ggml_free(ictx); ictx = nullptr; }
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
        for (auto & b : slot) if (b) { ggml_backend_buffer_free(b); b = nullptr; }
        if (npu) { ggml_backend_free(npu); npu = nullptr; fprintf(stderr, "[wan] NPU-sessionen stangd\n"); }
        if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
        if (pk) { fclose(pk); pk = nullptr; }
        if (gf) { fclose(gf); gf = nullptr; }
        if (g) { gguf_free(g); g = nullptr; }
        if (meta) { ggml_free(meta); meta = nullptr; }
    }
    ~Engine() { close(); }
};

// RoPE positions [4 x L] axis-major (frame, row, column, 0) and the per-pair frequency factors (see the header)
static void rope_inputs(int F, int H, int Wd, std::vector<int32_t> & ids, std::vector<float> & ffv) {
    const int L = F * H * Wd;
    ids.assign((size_t) 4 * L, 0);
    for (int f = 0; f < F; f++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < Wd; x++) {
                const int s = (f * H + y) * Wd + x;
                ids[s] = f; ids[(size_t) L + s] = y; ids[(size_t) 2 * L + s] = x;
            }
    const int sec[3] = { 22, 21, 21 };
    const double base = 10000.0, hd = 128.0;
    ffv.assign(64, 1.0f);
    int i = 0;
    for (int a = 0; a < 3; a++) {
        const double kdim = 2.0 * sec[a];                      // 44, 42, 42 channels
        for (int j = 0; j < sec[a]; j++, i++) ffv[i] = (float) pow(base, 2.0 * j * (1.0 / kdim - 1.0 / hd));
    }
    if (env_int("WAN_FF_ONE", 0)) ffv.assign(64, 1.0f);       // the check's RoPE control: ggml's own spacing (wrong for Wan)
}

// the six modulation vectors of block ib for one step: modulation (F16 [6 x 3072] in the file) + e0, the two scales +1
static std::vector<float> block_mods(gguf_context * g, FILE * gf, size_t base, int ib, const std::vector<float> & e0, int dim) {
    const std::string nm = "blocks." + std::to_string(ib) + ".modulation";
    const int64_t id = gguf_find_tensor(g, nm.c_str());
    const ggml_type ty = id < 0 ? GGML_TYPE_COUNT : gguf_get_tensor_type(g, id);
    if (ty != GGML_TYPE_F16 && ty != GGML_TYPE_F32) throw std::runtime_error("saknar/fel typ " + nm);   // 10-15: S2V's are F32
    std::vector<float> m((size_t) 6 * dim);
    _fseeki64(gf, (long long) (base + gguf_get_tensor_offset(g, id)), SEEK_SET);
    if (ty == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(m.size());
        if (fread(h.data(), sizeof(ggml_fp16_t), h.size(), gf) != h.size()) throw std::runtime_error("kort lasning " + nm);
        ggml_fp16_to_fp32_row(h.data(), m.data(), (int64_t) m.size());
    } else if (fread(m.data(), sizeof(float), m.size(), gf) != m.size()) {
        throw std::runtime_error("kort lasning " + nm);
    }
    for (size_t i = 0; i < m.size(); i++) m[i] += e0[i];
    for (int k : { 1, 4 }) for (int i = 0; i < dim; i++) m[(size_t) k * dim + i] += 1.0f;
    return m;
}

// the resident mode (see the header of patch_wan_stream_serve.py)
static int serve(const std::string & gguf_path, const std::string & dir) {
    const char * dv = getenv("WAN_DEVICE");
    const std::string device = dv && *dv ? dv : "CPU";
    Engine E;
    E.tap_dir = dir;
    if (!E.open(gguf_path, device)) { printf("err open\n"); fflush(stdout); return 1; }
    printf("ok ready %s\n", device.c_str());
    fflush(stdout);
    std::map<std::string, std::vector<float>> texts;
    int gF = -1, gH = -1, gW = -1;
    std::vector<int32_t> ids;
    std::vector<float> ffv;
    ggml_tensor * W[N_BT] = {};
    char line[512];
    while (fgets(line, sizeof line, stdin)) {
        char cmd[32] = {}, name[128] = {};
        int F = 0, H = 0, Wd = 0, n0s = 0;
        if (sscanf(line, "%31s", cmd) != 1) continue;
        try {
            if (!strcmp(cmd, "quit")) break;
            if (!strcmp(cmd, "text") && sscanf(line, "%*s %127s", name) == 1) {
                std::vector<float> t = read_bin<float>(dir + "\\" + name + ".f32");
                if (t.empty() || t.size() % E.dim) { printf("err text %s size %zu\n", name, t.size()); fflush(stdout); continue; }
                const size_t lc = t.size() / E.dim;
                texts[name] = std::move(t);
                printf("ok text %s %zu\n", name, lc);
            } else if (!strcmp(cmd, "pass") && sscanf(line, "%*s %d %d %d %127s", &F, &H, &Wd, name) == 4) {
                int n0 = 0;
                sscanf(line, "%*s %*d %*d %*d %*s %d", &n0);
                auto it = texts.find(name);
                if (it == texts.end()) { printf("err pass: no text %s\n", name); fflush(stdout); continue; }
                const int L = F * H * Wd, Lc = (int) (it->second.size() / E.dim);
                std::vector<float> xs = read_bin<float>(dir + "\\x.f32"), e0 = read_bin<float>(dir + "\\e0.f32");
                if (xs.size() != (size_t) L * E.dim || e0.size() != (size_t) 6 * E.dim) {
                    printf("err pass: x %zu (want %d), e0 %zu\n", xs.size(), L * E.dim, e0.size()); fflush(stdout); continue;
                }
                if (F != gF || H != gH || Wd != gW) { rope_inputs(F, H, Wd, ids, ffv); gF = F; gH = H; gW = Wd; }
                std::vector<float> e0b;
                if (n0 > 0) {
                    e0b = read_bin<float>(dir + "\\e0b.f32");
                    if (e0b.size() != (size_t) 6 * E.dim) { printf("err pass: e0b %zu\n", e0b.size()); fflush(stdout); continue; }
                }
                const double tp = now_s(), r0 = E.t_read, c0 = E.t_copy, u0 = E.t_run;
                bool ok = true;
                static const bool prefetch = E.packed && env_int("WAN_PREFETCH", 1) != 0;
                static bool told = false;
                if (!told) { fprintf(stderr, "[wan] prefetch %s\n", prefetch ? "on (block b+1 read beside block b)" : "off (serial)"); told = true; }
                static std::vector<uint8_t> hb[2];
                uint64_t hbase[2] = {};
                if (prefetch) {                                          // the pass's first block: read now
                    const double tr = now_s();
                    ok = E.read_block(0, hb[0], hbase[0]);
                    E.t_read += now_s() - tr;
                }
                for (int b = 0; ok && b < E.n_blocks; b++) {
                    std::thread th;
                    bool next_ok = true;
                    if (prefetch && b + 1 < E.n_blocks)
                        th = std::thread([&, b] { next_ok = E.read_block(b + 1, hb[(b + 1) & 1], hbase[(b + 1) & 1]); });
                    ok = prefetch ? E.upload_block(b, b & 1, W, hb[b & 1], hbase[b & 1]) : E.load_block(b, b & 1, W);
                    std::vector<float> mbv;
                    if (n0 > 0) mbv = block_mods(E.g, E.gf, E.data_base, b, e0b, E.dim);
                    ok = ok && E.run_block(W, xs, L, block_mods(E.g, E.gf, E.data_base, b, e0, E.dim), it->second, Lc, ids, ffv,
                                           n0 > 0 ? &mbv : nullptr, n0);
                    if (th.joinable()) {                                 // the wait, if the read was slower than the block
                        const double tw = now_s();
                        th.join();
                        E.t_read += now_s() - tw;
                        ok = ok && next_ok;
                    }
                }
                if (!ok) { printf("err pass failed\n"); fflush(stdout); continue; }
                write_bin(dir + "\\y.f32", xs.data(), xs.size());
                printf("ok pass %.3f %.3f %.3f %.3f\n", now_s() - tp, E.t_read - r0, E.t_copy - c0, E.t_run - u0);
            } else if (!strcmp(cmd, "spass") && sscanf(line, "%*s %d %d %d %d %127s", &F, &H, &Wd, &n0s, name) == 5) {
                // 10-15 S2V: spass L n0 nf na <text> (F = L, H = n0, Wd = nf here; n0s = na)
                const int L = F, n0 = H, nf = Wd, na = n0s;
                auto it = texts.find(name);
                if (it == texts.end()) { printf("err spass: no text %s\n", name); fflush(stdout); continue; }
                const int Lc = (int) (it->second.size() / E.dim);
                std::vector<float> xs = read_bin<float>(dir + "\\x.f32"), e0 = read_bin<float>(dir + "\\e0.f32"),
                                   e0b = read_bin<float>(dir + "\\e0b.f32"), actx = read_bin<float>(dir + "\\actx.f32"),
                                   ada = read_bin<float>(dir + "\\ada.f32");
                std::vector<int32_t> sids = read_bin<int32_t>(dir + "\\ids.i32");
                const size_t ni = E.inj.size();
                if (xs.size() != (size_t) L * E.dim || e0.size() != (size_t) 6 * E.dim || e0b.size() != (size_t) 6 * E.dim ||
                    sids.size() != (size_t) 4 * L || actx.size() != (size_t) nf * na * E.dim || ada.size() != ni * nf * 2 * E.dim) {
                    printf("err spass: sizes x %zu e0 %zu e0b %zu ids %zu actx %zu ada %zu (L %d n0 %d nf %d na %d inj %zu)\n", xs.size(),
                           e0.size(), e0b.size(), sids.size(), actx.size(), ada.size(), L, n0, nf, na, ni);
                    fflush(stdout); continue;
                }
                if (ffv.empty()) { std::vector<int32_t> tmp; rope_inputs(1, 1, 1, tmp, ffv); }   // the frequency factors only
                const double tp = now_s(), r0 = E.t_read, c0 = E.t_copy, u0 = E.t_run;
                bool ok = true;
                int ii = 0;
                for (int b = 0; ok && b < E.n_blocks; b++) {
                    ok = E.load_block(b, b & 1, W);
                    std::vector<float> mbv = block_mods(E.g, E.gf, E.data_base, b, e0b, E.dim);
                    ok = ok && E.run_block(W, xs, L, block_mods(E.g, E.gf, E.data_base, b, e0, E.dim), it->second, Lc, sids, ffv, &mbv, n0);
                    if (ok && E.inj.count(b)) { ok = E.run_inject(b, xs, L, n0, nf, na, actx.data(), ada.data() + (size_t) ii * nf * 2 * E.dim); ii++; }
                }
                if (!ok) { printf("err spass failed\n"); fflush(stdout); continue; }
                write_bin(dir + "\\y.f32", xs.data(), xs.size());
                printf("ok spass %.3f %.3f %.3f %.3f\n", now_s() - tp, E.t_read - r0, E.t_copy - c0, E.t_run - u0);
            } else {
                printf("err unknown: %s", line);
            }
        } catch (const std::exception & e) {
            printf("err %s\n", e.what());
        }
        fflush(stdout);
    }
    printf("ok quit\n");
    fflush(stdout);
    return 0;                                   // E's destructor closes the NPU session
}

static int run(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "serve" && argc >= 4) return serve(argv[2], argv[3]);
    if (argc < 4 || (mode != "block" && mode != "pass") || (mode == "block" && argc < 5)) {
        fprintf(stderr, "usage: wan-stream block <gguf> <dir> <block>        (dir: meta.txt x.f32 mods.f32 ctx.f32 -> y.f32)\n"
                        "       wan-stream pass  <gguf> <dir> [first last]   (dir: meta.txt x.f32 e0.f32 ctx.f32 -> y.f32)\n"
                        "       wan-stream serve <gguf> <dir>                (stdin: text <name> | pass F H W <name> | quit)\n");
        return 1;
    }
    const std::string gguf_path = argv[2], dir = argv[3];
    const int ib = mode == "block" ? atoi(argv[4]) : 0;
    int F = 0, H = 0, Wd = 0, Lc = 0;
    { FILE * f = fopen((dir + "\\meta.txt").c_str(), "r"); if (!f || fscanf(f, "%d %d %d %d", &F, &H, &Wd, &Lc) != 4) { fprintf(stderr, "[wan] meta.txt\n"); return 1; } fclose(f); }
    const int L = F * H * Wd;
    std::vector<float> xs = read_bin<float>(dir + "\\x.f32"), ctx = read_bin<float>(dir + "\\ctx.f32");
    std::vector<float> mods = read_bin<float>(dir + (mode == "block" ? "\\mods.f32" : "\\e0.f32"));   // pass: e0 [6 x 3072]
    std::vector<int32_t> ids;
    std::vector<float> ffv;
    rope_inputs(F, H, Wd, ids, ffv);

    const char * dv = getenv("WAN_DEVICE");
    const std::string device = dv && *dv ? dv : "CPU";
    Engine E;
    E.tap_dir = dir;
    if (!E.open(gguf_path, device)) { fprintf(stderr, "[wan] open misslyckades\n"); return 1; }
    if (xs.size() != (size_t) L * E.dim || mods.size() != (size_t) 6 * E.dim || ctx.size() != (size_t) Lc * E.dim) {   // 10-15
        fprintf(stderr, "[wan] fel storlek: x %zu (%d), mods %zu, ctx %zu (%d)\n", xs.size(), L * E.dim, mods.size(), ctx.size(), Lc * E.dim);
        return 1;
    }
    ggml_tensor * W[N_BT] = {};
    bool ok = true;
    if (mode == "block") {
        const double t0 = now_s();
        ok = E.load_block(ib, 0, W);
        fprintf(stderr, "[wan] block %d laddat %.2f s, L %d (%dx%dx%d), text %d\n", ib, now_s() - t0, L, F, H, Wd, Lc);
        const int n0b = env_int("WAN_N0", 0);
        std::vector<float> mods_b;
        if (n0b > 0) mods_b = read_bin<float>(dir + "\\mods_b.f32");
        ok = ok && E.run_block(W, xs, L, mods, ctx, Lc, ids, ffv, n0b > 0 ? &mods_b : nullptr, n0b);
    } else {
        const int first = argc > 5 ? atoi(argv[4]) : 0, last = argc > 5 ? atoi(argv[5]) : 29;
        const double tp = now_s();
        for (int b = first; ok && b <= last; b++) {
            ok = E.load_block(b, b & 1, W);
            ok = ok && E.run_block(W, xs, L, block_mods(E.g, E.gf, E.data_base, b, mods, E.dim), ctx, Lc, ids, ffv);
        }
        fprintf(stderr, "[wan] pass block %d..%d pa %s: %.2f s (lasning %.2f, kopiering %.2f, rakning %.2f), L %d, text %d\n", first, last,
                device.c_str(), now_s() - tp, E.t_read, E.t_copy, E.t_run, L, Lc);
    }
    if (ok) write_bin(dir + "\\y.f32", xs.data(), xs.size());
    E.close();
    printf("[wan] %s\n", ok ? "klar -> y.f32" : "MISSLYCKADES");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    HANDLE me = GetCurrentProcess();                  // a heavy job: below-normal, cores 2-7 (0-1 stay free for the desktop)
    SetPriorityClass(me, BELOW_NORMAL_PRIORITY_CLASS);
    SetProcessAffinityMask(me, 0xFC);
    // ggml-hexagon's flash softmax floors P at 2^-15; P * 2^3 measured on Wan's attention 1.62e-3 vs 3.70e-3 (10-09).
    // _putenv_s, not SetEnvironmentVariable: ggml reads it with getenv (the CRT's copy). -1 = the old exp.
    if (!getenv("GGML_HEXAGON_FA_SCALE_EXP2")) _putenv_s("GGML_HEXAGON_FA_SCALE_EXP2", "3");
    // the same exp2 in 14 ops instead of 18 (patch_ggml_fa_trunc_exp2.py + _trunc_qf.py): self-attention 101.0 -> 92.9 ms,
    // torch 1.563e-3 vs 1.623e-3 (10-10). 1 = 15 ops (f rounded to fp16, 96.4 ms), 0 = ggml's recipe.
    // 10-12: 4 = the same exp2 in FIXED POINT (htp_patches/patch_fa_exp2_fixp_1012.py): trq spends 10 of 14 ops on the
    // single shift unit; here 4, the rest on the multipliers / ALU (asm 81 -> 46 packets per 8 vectors). 720p block
    // self-attention 2094 -> 1711 ms; block 0 vs torch 1.3047e-3 -> 1.3014e-3 (resident), 1.3407e-3 -> 1.3435e-3 (long).
    if (!getenv("GGML_HEXAGON_FA_TRUNC_EXP2")) _putenv_s("GGML_HEXAGON_FA_TRUNC_EXP2", "4");
    // MUL+ADD -> FMA (patch_ggml_fma.py): gate+residual and the norm modulation in one pass each. 0 = separate ops.
    if (!getenv("GGML_HEXAGON_FUSE_FMA")) _putenv_s("GGML_HEXAGON_FUSE_FMA", "1");
    // the softmax row sums on the idle HMX (htp_patches/patch_fa_hmx_rowsum_1012.py): only the main pipelined loop, i.e.
    // the long sequences' 768 x 512 without resident K/V. 720p block self-attention 2757.8 -> 2309.2 ms, block vs before
    // rel L2 3.5e-5; 5070 tokens at 768 x 512 101.6 -> 88.5 ms, torch 1.6891e-3 both (10-12). 0 = the HVX sums.
    if (!getenv("GGML_HEXAGON_FA_HMX_ROWSUM")) _putenv_s("GGML_HEXAGON_FA_HMX_ROWSUM", "1");
    // each head's prepared K kept in a DDR cache after its first Q tile (htp_patches/patch_fa_kcache_1012.py + _b.._e):
    // the other Q tiles DMA the tiles in instead of the vscatter K-prep. Same main loop only. 720p block self-attention
    // 2309 -> 2100 ms, output bit-identical (10-12). 7.1 MB DSP heap at 27280 tokens. 0 = K-prep per Q tile.
    if (!getenv("GGML_HEXAGON_FA_KCACHE")) _putenv_s("GGML_HEXAGON_FA_KCACHE", "1");
    // the self-attention's tiling, set by the harness as zimage_stream.cpp does (the backend's cost model must serve every
    // shape): Br 256 x Bc 1280 (4 KV blocks) measured 103.2 ms vs the model's Bc 1088 (5 blocks) 105.7 ms (10-09). The cross
    // attention (512 keys) refuses Bc 1280 and keeps the cost model's tiling. WAN_FA_TILING=0 leaves both to the model.
    if (env_int("WAN_FA_TILING", 1) && !getenv("GGML_HEXAGON_FA_BR") && !getenv("GGML_HEXAGON_FA_BC")) {
        _putenv_s("GGML_HEXAGON_FA_BR", "256");
        _putenv_s("GGML_HEXAGON_FA_BC", "1280");
    }
    // MUL_MAT+ADD+GELU on HMX (patch_ggml_mm_gelu.py) is NOT set here: measured 10-09, ffn.0 + GELU 133.7 ms fused vs
    // 43.5 + 30.5 separate - the GELU in the output transfer runs on the few HVX threads between HMX chunks and stalls the
    // pipeline (the activation is compute-bound, as SiLU was). GGML_HEXAGON_FUSE_MM_GELU=1 turns it on for an experiment.
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[wan] avbrott: %s\n", e.what());
        return 1;
    }
}
