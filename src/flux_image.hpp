// flux_image.hpp - pictures for flux-make: read any picture file (Windows' own codecs), write PNG, resize.
// 2026-10-02. The resize is Pillow's (Image.resize with LANCZOS: a = 3, the support widened by the scale when
// shrinking, weights normalised per output pixel, horizontal pass then vertical) in floating point with one rounding
// at the end - flux.py scales its reference pictures and its upscaled picture with Pillow, and the two programs
// should make the same picture. (Pillow works in fixed point per pass; a value can differ by one step.)
#pragma once
#include <windows.h>
#include <wincodec.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fluximg {

struct Rgb {            // 8-bit RGB, row-major, 3 bytes per pixel
    int w = 0, h = 0;
    std::vector<uint8_t> px;
};

inline std::wstring wide(const std::string & s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

struct Com { Com() { CoInitializeEx(nullptr, COINIT_MULTITHREADED); } ~Com() { CoUninitialize(); } };

inline Rgb load(const std::string & path) {
    Com com;
    IWICImagingFactory * fac = nullptr;
    IWICBitmapDecoder * dec = nullptr;
    IWICBitmapFrameDecode * fr = nullptr;
    IWICFormatConverter * cv = nullptr;
    Rgb r;
    UINT w = 0, h = 0;
    bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac))) &&
              SUCCEEDED(fac->CreateDecoderFromFilename(wide(path).c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec)) &&
              SUCCEEDED(dec->GetFrame(0, &fr)) && SUCCEEDED(fac->CreateFormatConverter(&cv)) &&
              SUCCEEDED(cv->Initialize(fr, GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, nullptr, 0.0, WICBitmapPaletteTypeCustom)) &&
              SUCCEEDED(cv->GetSize(&w, &h));
    if (ok) {
        r.w = (int) w; r.h = (int) h;
        r.px.resize((size_t) w * h * 3);
        ok = SUCCEEDED(cv->CopyPixels(nullptr, w * 3, (UINT) r.px.size(), r.px.data()));
        for (size_t i = 0; ok && i + 2 < r.px.size(); i += 3) std::swap(r.px[i], r.px[i + 2]);      // BGR -> RGB
    }
    if (cv) cv->Release();
    if (fr) fr->Release();
    if (dec) dec->Release();
    if (fac) fac->Release();
    if (!ok) throw std::runtime_error("cannot read the picture " + path);
    return r;
}

inline void save_png(const std::string & path, const Rgb & im) {
    Com com;
    IWICImagingFactory * fac = nullptr;
    IWICStream * st = nullptr;
    IWICBitmapEncoder * enc = nullptr;
    IWICBitmapFrameEncode * fr = nullptr;
    std::vector<uint8_t> bgr(im.px.size());
    for (size_t i = 0; i + 2 < im.px.size(); i += 3) { bgr[i] = im.px[i + 2]; bgr[i + 1] = im.px[i + 1]; bgr[i + 2] = im.px[i]; }
    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
    const bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac))) &&
                    SUCCEEDED(fac->CreateStream(&st)) && SUCCEEDED(st->InitializeFromFilename(wide(path).c_str(), GENERIC_WRITE)) &&
                    SUCCEEDED(fac->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(st, WICBitmapEncoderNoCache)) &&
                    SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr)) && SUCCEEDED(fr->SetSize((UINT) im.w, (UINT) im.h)) &&
                    SUCCEEDED(fr->SetPixelFormat(&fmt)) && IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR) &&
                    SUCCEEDED(fr->WritePixels((UINT) im.h, (UINT) im.w * 3, (UINT) bgr.size(), bgr.data())) && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
    if (fr) fr->Release();
    if (enc) enc->Release();
    if (st) st->Release();
    if (fac) fac->Release();
    if (!ok) throw std::runtime_error("cannot write the picture " + path);
}

// JPEG, by Windows' own encoder. quality 1..100 (zimage.py's default is 95). full_chroma: no colour subsampling
// (4:4:4) - for a picture with text on it, where 4:2:0 smears the edges of the letters; photographs take 4:2:0.
inline void save_jpeg(const std::string & path, const Rgb & im, int quality = 95, bool full_chroma = false) {
    Com com;
    IWICImagingFactory * fac = nullptr;
    IWICStream * st = nullptr;
    IWICBitmapEncoder * enc = nullptr;
    IWICBitmapFrameEncode * fr = nullptr;
    IPropertyBag2 * bag = nullptr;
    std::vector<uint8_t> bgr(im.px.size());
    for (size_t i = 0; i + 2 < im.px.size(); i += 3) { bgr[i] = im.px[i + 2]; bgr[i + 1] = im.px[i + 1]; bgr[i + 2] = im.px[i]; }
    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
    bool ok = SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac))) &&
              SUCCEEDED(fac->CreateStream(&st)) && SUCCEEDED(st->InitializeFromFilename(wide(path).c_str(), GENERIC_WRITE)) &&
              SUCCEEDED(fac->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &enc)) && SUCCEEDED(enc->Initialize(st, WICBitmapEncoderNoCache)) &&
              SUCCEEDED(enc->CreateNewFrame(&fr, &bag)) && bag;
    if (ok) {
        PROPBAG2 opt[2] = {};
        VARIANT val[2];
        VariantInit(&val[0]); VariantInit(&val[1]);
        opt[0].pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        val[0].vt = VT_R4; val[0].fltVal = (float) (quality < 1 ? 1 : quality > 100 ? 100 : quality) / 100.0f;
        opt[1].pstrName = const_cast<LPOLESTR>(L"JpegYCrCbSubsampling");
        val[1].vt = VT_UI1; val[1].bVal = (BYTE) (full_chroma ? WICJpegYCrCbSubsampling444 : WICJpegYCrCbSubsampling420);
        ok = SUCCEEDED(bag->Write(2, opt, val)) && SUCCEEDED(fr->Initialize(bag)) && SUCCEEDED(fr->SetSize((UINT) im.w, (UINT) im.h)) &&
             SUCCEEDED(fr->SetPixelFormat(&fmt)) && IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR) &&
             SUCCEEDED(fr->WritePixels((UINT) im.h, (UINT) im.w * 3, (UINT) bgr.size(), bgr.data())) && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
    }
    if (bag) bag->Release();
    if (fr) fr->Release();
    if (enc) enc->Release();
    if (st) st->Release();
    if (fac) fac->Release();
    if (!ok) throw std::runtime_error("cannot write the picture " + path);
}

// by the file name: .jpg / .jpeg -> JPEG (quality, full_chroma), anything else -> PNG
inline bool is_jpeg_name(const std::string & path) {
    const size_t d = path.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = path.substr(d);
    for (char & c : e) c = (char) tolower((unsigned char) c);
    return e == ".jpg" || e == ".jpeg";
}
inline void save(const std::string & path, const Rgb & im, int quality = 95, bool full_chroma = false) {
    if (is_jpeg_name(path)) save_jpeg(path, im, quality, full_chroma);
    else save_png(path, im);
}

inline double lanczos3(double x) {
    if (x < 0) x = -x;
    if (x >= 3.0) return 0.0;
    if (x < 1e-12) return 1.0;
    const double a = 3.14159265358979323846 * x;
    return 3.0 * sin(a) * sin(a / 3.0) / (a * a);
}

// one axis: n_in samples -> n_out, for every output its first input and its weights (Pillow's precompute_coeffs)
struct Taps { std::vector<int> first, count; std::vector<double> k; int width = 0; };
inline Taps taps_for(int n_in, int n_out) {
    Taps t;
    const double scale = (double) n_in / n_out, fs = scale < 1.0 ? 1.0 : scale, support = 3.0 * fs;
    t.width = (int) ceil(support) * 2 + 1;
    t.first.resize((size_t) n_out); t.count.resize((size_t) n_out); t.k.assign((size_t) n_out * t.width, 0.0);
    for (int o = 0; o < n_out; o++) {
        const double center = (o + 0.5) * scale;
        int lo = (int) (center - support + 0.5), hi = (int) (center + support + 0.5);
        if (lo < 0) lo = 0;
        if (hi > n_in) hi = n_in;
        double sum = 0;
        double * k = &t.k[(size_t) o * t.width];
        for (int x = lo; x < hi; x++) { k[x - lo] = lanczos3((x - center + 0.5) / fs); sum += k[x - lo]; }
        for (int x = 0; x < hi - lo; x++) k[x] /= sum;
        t.first[(size_t) o] = lo; t.count[(size_t) o] = hi - lo;
    }
    return t;
}

inline Rgb resize_lanczos(const Rgb & im, int W, int H) {
    if (im.w == W && im.h == H) return im;
    const Taps tx = taps_for(im.w, W), ty = taps_for(im.h, H);
    std::vector<float> mid((size_t) W * im.h * 3);                    // horizontal pass, kept unrounded
    for (int y = 0; y < im.h; y++)
        for (int x = 0; x < W; x++) {
            const double * k = &tx.k[(size_t) x * tx.width];
            double a[3] = { 0, 0, 0 };
            const uint8_t * s = &im.px[((size_t) y * im.w + tx.first[(size_t) x]) * 3];
            for (int i = 0; i < tx.count[(size_t) x]; i++, s += 3) { a[0] += k[i] * s[0]; a[1] += k[i] * s[1]; a[2] += k[i] * s[2]; }
            float * d = &mid[((size_t) y * W + x) * 3];
            for (int c = 0; c < 3; c++) {                                // Pillow rounds and clips to 8 bits between the passes
                double v = floor(a[c] + 0.5);
                d[c] = (float) (v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }
    Rgb out;
    out.w = W; out.h = H;
    out.px.resize((size_t) W * H * 3);
    for (int y = 0; y < H; y++) {
        const double * k = &ty.k[(size_t) y * ty.width];
        for (int x = 0; x < W; x++) {
            double a[3] = { 0, 0, 0 };
            for (int i = 0; i < ty.count[(size_t) y]; i++) {
                const float * s = &mid[((size_t) (ty.first[(size_t) y] + i) * W + x) * 3];
                a[0] += k[i] * s[0]; a[1] += k[i] * s[1]; a[2] += k[i] * s[2];
            }
            uint8_t * d = &out.px[((size_t) y * W + x) * 3];
            for (int c = 0; c < 3; c++) { const double v = floor(a[c] + 0.5); d[c] = (uint8_t) (v < 0 ? 0 : v > 255 ? 255 : v); }
        }
    }
    return out;
}

// scale so the picture covers w x h, then cut the middle out (what flux.py does with a reference picture)
inline Rgb cover(const Rgb & im, int w, int h) {
    const double k = (std::max)((double) w / im.w, (double) h / im.h);
    const int W = (std::max)(w, (int) nearbyint(im.w * k)), H = (std::max)(h, (int) nearbyint(im.h * k));
    const Rgb big = resize_lanczos(im, W, H);
    const int x0 = (W - w) / 2, y0 = (H - h) / 2;
    Rgb out;
    out.w = w; out.h = h;
    out.px.resize((size_t) w * h * 3);
    for (int y = 0; y < h; y++) memcpy(&out.px[(size_t) y * w * 3], &big.px[((size_t) (y + y0) * W + x0) * 3], (size_t) w * 3);
    return out;
}

}  // namespace fluximg
