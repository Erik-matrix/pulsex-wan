// wan_gui.cpp - PulseX Wan: the window for Wan 2.2 (text or a picture -> a film), in PulseX Image's look
// (flux_dit/flux_gui.cpp: Dear ImGui + DX11, our own title bar, Swedish / English).
//
// 2026-10-11: a window for Wan, like pulsex-image.exe. The window does no model work itself: it runs
// wan_pipeline.py (Python, in the python folder beside bin) with --no-png --preview and reads its lines. The engines open and
// close the NPU themselves, one at a time, exactly as on the command line; the window cannot stop a film halfway (an
// engine killed mid-call can leave the NPU pinned) - closing it waits for the film, then goes away.
//   * films land in Videos\PulseX Wan (the mp4 only, never overwritten); the work (latent, the newest preview, the
//     list of films) lives in <app>\data (PULSEX_WAN_DATA) - the same files each time, nothing piles up
//   * the choices are what exists: the big VAE needs its NPU contexts, "fast" the taew context (49 frames on the NPU,
//     121 on the CPU)
//   * test hooks: --shot file.png (draw, save the window, exit), --demo-run "prompt" (press the button),
//     --demo-cpu (with --demo-run: a 1-block, 1-step, 5-frame film on the CPU - the whole path without the NPU),
//     --demo-image file, --lang sv|en, --window W H
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#include <dwmapi.h>
#include <tlhelp32.h>
#pragma comment(lib,"shell32.lib")
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"dwmapi.lib")
#pragma comment(lib,"comdlg32.lib")
#include "flux_image.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define APP_NAME    "PulseX Wan"
#define APP_NAME_W  L"PulseX Wan"
#define APP_CLASS_W L"pulsex_wan"
#define APP_ICON    "pulsex-wan.ico"

// ── small things (flux_gui.cpp's). The manifest makes the ANSI code page UTF-8, so every narrow string is UTF-8. ──
static std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}
static std::wstring to_w(const std::string & s) { return fluximg::wide(s); }
static double now_s() { return (double) GetTickCount64() / 1000.0; }
static bool file_exists(const std::string & p) { const DWORD a = GetFileAttributesW(to_w(p).c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY); }
static std::string app_dir() {
    wchar_t b[MAX_PATH] = { 0 };
    GetModuleFileNameW(nullptr, b, MAX_PATH);
    const std::string p = narrow(b);
    const size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}
static std::string known_folder(REFKNOWNFOLDERID id) {
    PWSTR wp = nullptr;
    std::string out;
    if (SUCCEEDED(SHGetKnownFolderPath(id, 0, nullptr, &wp)) && wp) out = narrow(wp);
    if (wp) CoTaskMemFree(wp);
    return out;
}
static std::string read_all(const std::string & p) {
    FILE * f = _wfopen(to_w(p).c_str(), L"rb");
    if (!f) return "";
    std::string s;
    char b[1 << 16];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}
static bool write_all(const std::string & p, const std::string & s, const wchar_t * mode = L"wb") {
    FILE * f = _wfopen(to_w(p).c_str(), mode);
    if (!f) return false;
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
    return true;
}
static std::string esc(const std::string & s) {          // one line: a line break becomes \n, a backslash \\, a tab a space
    std::string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += ' ';
        else if (c != '\r') o += c;
    }
    return o;
}
static std::string unesc(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) { i++; o += s[i] == 'n' ? '\n' : s[i]; }
        else o += s[i];
    }
    return o;
}
static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v;
    size_t a = 0;
    for (;;) {
        const size_t b = s.find(sep, a);
        v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return v;
}
static std::string name_of(const std::string & p) { const size_t s = p.find_last_of("\\/"); return s == std::string::npos ? p : p.substr(s + 1); }
static bool is_picture_file(const std::string & p) {
    const size_t d = p.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = p.substr(d);
    for (char & c : e) c = (char) tolower((unsigned char) c);
    for (const char * k : { ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff" }) if (e == k) return true;
    return false;
}
static std::string quote_arg(const std::string & s) {    // one argument for CreateProcess (CommandLineToArgvW's rules)
    std::string o = "\"";
    size_t bs = 0;
    for (char c : s) {
        if (c == '\\') { bs++; continue; }
        if (c == '"') { o.append(bs * 2 + 1, '\\'); o += '"'; bs = 0; continue; }
        o.append(bs, '\\'); bs = 0;
        o += c;
    }
    o.append(bs * 2, '\\');
    return o + "\"";
}

static int g_lang = 0;                                    // 0 = svenska, 1 = English
static const char * tr(const char * sv, const char * en) { return g_lang ? en : sv; }

// ── DX11 plumbing ──
static ID3D11Device *           g_pd3dDevice        = nullptr;
static ID3D11DeviceContext *    g_pd3dDeviceContext = nullptr;
static IDXGISwapChain *         g_pSwapChain        = nullptr;
static ID3D11RenderTargetView * g_mainRTV           = nullptr;
static void CreateRenderTarget() {
    ID3D11Texture2D * back = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (back) { g_pd3dDevice->CreateRenderTargetView(back, nullptr, &g_mainRTV); back->Release(); }
}
static void CleanupRenderTarget() { if (g_mainRTV) { g_mainRTV->Release(); g_mainRTV = nullptr; } }
static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60; sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hWnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL lvls[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, lvls, 2, D3D11_SDK_VERSION,
                                      &sd, &g_pSwapChain, &g_pd3dDevice, &fl, &g_pd3dDeviceContext) != S_OK) return false;
    CreateRenderTarget();
    return true;
}
static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}
struct Tex {
    ID3D11ShaderResourceView * srv = nullptr;
    int w = 0, h = 0;
    void drop();                                          // released after the frame is shown (release_later)
    ImTextureRef ref() const { return ImTextureRef((ImTextureID) (uintptr_t) srv); }
};
// 10-11 (a crash: "Change …" on a picture, then OK -> qcdx11arm64xum.dll access violation): a texture released while
// the frame being built still draws it is used by the GPU after it is gone. Every texture the window gives up goes here
// and is released only after Present (release_frame_end), when no draw list can hold it any more.
static std::vector<IUnknown *> g_release_later;
static void release_later(IUnknown * p) { if (p) g_release_later.push_back(p); }
static void release_frame_end() { for (IUnknown * p : g_release_later) p->Release(); g_release_later.clear(); }
void Tex::drop() { release_later(srv); srv = nullptr; w = h = 0; }
static Tex make_tex(const uint8_t * rgba, int w, int h) {
    Tex t;
    D3D11_TEXTURE2D_DESC d{};
    d.Width = (UINT) w; d.Height = (UINT) h; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd{ rgba, (UINT) w * 4, 0 };
    ID3D11Texture2D * tx = nullptr;
    if (FAILED(g_pd3dDevice->CreateTexture2D(&d, &sd, &tx)) || !tx) return t;
    if (SUCCEEDED(g_pd3dDevice->CreateShaderResourceView(tx, nullptr, &t.srv))) { t.w = w; t.h = h; }
    tx->Release();
    return t;
}
// a picture file shrunk (area average) so its longer side is at most max_side, as RGBA; WIC on a thread of its own
static bool load_rgba(const std::string & path, int max_side, std::vector<uint8_t> & rgba, int & w, int & h) {
    fluximg::Rgb im;
    bool ok = false;
    std::thread t([&]() { try { im = fluximg::load(path); ok = im.w > 0 && im.h > 0; } catch (...) {} });
    t.join();
    if (!ok) return false;
    const double k = (double) max_side / (double) (std::max)(im.w, im.h);
    w = k >= 1.0 ? im.w : (std::max)(1, (int) (im.w * k + 0.5));
    h = k >= 1.0 ? im.h : (std::max)(1, (int) (im.h * k + 0.5));
    rgba.resize((size_t) w * h * 4);
    for (int y = 0; y < h; y++) {
        const int y0 = (int) ((int64_t) y * im.h / h), y1 = (std::max)(y0 + 1, (int) ((int64_t) (y + 1) * im.h / h));
        for (int x = 0; x < w; x++) {
            const int x0 = (int) ((int64_t) x * im.w / w), x1 = (std::max)(x0 + 1, (int) ((int64_t) (x + 1) * im.w / w));
            unsigned a[3] = { 0, 0, 0 };
            for (int yy = y0; yy < y1; yy++) {
                const uint8_t * s = &im.px[((size_t) yy * im.w + x0) * 3];
                for (int xx = x0; xx < x1; xx++, s += 3) { a[0] += s[0]; a[1] += s[1]; a[2] += s[2]; }
            }
            const unsigned n = (unsigned) ((y1 - y0) * (x1 - x0));
            uint8_t * d = &rgba[((size_t) y * w + x) * 4];
            d[0] = (uint8_t) ((a[0] + n / 2) / n); d[1] = (uint8_t) ((a[1] + n / 2) / n); d[2] = (uint8_t) ((a[2] + n / 2) / n); d[3] = 255;
        }
    }
    return true;
}

// ── window: no native frame; the title bar is ours (flux_gui.cpp's WndProc) ──
static std::string g_test_swap;         // --demo-swap: a picture to change to from inside the frame, as the Change button does
static float g_scale    = 1.0f;
static UINT  g_new_dpi  = 0;
static int   g_cap_h    = 0;
static RECT  g_nodrag[2] = {};
static std::vector<std::string> g_dropped;
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wP, LPARAM lP) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wP, lP)) return true;
    switch (msg) {
        case WM_SIZE:
            if (g_pd3dDevice && wP != SIZE_MINIMIZED) {
                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, (UINT) LOWORD(lP), (UINT) HIWORD(lP), DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
            return 0;
        case WM_NCCALCSIZE: if (wP) return 0; break;
        case WM_NCHITTEST: {
            const LONG b = (LONG) (8 * g_scale);
            POINT pt = { GET_X_LPARAM(lP), GET_Y_LPARAM(lP) };
            RECT r;
            GetWindowRect(hWnd, &r);
            if (!IsZoomed(hWnd)) {
                const bool L = pt.x < r.left + b, R = pt.x >= r.right - b, T = pt.y < r.top + b, B = pt.y >= r.bottom - b;
                if (T && L) return HTTOPLEFT; if (T && R) return HTTOPRIGHT; if (B && L) return HTBOTTOMLEFT; if (B && R) return HTBOTTOMRIGHT;
                if (L) return HTLEFT; if (R) return HTRIGHT; if (T) return HTTOP; if (B) return HTBOTTOM;
            }
            POINT c = pt;
            ScreenToClient(hWnd, &c);
            if (c.y < g_cap_h && !PtInRect(&g_nodrag[0], c) && !PtInRect(&g_nodrag[1], c)) return HTCAPTION;
            return HTCLIENT;
        }
        case WM_GETMINMAXINFO: {
            MINMAXINFO * mmi = (MINMAXINFO *) lP;
            HMONITOR mon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi{ sizeof(mi) };
            if (GetMonitorInfo(mon, &mi)) {
                mmi->ptMaxPosition.x = mi.rcWork.left - mi.rcMonitor.left; mmi->ptMaxPosition.y = mi.rcWork.top - mi.rcMonitor.top;
                mmi->ptMaxSize.x = mi.rcWork.right - mi.rcWork.left; mmi->ptMaxSize.y = mi.rcWork.bottom - mi.rcWork.top;
            }
            mmi->ptMinTrackSize.x = (LONG) (900 * g_scale); mmi->ptMinTrackSize.y = (LONG) (620 * g_scale);
            return 0;
        }
        case WM_DPICHANGED: {
            g_new_dpi = HIWORD(wP);
            const RECT * r = (const RECT *) lP;
            SetWindowPos(hWnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_DROPFILES: {
            HDROP d = (HDROP) wP;
            const UINT n = DragQueryFileW(d, 0xFFFFFFFF, nullptr, 0);
            for (UINT i = 0; i < n; i++) {
                wchar_t b[1024];
                if (DragQueryFileW(d, i, b, 1024)) g_dropped.push_back(narrow(b));
            }
            DragFinish(d);
            return 0;
        }
        case WM_SYSCOMMAND: if ((wP & 0xfff0) == SC_KEYMENU) return 0; break;
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hWnd, msg, wP, lP);
}

// ════════════════════════ look (PulseX Image's; violet, so the three windows are told apart) ════════════════════════
static const float FS = 21.0f;
static ImFont * g_font = nullptr, * g_semi = nullptr;
static float px(float v) { return v * g_scale; }
static ImVec4 rgb(int r, int g, int b, float a = 1.0f) { return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a); }
static const ImVec4 C_BG    = rgb( 58, 63, 71);
static const ImVec4 C_CARD  = rgb( 71, 77, 87);
static const ImVec4 C_INPUT = rgb( 47, 51, 59);
static const ImVec4 C_BTN   = rgb( 92, 99,112);
static const ImVec4 C_BTN_H = rgb(106,114,128);
static const ImVec4 C_TEXT  = rgb(236,239,243);
static const ImVec4 C_DIM   = rgb(170,178,190);
static const ImVec4 C_ACC   = rgb(140,104,206);
static const ImVec4 C_ACC_H = rgb(166,132,230);
static const ImVec4 C_ACC_L = rgb(116, 84,178);
static const ImVec4 C_WARN  = rgb(232,184, 92);
static const ImVec4 C_ERR   = rgb(236,120,110);
static const ImVec4 C_OK    = rgb(126,200,140);

static void apply_style() {
    ImGuiStyle st;
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 0; st.ChildRounding = 20; st.FrameRounding = 12; st.PopupRounding = 12; st.ScrollbarRounding = 10; st.GrabRounding = 10;
    st.WindowPadding = ImVec2(14, 12); st.FramePadding = ImVec2(11, 8); st.ItemSpacing = ImVec2(10, 9); st.ItemInnerSpacing = ImVec2(8, 6);
    st.ScrollbarSize = 11; st.WindowBorderSize = 0; st.ChildBorderSize = 0; st.PopupBorderSize = 1; st.FrameBorderSize = 1;
    ImVec4 * c = st.Colors;
    c[ImGuiCol_WindowBg] = C_BG; c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0); c[ImGuiCol_PopupBg] = rgb(50, 55, 63);
    c[ImGuiCol_Text] = C_TEXT; c[ImGuiCol_TextDisabled] = C_DIM; c[ImGuiCol_Border] = ImVec4(1, 1, 1, 0.07f);
    c[ImGuiCol_FrameBg] = C_INPUT; c[ImGuiCol_FrameBgHovered] = rgb(52, 57, 66); c[ImGuiCol_FrameBgActive] = rgb(52, 57, 66);
    c[ImGuiCol_Button] = C_BTN; c[ImGuiCol_ButtonHovered] = C_BTN_H; c[ImGuiCol_ButtonActive] = rgb(82, 89, 101);
    c[ImGuiCol_Header] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.30f); c[ImGuiCol_HeaderHovered] = ImVec4(1, 1, 1, 0.07f); c[ImGuiCol_HeaderActive] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.45f);
    c[ImGuiCol_CheckMark] = C_ACC_H; c[ImGuiCol_TextSelectedBg] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.45f); c[ImGuiCol_InputTextCursor] = C_TEXT;
    c[ImGuiCol_SliderGrab] = C_ACC; c[ImGuiCol_SliderGrabActive] = C_ACC_H;
    c[ImGuiCol_PlotHistogram] = C_ACC;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0); c[ImGuiCol_ScrollbarGrab] = ImVec4(1, 1, 1, 0.16f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1, 1, 1, 0.26f); c[ImGuiCol_ScrollbarGrabActive] = ImVec4(1, 1, 1, 0.34f);
    c[ImGuiCol_Separator] = ImVec4(1, 1, 1, 0.08f); c[ImGuiCol_NavCursor] = C_ACC_H;
    st.ScaleAllSizes(g_scale); st.FontSizeBase = FS; st.FontScaleDpi = g_scale;
    ImGui::GetStyle() = st;
}
static bool begin_card(const char * id, ImVec2 size, const char * title = nullptr, ImGuiChildFlags cf = 0, ImGuiWindowFlags wf = 0) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, C_CARD);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(20), px(16)));
    const bool v = ImGui::BeginChild(id, size, ImGuiChildFlags_AlwaysUseWindowPadding | cf, wf);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (title) { ImGui::PushFont(g_semi, FS * 1.08f); ImGui::TextUnformatted(title); ImGui::PopFont(); }
    return v;
}
static void end_card() { ImGui::EndChild(); }
static void hint(const char * t, const ImVec4 & col = C_DIM) {
    ImGui::PushFont(nullptr, FS * 0.84f);
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextWrapped("%s", t);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}
static float lines_h(int n) { return n * ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().FramePadding.y + px(2); }
static bool primary_button(const char * t, ImVec2 size = ImVec2(0, 0)) {
    ImGui::PushStyleColor(ImGuiCol_Button, C_ACC); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C_ACC_H); ImGui::PushStyleColor(ImGuiCol_ButtonActive, C_ACC_L);
    ImGui::PushFont(g_semi, 0.0f);
    const bool r = ImGui::Button(t, size);
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    return r;
}
static bool chip(const char * t, bool on, bool enabled = true) {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ImGui::GetFrameHeight() * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(15), ImGui::GetStyle().FramePadding.y));
    ImGui::PushStyleColor(ImGuiCol_Button, on ? C_ACC : C_INPUT);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, on ? C_ACC_H : rgb(60, 65, 75));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, C_ACC_L);
    ImGui::BeginDisabled(!enabled);
    const bool r = ImGui::Button(t);
    ImGui::EndDisabled();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    return r && enabled;
}
static void chip_flow(const char * next) {
    const float w = ImGui::CalcTextSize(next, nullptr, true).x + 2 * px(15);
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    if (ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + w <= right) ImGui::SameLine();
}

// ════════════════════════ the app ════════════════════════
// Seconds, measured 10-11 on this machine (image to video, 480x832, UniPC, CFG the first 60 %): a CFG step 22.8 s and a
// one-pass step 11.5 s at 49 frames, 74 / 37 s at 121; the big VAE 150 / 342 s; taew 2 s (NPU, 49) / 54 s (CPU, 121).
struct Est { double start, text, picture, cfg_step, one_step, vae, write; };
static Est estimate(int frames, bool big_vae, bool picture, bool hd = false) {
    // 10-12 evening: today's attention (HMX row sums, K cache, fixed-point exp) and the
    // big VAE (golden control, 4D step graph 0.99 s per call) - the 10-11 numbers stay in the comments' history
    if (hd) {   // 1280 x 704
        if (frames > 49) {   // 121 frames (27280 tokens), measured 10-12 (a 720p 5 s film, 39.5 min fast): a CFG step 193.9 s, one pass 96.5 s, the big VAE 602.8 s (was 269.5 / 135.4 / 776)
            // 10-13: the big VAE w8a8 by default; since 10-13 pm with fp16 at its boundary (0.42 s per call; the VAE 259 s, the film 33.6 min)
            Est e{ 20.0, 1.0, picture ? 6.0 : 0.0, 193.9, 96.5, big_vae ? 259.0 : 160.0, 8.0 };
            return e;
        }
        // 49 frames (11440 tokens): ESTIMATED from 480p 5 s (12090 tokens, 74.3 -> 62.2 s per CFG step, -16 %) on the 10-11
        // 68.9 / 34.3 s; the big VAE 18 tiles x 11 calls x 0.99 s + frames 0-1 ~ 240 s
        // 10-13: the w8a8 VAE: 18 tiles x 11 calls x 0.46 s + frames 0-1 39 s ~ 135 s
        Est e{ 20.0, 1.0, picture ? 6.0 : 0.0, 58.0, 29.0, big_vae ? 135.0 : 60.0, 4.0 };   // the picture: the 720p NPU encoder, 5.5 s
        return e;
    }
    // 480 x 832. 121 frames measured 10-12 (a film, 16.9 min, not fast): a CFG step 62.2 s, one pass 31.2 s, the big
    // VAE 252 s (was 74.3 / 37.2 / 342). 49 frames (5070 tokens, resident flash path): ESTIMATED 22.8 / 11.5 -> 21.5 / 11.0,
    // the big VAE 8 tiles x 11 calls x 0.99 s + frames 0-1 ~ 110 s (was 150)
    const bool l = frames > 49;
    // 10-13: the w8a8 VAE: 8 tiles x 29 / 11 calls x 0.46 s + frames 0-1 19 s ~ 132 / 63 s
    Est e{ 18.0, 1.0, picture ? 3.0 : 0.0, l ? 62.2 : 21.5, l ? 31.2 : 11.0, big_vae ? (l ? 132.0 : 63.0) : (l ? 54.0 : 3.0), l ? 4.0 : 2.0 };
    return e;
}
static int cfg_steps(int steps) { return (int) (steps * 0.4 + 0.5 - 1e-9); }        // wan_pipeline: "guide ... for the first N"; 10-23: 0.6 -> 0.4 (OPTIMERING_1023 §5: -9..12 %, 43.5 dB = the run-to-run floor)
// 10-11: "15 snabb" - these of the 15 steps reuse the last residual instead of running the 30 blocks (TeaCache-style;
// single skips only: pairs broke image to video at 5 s). A skipped step takes about 3 s.
// 10-13: the "15 snabb" button is gone, Turbo took its place. The recipe
// stays: wan_pipeline.py ... --steps 15 --skip-steps 6,8,10,12 (720p 5 s 33.6 min, 40.2 dB vs the facit); the films
// made with it keep their "fast" in films.tsv and in the history, and fast_skipped still times them.
static const char * FAST_SKIPS = "6,8,10,12";
static bool fast_skipped(bool fast, int steps, int k) { return fast && steps == 15 && (k == 5 || k == 7 || k == 9 || k == 11); }   // k 0-based

struct Film { std::string when, mp4, prompt, picture; int frames = 49, steps = 15; bool big = true, hd = false, calm = false, fast = false, turbo = false; unsigned seed = 0; double secs = 0; };   // fast: 15 steps with skips   // hd: 1280 x 704; calm: image to video at CFG 3   // turbo: Wan2.2-TI2V-5B-Turbo, 4 steps, no CFG
// 10-13: Wan2.2-TI2V-5B-Turbo (Self-Forcing/DMD, hum-ma's Q4_0; models\turbo,
// converted like the base model) - 4 steps, one pass each (guide 1), euler. 720p 5 s 11.3 min (pass 98 s) vs 33.6;
// more motion, sharper, slightly warmer; reflections and background move more than with the base model.
static const int TURBO_STEPS = 4;

struct App {
    // what exists
    std::string python, wan, models, out_dir, data_dir, not_ready;
    bool has_big = false, has_fast = false, has_turbo = false;
    // choices
    char prompt[2048] = "";
    std::string picture;                  // the start picture (empty = text to video)
    Tex pic_thumb;
    int frames = 49, steps = 15;
    bool turbo = false;                   // 10-13: Turbo (steps == TURBO_STEPS); took "15 snabb"'s place
    bool big_vae = true, random_seed = true;
    bool hd = false;                      // 10-11: 720p = 1280 x 704; 5 s since the FFN token pieces (wan_stream.cpp 7760561)
    // 10-11 (a calm image-to-video film turned aggressive, the background too alive): image to video "calm" = CFG 3 (the
    // default), "lively" = CFG 5 as before - CFG 3 with a calm prompt kept faces soft and reflections still
    // (tools/i2v_calm_probe_1011.py: window change 44 -> 22, flicker 4.2 -> 2.5); text to video keeps CFG 5 + rescale
    bool calm = true;
    int seed = 1;
    // the job
    std::thread th;
    std::mutex m;
    std::atomic<bool> running{ false }, finished{ false };
    int rc = 0;
    double t0 = 0, t_stage = 0;
    std::string stage, log, error, work, last_mp4, frame0;
    int step = 0, nsteps = 1;
    Film job;
    // the newest film's preview, played in the window
    std::vector<uint8_t> frames_rgb;
    int pw = 0, ph = 0, nf = 0, shown = -1;
    ID3D11Texture2D * tex = nullptr;
    ID3D11ShaderResourceView * srv = nullptr;
    double play0 = 0;
    std::string preview_path, result_line;
    std::vector<Film> films;              // newest first
    int film_ix = 0;
    bool test_run = false, demo_cpu = false;

    void find_things() {
        const char * e = getenv("PULSEX_PYTHON");
        python = e && *e ? e : "";
        if (python.empty()) {
            const std::string la = known_folder(FOLDERID_LocalAppData);
            for (const char * v : { "\\Programs\\Python\\Python312-arm64\\python.exe", "\\Programs\\Python\\Python312\\python.exe" })
                if (file_exists(la + v)) { python = la + v; break; }
        }
        // 10-24 (publishing): the folders from the environment (PULSEX_WAN, PULSEX_WAN_MODELS, PULSEX_WAN_BIN, PULSEX_QNN,
        // PULSEX_FFMPEG), else from wan.json beside the exe ({"paths": {"wan", "models", "bin", "qnn", "ffmpeg", ...}}, a
        // relative path from wan.json's folder), else the installed layout (bin\..\python, bin\..\models). What is found goes on to wan_pipeline.py
        // through the environment (pulsex_paths.py reads the same names, and wan.json itself via PULSEX_WAN_JSON);
        // nothing set and no wan.json = Python's own defaults, exactly as before.
        const std::string cfg_dir = app_dir(), cfg_json = cfg_dir + "\\wan.json", js = read_all(cfg_json);
        auto from_json = [&](const char * key) -> std::string {
            const size_t k = js.find(std::string("\"") + key + "\"");
            const size_t c = k == std::string::npos ? k : js.find(':', k + strlen(key) + 2);
            const size_t a = c == std::string::npos ? c : js.find('"', c);
            const size_t b = a == std::string::npos ? a : js.find('"', a + 1);
            if (b == std::string::npos) return "";
            std::string v;
            for (size_t i = a + 1; i < b; i++) {
                if (js[i] == '\\' && i + 1 < b && js[i + 1] == '\\') i++;     // JSON's escaped backslash
                v += js[i] == '/' ? '\\' : js[i];
            }
            if (!v.empty() && !(v.size() > 1 && v[1] == ':') && v[0] != '\\') v = cfg_dir + "\\" + v;
            return v;
        };
        auto setting = [&](const char * env, const char * key, bool hand_on) -> std::string {
            const std::wstring we = to_w(env);
            const wchar_t * e = _wgetenv(we.c_str());
            const std::string v = e && *e ? narrow(e) : from_json(key);
            if (hand_on && !v.empty()) SetEnvironmentVariableW(we.c_str(), to_w(v).c_str());   // wan_pipeline.py inherits it
            return v;
        };
        if (!js.empty() && !_wgetenv(L"PULSEX_WAN_JSON")) SetEnvironmentVariableW(L"PULSEX_WAN_JSON", to_w(cfg_json).c_str());
        wan = setting("PULSEX_WAN", "wan", false);
        if (wan.empty()) wan = cfg_dir + "\\..\\python";                     // the installed layout: <app>\python beside <app>\bin
        models = setting("PULSEX_WAN_MODELS", "models", true);
        if (models.empty()) models = cfg_dir + "\\..\\models";
        setting("PULSEX_WAN_BIN", "bin", true);
        setting("PULSEX_QNN", "qnn", true);
        setting("PULSEX_FFMPEG", "ffmpeg", true);
        // 10-24: "Sharp" = what wan_pipeline.py runs for --vae npu_full: the w8a8 step graph (its default) + the frames-0-1
        // graph. Was vae_npu_step_p3 (a junction to D: that the pipeline no longer reads): greyed out on any other machine.
        has_big = file_exists(models + "\\vae_npu_step_w8a8\\model.onnx") && file_exists(models + "\\vae_npu_step_w8a8\\model_qnn.bin") &&
                  file_exists(models + "\\vae_npu_first_p3\\model.onnx") && file_exists(models + "\\vae_npu_first_p3\\model.bin");
        has_fast = file_exists(models + "\\taew_npu_ctx\\model.bin");
        has_turbo = file_exists(models + "\\turbo\\Wan2.2-TI2V-5B-Turbo-Q4_0-ffn2q8-ggml-slim.gguf") &&
                    file_exists(models + "\\turbo\\Wan2.2-TI2V-5B-Turbo-Q4_0-ffn2q8-ggml-slim.hexpack") &&
                    file_exists(models + "\\turbo\\Wan2.2-TI2V-5B-Turbo-npu-extras.gguf");
        if (!has_big) big_vae = false;
        const char * o = getenv("PULSEX_WAN_OUT");
        out_dir = o && *o ? o : known_folder(FOLDERID_Videos) + "\\PulseX Wan";
        const char * d = getenv("PULSEX_WAN_DATA");
        data_dir = d && *d ? d : cfg_dir + "\\..\\data";                 // the data folder of the installed program: <app>\data
        CreateDirectoryW(to_w(out_dir).c_str(), nullptr);
        CreateDirectoryW(to_w(data_dir).c_str(), nullptr);
        work = data_dir + "\\work";
        CreateDirectoryW(to_w(work).c_str(), nullptr);
        preview_path = data_dir + "\\newest_preview.rgb";
        not_ready.clear();
        if (python.empty()) not_ready = tr("Python hittades inte (PULSEX_PYTHON).", "Python was not found (PULSEX_PYTHON).");
        else if (!file_exists(wan + "\\wan_pipeline.py")) not_ready = tr("wan_pipeline.py hittades inte (PULSEX_WAN).", "wan_pipeline.py was not found (PULSEX_WAN).");
        else if (!has_big && !has_fast) not_ready = std::string(tr("Inga VAE-kontexter i ", "No VAE contexts in ")) + models + " (PULSEX_WAN_MODELS / wan.json).";
    }

    // ── settings and the list of films (<app>\data) ──
    void settings_save() {
        std::string s = "lang=" + std::to_string(g_lang) + "\nframes=" + std::to_string(frames) + "\nsteps=" + std::to_string(steps) +
                        "\nbig=" + std::to_string(big_vae ? 1 : 0) + "\nrandom=" + std::to_string(random_seed ? 1 : 0) +
                        "\nseed=" + std::to_string(seed) + "\nhd=" + std::to_string(hd ? 1 : 0) + "\ncalm=" + std::to_string(calm ? 1 : 0) + "\nturbo=" + std::to_string(turbo ? 1 : 0) + "\nprompt=" + esc(prompt) + "\npicture=" + esc(picture) + "\n";
        write_all(data_dir + "\\settings.txt", s);
    }
    void settings_load() {
        for (const std::string & l : split(read_all(data_dir + "\\settings.txt"), '\n')) {
            const size_t e = l.find('=');
            if (e == std::string::npos) continue;
            const std::string k = l.substr(0, e), v = l.substr(e + 1);
            if (k == "lang") g_lang = atoi(v.c_str()) ? 1 : 0;
            else if (k == "frames") frames = atoi(v.c_str()) > 49 ? 121 : 49;
            else if (k == "steps") { const int s = atoi(v.c_str()); steps = s == 10 || s == 12 || s == TURBO_STEPS ? s : 15; }
            else if (k == "big") big_vae = atoi(v.c_str()) != 0 && has_big;
            else if (k == "random") random_seed = atoi(v.c_str()) != 0;
            else if (k == "seed") seed = atoi(v.c_str());
            else if (k == "hd") hd = atoi(v.c_str()) != 0;
            else if (k == "calm") calm = atoi(v.c_str()) != 0;
            else if (k == "turbo") turbo = atoi(v.c_str()) != 0;   // an old "fast=1" (15 snabb) is not read: plain 15
            else if (k == "prompt") snprintf(prompt, sizeof prompt, "%s", unesc(v).c_str());
            else if (k == "picture") { const std::string p = unesc(v); if (file_exists(p)) set_picture(p); }
        }
        if (!has_big) big_vae = false;
        if (!has_fast) big_vae = has_big;
        if (!has_turbo) turbo = false;
        if (turbo != (steps == TURBO_STEPS)) { turbo = false; if (steps == TURBO_STEPS) steps = 15; }
    }
    void films_load() {
        films.clear();
        for (const std::string & l : split(read_all(data_dir + "\\films.tsv"), '\n')) {
            const std::vector<std::string> f = split(l, '\t');
            if (f.size() < 9 || !file_exists(f[1])) continue;
            Film x;
            x.when = f[0]; x.mp4 = f[1]; x.frames = atoi(f[2].c_str()); x.steps = atoi(f[3].c_str()); x.big = f[4] == "1";
            x.seed = (unsigned) strtoul(f[5].c_str(), nullptr, 10); x.secs = atof(f[6].c_str()); x.picture = unesc(f[7]); x.prompt = unesc(f[8]);
            x.hd = f.size() > 9 && f[9] == "720";                  // a 10th field since 10-11 (older lines: 480p)
            x.calm = f.size() > 10 && f[10] == "calm";              // an 11th since 10-11 (older lines: lively, CFG 5)
            x.fast = f.size() > 11 && f[11] == "fast";              // a 12th since 10-11 night (older lines: no skips)
            x.turbo = f.size() > 12 && f[12] == "turbo";            // a 13th since 10-13 (older lines: the base model)
            films.insert(films.begin(), x);
        }
    }
    void film_add(const Film & x) {
        char secs[32];
        snprintf(secs, sizeof secs, "%.0f", x.secs);
        const std::string l = x.when + "\t" + x.mp4 + "\t" + std::to_string(x.frames) + "\t" + std::to_string(x.steps) + "\t" + (x.big ? "1" : "0") + "\t" +
                              std::to_string(x.seed) + "\t" + secs + "\t" + esc(x.picture) + "\t" + esc(x.prompt) + "\t" + (x.hd ? "720" : "480") + "\t" + (x.calm ? "calm" : "lively") + "\t" + (x.fast ? "fast" : "full") + "\t" + (x.turbo ? "turbo" : "base") + "\n";
        write_all(data_dir + "\\films.tsv", l, L"ab");
        films.insert(films.begin(), x);
        film_ix = 0;
    }

    void set_picture(const std::string & p) {
        std::vector<uint8_t> rgba;
        int w = 0, h = 0;
        if (!load_rgba(p, 640, rgba, w, h)) { std::lock_guard<std::mutex> l(m); error = tr("Bilden gick inte att läsa: ", "The picture could not be read: ") + name_of(p); return; }
        pic_thumb.drop();
        pic_thumb = make_tex(rgba.data(), w, h);
        picture = p;
    }

    static std::string npu_busy() {
        std::string hit;
        HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (s == INVALID_HANDLE_VALUE) return hit;
        PROCESSENTRY32W pe{ sizeof pe };
        if (Process32FirstW(s, &pe))
            do {
                std::string n = narrow(pe.szExeFile);
                for (char & c : n) c = (char) tolower((unsigned char) c);
                for (const char * k : { "wan-stream", "flux-make", "flux-stream", "flux-encode", "flux-vae", "flux-qnn", "zimage", "ltx", "llama-server", "test-backend-ops" })
                    if (n.compare(0, strlen(k), k) == 0) hit = n;
            } while (hit.empty() && Process32NextW(s, &pe));
        CloseHandle(s);
        return hit;
    }

    // ── the job: wan_pipeline.py, its stdout and stderr through one pipe, line by line ──
    void start() {
        if (running || !not_ready.empty()) return;
        std::string p = prompt;
        while (!p.empty() && (p.back() == ' ' || p.back() == '\n' || p.back() == '\r')) p.pop_back();
        if (p.empty()) { std::lock_guard<std::mutex> l(m); error = tr("Skriv vad filmen ska visa.", "Write what the film should show."); return; }
        const std::string busy = npu_busy();
        if (!busy.empty()) { std::lock_guard<std::mutex> l(m); error = std::string(tr("NPU:n används av ", "The NPU is in use by ")) + busy; return; }
        if (random_seed) { std::random_device rd; seed = (int) (rd() % 1000000); }
        SYSTEMTIME t; GetLocalTime(&t);
        char stamp[64];
        snprintf(stamp, sizeof stamp, "%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
        job = Film();
        job.when = stamp; job.prompt = p; job.picture = picture; job.frames = frames; job.steps = steps; job.big = big_vae; job.seed = (unsigned) seed;
        job.hd = hd;
        job.calm = calm && !picture.empty();                 // the choice is image to video's
        job.turbo = turbo && has_turbo;
        if (job.turbo) { job.steps = TURBO_STEPS; job.calm = false; }   // Turbo: one pass per step, no CFG to be calm with
        drop_preview();
        { std::lock_guard<std::mutex> l(m); stage = "start"; log.clear(); error.clear(); last_mp4.clear(); frame0.clear(); result_line.clear(); step = 0; nsteps = job.steps; }
        t0 = t_stage = now_s();
        finished = false; running = true;
        settings_save();
        if (th.joinable()) th.join();
        th = std::thread([this]() { work_(); });
    }
    std::string vae_arg() const {
        if (job.big) return "npu_full";
        return job.frames <= 49 && !job.hd ? "taew" : "taew_cpu";   // the NPU taew is compiled for 13 latent frames (49) at 480p
    }
    void work_() {
        const auto & q = quote_arg;
        std::string cmd = q(python) + " -u -X utf8 " + q(wan + "\\wan_pipeline.py") + " " + q(job.prompt);
        if (!job.picture.empty()) cmd += " --image " + q(job.picture);
        if (job.turbo) cmd += demo_cpu ? " --guide 1" : " --dit turbo --guide 1";   // 10-13: distilled without CFG - one pass, no rescale (the CPU test hook: the base blocks)
        else if (!job.picture.empty()) cmd += job.calm ? " --guide 3" : " --guide 5";
        // 10-11 (text to video came out cyan and blown): CFG rescale 0.7 for text to video - saturation 0.40 -> 0.25,
        // blue-green excess 60 -> 33, the same time (tools/t2v_colour_probe_1011.py). Image to video was natural already
        // (the picture anchors the colours) and is not tried with it: left as it was.
        else cmd += " --cfg-rescale 0.7";
        if (demo_cpu) cmd += " --backend cpu --blocks 1 --steps 1 --frames 5 --vae taew_cpu";      // the test hook: no NPU
        else cmd += " --backend ggml --vae " + vae_arg() + " --steps " + std::to_string(job.steps) + " --frames " + std::to_string(job.frames) +
                    (job.fast ? std::string(" --skip-steps ") + FAST_SKIPS : std::string());
        // Turbo: euler (shift 5 = the sigmas DMD trained at)
        cmd += std::string(job.turbo ? " --solver euler" : " --solver unipc") + " --cfg-until 0.4 --size " + (job.hd ? "704x1280" : "480x832") + " --seed " + std::to_string(job.seed) + " --name pulsex_wan --outdir " + q(work) +
               " --threads 6 --no-png --preview " + q(preview_path);
        std::wstring w = to_w(cmd);
        SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
        HANDLE rd = nullptr, wr = nullptr;
        int code = 100;
        if (CreatePipe(&rd, &wr, &sa, 0)) {
            SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
            STARTUPINFOW si{};
            si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wr; si.hStdError = wr;
            PROCESS_INFORMATION pi{};
            // heavy jobs: below normal priority, cores 2-7 (cores 0-1 stay free for the desktop); the engines inherit both
            const BOOL ok = CreateProcessW(nullptr, &w[0], nullptr, nullptr, TRUE, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS | CREATE_SUSPENDED,
                                           nullptr, to_w(wan).c_str(), &si, &pi);
            CloseHandle(wr);
            if (ok) {
                SetProcessAffinityMask(pi.hProcess, 0xFC);
                ResumeThread(pi.hThread);
                std::string line;
                char b[4096];
                DWORD n = 0;
                while (ReadFile(rd, b, sizeof b, &n, nullptr) && n)
                    for (DWORD i = 0; i < n; i++) {
                        if (b[i] == '\n') { on_line(line); line.clear(); }
                        else if (b[i] != '\r') line += b[i];
                    }
                if (!line.empty()) on_line(line);
                WaitForSingleObject(pi.hProcess, INFINITE);
                DWORD ec = 1;
                GetExitCodeProcess(pi.hProcess, &ec);
                code = (int) ec;
                CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
            } else {
                std::lock_guard<std::mutex> l(m);
                log += "CreateProcess failed: " + std::to_string(GetLastError()) + "\n";
            }
            CloseHandle(rd);
        }
        rc = code;
        finished = true;
    }
    void on_line(const std::string & line) {
        std::lock_guard<std::mutex> l(m);
        auto begins = [&](const char * k) { return line.compare(0, strlen(k), k) == 0; };
        auto next = [&](const char * s) { if (stage != s) { stage = s; t_stage = now_s(); } };
        if (begins("RESULT\t")) { result_line = line.substr(7); return; }    // tab-separated: the paths can hold spaces
        if (begins("frame 0 vs the picture: ")) { frame0 = line.substr(24); return; }
        if (begins("umT5") || begins("text cache")) { next(job.picture.empty() ? "steps0" : "picture"); }
        else if (begins("picture ")) next("steps0");
        else if (begins("step ")) {
            int a = 0, b = 0;
            if (sscanf(line.c_str(), "step %d/%d", &a, &b) == 2) { step = a; nsteps = (std::max)(1, b); stage = "steps"; t_stage = now_s(); }
        }
        else if (begins("wan-stream totals")) next("vae");
        else if (begins("wrote ") && line.find(".mp4") != std::string::npos) next("write");
        log += line + "\n";
        if (log.size() > 60000) log.erase(0, log.size() - 40000);
    }
    // how far along (0..1), what is happening, about how long is left
    float fraction(std::string & what, double & left) {
        std::lock_guard<std::mutex> l(m);
        const Est e = estimate(job.frames, job.big, !job.picture.empty(), job.hd);
        const int nc = job.turbo ? 0 : cfg_steps(job.steps);             // Turbo: every step one pass
        auto step_s = [&](int k) { return fast_skipped(job.fast, job.steps, k) ? 3.0 : k < nc ? e.cfg_step : e.one_step; };   // step k (0-based)
        double steps_total = 0;
        for (int k = 0; k < job.steps; k++) steps_total += step_s(k);
        const double total = e.start + e.text + e.picture + steps_total + e.vae + e.write;
        const double el = now_s() - t_stage;
        double before = 0, cur = e.start;
        char b[128];
        if (stage == "start") what = tr("Startar och läser in modellerna …", "Starting, loading the models …");
        else if (stage == "picture") { before = e.start + e.text; cur = e.picture; what = tr("Läser startbilden …", "Reading the start picture …"); }
        else if (stage == "steps0" || stage == "steps") {
            before = e.start + e.text + e.picture;
            for (int k = 0; k < step && k < job.steps; k++) before += step_s(k);
            cur = step_s(step);
            snprintf(b, sizeof b, tr("Skapar rörelsen – steg %d av %d", "Making the motion – step %d of %d"), (std::min)(step + 1, nsteps), nsteps);
            what = b;
        }
        else if (stage == "vae") { before = total - e.vae - e.write; cur = e.vae; what = job.big ? tr("Målar rutorna (stora VAE:n) …", "Painting the frames (the big VAE) …") : tr("Målar rutorna …", "Painting the frames …"); }
        else if (stage == "write") { before = total - e.write; cur = e.write; what = tr("Sparar filmen …", "Saving the film …"); }
        const double done = before + (std::min)(el, cur * 0.97);
        left = (std::max)(0.0, total - done);
        return (float) (std::min)(0.99, done / total);
    }

    void drop_preview() {
        release_later(srv); srv = nullptr;
        release_later(tex); tex = nullptr;
        frames_rgb.clear(); shown = -1; nf = 0;
    }
    bool load_preview(const std::string & path, int w, int h, int n) {
        drop_preview();
        const std::string raw = read_all(path);
        if (w <= 0 || h <= 0 || n <= 0 || raw.size() != (size_t) w * h * 3 * n) return false;
        frames_rgb.assign(raw.begin(), raw.end());
        pw = w; ph = h; nf = n;
        D3D11_TEXTURE2D_DESC d{};
        d.Width = (UINT) pw; d.Height = (UINT) ph; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DYNAMIC; d.BindFlags = D3D11_BIND_SHADER_RESOURCE; d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(g_pd3dDevice->CreateTexture2D(&d, nullptr, &tex)) || !tex) { drop_preview(); return false; }
        g_pd3dDevice->CreateShaderResourceView(tex, nullptr, &srv);
        play0 = now_s();
        return true;
    }
    void preview_meta_save(int w, int h, int n) { write_all(data_dir + "\\newest_preview.txt", std::to_string(w) + " " + std::to_string(h) + " " + std::to_string(n) + "\n"); }
    void preview_meta_load() {
        int w = 0, h = 0, n = 0;
        if (sscanf(read_all(data_dir + "\\newest_preview.txt").c_str(), "%d %d %d", &w, &h, &n) == 3) load_preview(preview_path, w, h, n);
    }

    void pump() {
        if (running && finished) {
            if (th.joinable()) th.join();
            running = false;
            std::string res;
            {
                std::lock_guard<std::mutex> l(m);
                res = result_line;
                write_all(data_dir + "\\last_run.log", "exit " + std::to_string(rc) + "\n" + log);   // kept beyond the window
            }
            // 10-11 (the first film): split on spaces, "%LOCALAPPDATA%\PulseX Wan\..." fell apart - the film was made but
            // never moved, no preview, and the message went with the window. Tabs now (wan_pipeline.py writes them).
            const std::vector<std::string> f = split(res, '\t');
            if (rc == 0 && f.size() >= 5) {
                // the film to Videos\PulseX Wan, under a name of its own (never over another)
                std::string dst = out_dir + "\\pulsex_wan_" + job.when + ".mp4";
                for (int k = 2; file_exists(dst); k++) dst = out_dir + "\\pulsex_wan_" + job.when + "_" + std::to_string(k) + ".mp4";
                if (MoveFileExW(to_w(f[0]).c_str(), to_w(dst).c_str(), MOVEFILE_COPY_ALLOWED)) {
                    job.mp4 = dst;
                    job.secs = now_s() - t0;
                    film_add(job);
                    last_mp4 = dst;
                } else { std::lock_guard<std::mutex> l(m); error = tr("Filmen kunde inte flyttas till ", "The film could not be moved to ") + out_dir; }
                const int w = atoi(f[2].c_str()), h = atoi(f[3].c_str()), n = atoi(f[4].c_str());
                if (load_preview(f[1], w, h, n)) preview_meta_save(w, h, n);
            } else {
                std::lock_guard<std::mutex> l(m);
                if (error.empty()) {
                    const size_t k = log.size() > 900 ? log.size() - 900 : 0;
                    error = std::string(tr("Filmen blev inte klar. Det sista som hände:\n", "The film did not finish. The last thing that happened:\n")) + log.substr(k);
                }
            }
        }
        if (srv && !frames_rgb.empty()) {                  // the frame of this moment, 24 a second, round and round
            const int fr = (int) ((now_s() - play0) * 24.0) % nf;
            if (fr != shown) {
                D3D11_MAPPED_SUBRESOURCE ms{};
                if (SUCCEEDED(g_pd3dDeviceContext->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms))) {
                    const uint8_t * s3 = frames_rgb.data() + (size_t) fr * pw * ph * 3;
                    for (int y = 0; y < ph; y++) {
                        uint8_t * d4 = (uint8_t *) ms.pData + (size_t) y * ms.RowPitch;
                        for (int x = 0; x < pw; x++, s3 += 3, d4 += 4) { d4[0] = s3[0]; d4[1] = s3[1]; d4[2] = s3[2]; d4[3] = 255; }
                    }
                    g_pd3dDeviceContext->Unmap(tex, 0);
                    shown = fr;
                }
            }
        }
    }
};

static bool pick_picture(HWND owner, std::string & out) {
    wchar_t file[1024] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof ofn; ofn.hwndOwner = owner; ofn.lpstrFile = file; ofn.nMaxFile = 1024;
    ofn.lpstrFilter = L"Pictures\0*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff\0\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;
    if (!GetOpenFileNameW(&ofn)) return false;
    out = narrow(file);
    return true;
}
static std::string when_text(const std::string & w) {          // "20261011_105511" -> "2026-10-11 10:55"
    if (w.size() < 13) return w;
    return w.substr(0, 4) + "-" + w.substr(4, 2) + "-" + w.substr(6, 2) + " " + w.substr(9, 2) + ":" + w.substr(11, 2);
}
static std::string mins(double s) {
    char b[48];
    if (s < 90) snprintf(b, sizeof b, "%.0f s", s);
    else snprintf(b, sizeof b, "%.0f min", s / 60.0);
    return b;
}

// ── left: what to make ──
static void draw_left(App & a, HWND hWnd) {
    const float action_h = a.running ? px(118) : px(66);           // the progress needs room only while a film is made
    ImGui::BeginChild("##choices", ImVec2(0, ImGui::GetContentRegionAvail().y - action_h - ImGui::GetStyle().ItemSpacing.y), 0, 0);
    ImGui::BeginDisabled(a.running);

    begin_card("##prompt", ImVec2(0, 0), tr("Beskriv filmen", "Describe the film"), ImGuiChildFlags_AutoResizeY);
    ImGui::InputTextMultiline("##p", a.prompt, sizeof a.prompt, ImVec2(-1, lines_h(3)), ImGuiInputTextFlags_WordWrap);
    if (!a.prompt[0]) {
        static const char * EX[] = { "A powerful waterfall on a wide northern Swedish river, white water over a rocky ledge, spray at the foot, dark pine forest, overcast sky, static camera, natural motion",
                                     "A red wooden cottage by a calm lake at dusk, mist drifting over the water, a slow camera pan" };
        for (int i = 0; i < 2; i++) { ImGui::PushID(i); if (ImGui::SmallButton(i == 0 ? tr("Exempel: forsen", "Example: the rapids") : tr("Exempel: stugan", "Example: the cottage"))) snprintf(a.prompt, sizeof a.prompt, "%s", EX[i]); ImGui::PopID(); if (i == 0) ImGui::SameLine(); }
    } else if (!a.picture.empty())                          // 10-11: image to video - what should stay still matters as much
        hint(tr("Engelska fungerar bäst. Beskriv uttrycket och det som ska stå stilla, t.ex. \"calm, gentle expression; the background stays still\".",
                "English works best. Describe the expression and what should stay still, e.g. \"calm, gentle expression; the background stays still\"."));
    else hint(tr("Engelska fungerar bäst; säg vad som rör sig och hur kameran står.", "English works best; say what moves and how the camera stands."));
    end_card();

    begin_card("##pic", ImVec2(0, 0), tr("Startbild (valfri)", "Start picture (optional)"), ImGuiChildFlags_AutoResizeY);
    if (!a.picture.empty() && a.pic_thumb.srv) {
        const float k = (std::min)(px(150) / a.pic_thumb.w, px(86) / a.pic_thumb.h);
        ImGui::Image(a.pic_thumb.ref(), ImVec2(a.pic_thumb.w * k, a.pic_thumb.h * k));
        ImGui::SameLine();
        ImGui::BeginGroup();
        hint(name_of(a.picture).c_str(), C_TEXT);
        hint(a.hd ? tr("Beskärs till 1280 × 704.", "Cropped to 1280 × 704.") : tr("Beskärs till 832 × 480.", "Cropped to 832 × 480."));
        if (ImGui::Button(tr("Byt …", "Change …"))) { std::string f; if (pick_picture(hWnd, f)) a.set_picture(f); }
        else if (!g_test_swap.empty()) { a.set_picture(g_test_swap); g_test_swap.clear(); }     // test hook --demo-swap: the same, mid-frame
        ImGui::SameLine();
        if (ImGui::Button(tr("Ingen bild", "No picture"))) { a.picture.clear(); a.pic_thumb.drop(); }
        ImGui::EndGroup();
    } else {
        if (ImGui::Button(tr("Välj bild …", "Choose a picture …"))) { std::string f; if (pick_picture(hWnd, f)) a.set_picture(f); }
        ImGui::SameLine();
        ImGui::BeginGroup();
        hint(tr("eller dra in en bild. Utan bild görs filmen ur texten.", "or drop one on the window. Without one, the film comes from the text."));
        ImGui::EndGroup();
    }
    end_card();

    begin_card("##len", ImVec2(0, 0), tr("Längd och kvalitet", "Length and quality"), ImGuiChildFlags_AutoResizeY);
    const float lab = px(74);
    auto row_label = [&](const char * t) { ImGui::AlignTextToFramePadding(); hint(t); ImGui::SameLine(lab); };
    row_label(tr("Längd", "Length"));
    if (chip(tr("2 s###f49", "2 s###f49"), a.frames == 49)) a.frames = 49;
    ImGui::SameLine();
    if (chip(tr("5 s###f121", "5 s###f121"), a.frames == 121)) a.frames = 121;
    if (a.hd) ImGui::SetItemTooltip("%s", tr("5 s i 720p tar 30–40 minuter.", "5 s at 720p takes 30–40 minutes."));
    ImGui::SameLine(0, px(26));                               // 10-11: the resolution on the same row
    if (chip("480p###r480", !a.hd)) a.hd = false;
    ImGui::SetItemTooltip("%s", tr("832 × 480 – snabbast.", "832 × 480 – the fastest."));
    ImGui::SameLine();
    if (chip("720p###r720", a.hd)) a.hd = true;
    ImGui::SetItemTooltip("%s", tr("1280 × 704 – Wans egen upplösning; ungefär tre gånger så lång tid.", "1280 × 704 – Wan's own resolution; about three times as long."));
    row_label(tr("Bild", "Picture"));
    if (chip(tr("Skarp###vb", "Sharp###vb"), a.big_vae, a.has_big)) a.big_vae = true;
    ImGui::SetItemTooltip("%s", tr("Stora VAE:n på NPU:n – skarpast bild.", "The big VAE on the NPU – the sharpest picture."));
    ImGui::SameLine();
    if (chip(tr("Snabb###vf", "Fast###vf"), !a.big_vae, a.has_fast)) a.big_vae = false;
    ImGui::SetItemTooltip("%s", tr("Den lilla avkodaren (taew) – några minuter snabbare, mjukare bild.", "The small decoder (taew) – minutes faster, a softer picture."));
    if (!a.picture.empty() && !a.turbo) {                     // 10-11: image to video's calm (CFG 3, default) or lively (CFG 5); Turbo has no CFG
        row_label(tr("Rörelse", "Motion"));
        if (chip(tr("Lugn###mc", "Calm###mc"), a.calm)) a.calm = true;
        ImGui::SetItemTooltip("%s", tr("Följer bilden och håller uttryck och bakgrund lugna (CFG 3). Förvalet.",
                                       "Keeps to the picture, expression and background calm (CFG 3). The default."));
        ImGui::SameLine();
        if (chip(tr("Livlig###ml", "Lively###ml"), !a.calm)) a.calm = false;
        ImGui::SetItemTooltip("%s", tr("Följer beskrivningen hårdare – mer rörelse, men uttryck och bakgrund kan dra iväg (CFG 5).",
                                       "Follows the description harder – more motion, but expression and background can drift (CFG 5)."));
    }
    row_label(tr("Steg", "Steps"));
    if (chip("15###s15", a.steps == 15 && !a.turbo)) { a.steps = 15; a.turbo = false; }
    ImGui::SetItemTooltip("%s", tr("Förvalet.", "The default."));
    ImGui::SameLine();
    // 10-13: Turbo in "15 snabb"'s place (its recipe in the comment at FAST_SKIPS)
    if (chip("Turbo###sturbo", a.turbo, a.has_turbo)) { a.steps = TURBO_STEPS; a.turbo = true; }
    ImGui::SetItemTooltip("%s", a.has_turbo ? tr("Turbo-modellen: 4 steg utan CFG – ungefär en tredjedel av tiden. Skarpare och mer rörelse; spegelbilder och bakgrund rör sig mer.",
                                                 "The Turbo model: 4 steps without CFG – about a third of the time. Sharper and more motion; reflections and background move more.")
                                            : tr("Turbo ingår inte i den här versionen.", "Turbo is not part of this version."));
    ImGui::SameLine();
    if (chip("12###s12", a.steps == 12 && !a.turbo)) { a.steps = 12; a.turbo = false; }
    ImGui::SetItemTooltip("%s", tr("Snabbare; ändrar filmen mer än 15.", "Faster; changes the film more than 15."));
    ImGui::SameLine();
    if (chip("10###s10", a.steps == 10 && !a.turbo)) { a.steps = 10; a.turbo = false; }
    ImGui::SetItemTooltip("%s", tr("Snabbast av grundmodellens; ändrar rörelsen mer.", "The base model's fastest; changes the motion more."));
    {
        const Est e = estimate(a.frames, a.big_vae, !a.picture.empty(), a.hd);
        double s = e.start + e.text + e.picture + e.vae + e.write;
        for (int k = 0; k < a.steps; k++) s += !a.turbo && k < cfg_steps(a.steps) ? e.cfg_step : e.one_step;
        char b[200];
        snprintf(b, sizeof b, tr("Tar ungefär %s.", "Takes about %s."), mins(s).c_str());
        ImGui::AlignTextToFramePadding();
        hint(b);
    }
    ImGui::SameLine(px(200));                                        // the seed on the estimate's line
    ImGui::Checkbox(tr("Slumpa fröet", "Random seed"), &a.random_seed);
    if (!a.random_seed) { ImGui::SameLine(); ImGui::SetNextItemWidth(px(120)); ImGui::InputInt("##seed", &a.seed, 0, 0); if (a.seed < 0) a.seed = 0; }
    end_card();

    ImGui::EndDisabled();
    ImGui::EndChild();

    // the action
    if (!a.not_ready.empty()) hint(a.not_ready.c_str(), C_WARN);
    else if (a.running) {
        std::string what;
        double left = 0;
        const float f = a.fraction(what, left);
        char b[220];
        snprintf(b, sizeof b, tr("%s   %s gått, ungefär %s kvar", "%s   %s gone, about %s left"), what.c_str(), mins(now_s() - a.t0).c_str(), mins(left).c_str());
        hint(b, C_TEXT);
        ImGui::ProgressBar(f, ImVec2(-1, px(12)), "");
        hint(tr("Datorn går att använda under tiden. Filmen kan inte avbrytas halvvägs (NPU:n ska stängas rent).",
                "The PC can be used meanwhile. The film cannot be stopped halfway (the NPU has to close cleanly)."));
    } else {
        if (primary_button(tr("Gör filmen", "Make the film"), ImVec2(-1, px(56)))) a.start();
        std::lock_guard<std::mutex> l(a.m);
        if (!a.error.empty()) hint(a.error.c_str(), C_ERR);
    }
}

// ── right: the film ──
static void draw_right(App & a) {
    begin_card("##view", ImVec2(0, 0), nullptr, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const ImVec2 av = ImGui::GetContentRegionAvail();
    const float foot = ImGui::GetFrameHeight() + ImGui::GetTextLineHeightWithSpacing() + px(14);   // the film's line + the buttons, at the bottom
    const float area_h = av.y - foot;
    if (a.srv) {
        const float k = (std::min)(av.x / a.pw, area_h / a.ph);
        const ImVec2 sz(a.pw * k, a.ph * k);
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + (av.x - sz.x) * 0.5f, ImGui::GetCursorPosY() + (area_h - sz.y) * 0.5f));
        ImGui::ImageWithBg(ImTextureRef((ImTextureID) (uintptr_t) a.srv), sz, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, a.running ? 0.45f : 1.0f));
    } else if (a.pic_thumb.srv && a.running) {
        const float k = (std::min)(av.x / a.pic_thumb.w, area_h / a.pic_thumb.h);
        const ImVec2 sz(a.pic_thumb.w * k, a.pic_thumb.h * k);
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + (av.x - sz.x) * 0.5f, ImGui::GetCursorPosY() + (area_h - sz.y) * 0.5f));
        ImGui::ImageWithBg(a.pic_thumb.ref(), sz, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(1, 1, 1, 0.45f));
    } else {
        ImGui::SetCursorPosY(area_h * 0.45f);
        hint(tr("Filmen spelas här när den är klar.", "The film plays here when it is done."));
    }
    ImGui::SetCursorPosY(ImGui::GetWindowHeight() - px(16) - ImGui::GetFrameHeight() - ImGui::GetTextLineHeightWithSpacing() - px(4));
    const Film * cur = a.films.empty() ? nullptr : &a.films[(size_t) (std::min)((int) a.films.size() - 1, a.film_ix)];
    if (cur) {
        std::string info = when_text(cur->when) + "  ·  " +
                           (cur->frames > 49 ? tr("5 s", "5 s") : tr("2 s", "2 s")) + "  ·  " + (cur->hd ? "720p" : "480p") + (cur->picture.empty() ? std::string() : std::string("  ·  ") + (cur->calm ? tr("lugn", "calm") : tr("livlig", "lively"))) +
                           "  ·  " + std::to_string(cur->steps) + tr(" steg", " steps") + (cur->fast ? tr(" snabb", " fast") : "") + (cur->turbo ? " Turbo" : "") +
                           "  ·  " + (cur->big ? tr("skarp", "sharp") : tr("snabb", "fast")) + "  ·  " + mins(cur->secs);
        if (a.film_ix == 0 && !a.frame0.empty()) {
            std::string db = a.frame0;
            if (!g_lang) std::replace(db.begin(), db.end(), '.', ',');      // Swedish decimal comma
            info += std::string("  ·  ") + tr("ruta 0 mot bilden ", "frame 0 vs the picture ") + db;
        }
        hint(info.c_str());
    }
    ImGui::BeginDisabled(!cur);
    if (ImGui::Button(tr("Spela filmen", "Play the film")) && cur) ShellExecuteW(nullptr, L"open", to_w(cur->mp4).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Öppna mappen", "Open the folder"))) ShellExecuteW(nullptr, L"open", to_w(a.out_dir).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if (cur && !a.running) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Använd beskrivningen", "Use the description"))) {
            snprintf(a.prompt, sizeof a.prompt, "%s", cur->prompt.c_str());
            if (!cur->picture.empty() && file_exists(cur->picture)) a.set_picture(cur->picture);
        }
    }
    if (a.films.size() > 1) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(px(260));
        std::string lbl = std::string(tr("Tidigare filmer (", "Earlier films (")) + std::to_string(a.films.size()) + ")";
        if (ImGui::BeginCombo("##films", lbl.c_str())) {
            for (int i = 0; i < (int) a.films.size(); i++) {
                const Film & f = a.films[(size_t) i];
                std::string t = when_text(f.when) + "  " + f.prompt.substr(0, 40);
                if (ImGui::Selectable((t + "##f" + std::to_string(i)).c_str(), i == a.film_ix)) a.film_ix = i;
            }
            ImGui::EndCombo();
        }
    }
    end_card();
}

static void draw_titlebar(HWND hWnd, bool & done, App & a) {
    ImDrawList * dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    const float W = ImGui::GetWindowSize().x, H = px(54);
    g_cap_h = (int) H;
    const float cy = o.y + H * 0.5f;
    dl->AddCircleFilled(ImVec2(o.x + px(30), cy), px(11), ImGui::GetColorU32(ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.28f)), 28);
    dl->AddCircleFilled(ImVec2(o.x + px(30), cy), px(6), ImGui::GetColorU32(C_ACC_H), 24);
    ImGui::PushFont(g_semi, FS * 1.05f);
    const float th = ImGui::GetTextLineHeight();
    ImGui::SetCursorScreenPos(ImVec2(o.x + px(50), cy - th * 0.5f));
    ImGui::TextUnformatted(APP_NAME);
    ImGui::PopFont();
    const float bw = px(46), bx = o.x + W - 3 * bw;
    const float ph = ImGui::GetFrameHeight(), lw_btn = px(104);
    const float lx = bx - lw_btn - px(14);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ph * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.06f));
    ImGui::PushStyleColor(ImGuiCol_Text, C_DIM);
    ImGui::SetCursorScreenPos(ImVec2(lx, cy - ph * 0.5f));
    if (ImGui::Button(g_lang ? "Svenska###lang" : "English###lang", ImVec2(lw_btn, 0))) { g_lang = !g_lang; a.find_things(); a.settings_save(); }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    g_nodrag[0] = { (LONG) (lx - o.x), (LONG) (cy - ph * 0.5f - o.y), (LONG) (lx + lw_btn - o.x), (LONG) (cy + ph * 0.5f - o.y) };
    g_nodrag[1] = { (LONG) (bx - o.x), 0, (LONG) W, (LONG) H };
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.09f)); ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.16f));
    const ImU32 ic = ImGui::GetColorU32(C_TEXT);
    const float k = px(5), lw = px(1.2f);
    ImGui::SetCursorScreenPos(ImVec2(bx, o.y));
    if (ImGui::Button("##min", ImVec2(bw, H))) ShowWindow(hWnd, SW_MINIMIZE);
    { const ImVec2 c(bx + bw * 0.5f, cy); dl->AddLine(ImVec2(c.x - k, c.y), ImVec2(c.x + k, c.y), ic, lw); }
    ImGui::SetCursorScreenPos(ImVec2(bx + bw, o.y));
    {
        const bool mx = IsZoomed(hWnd) != 0;
        if (ImGui::Button("##max", ImVec2(bw, H))) ShowWindow(hWnd, mx ? SW_RESTORE : SW_MAXIMIZE);
        const ImVec2 c(bx + bw * 1.5f, cy);
        if (mx) {
            dl->AddRect(ImVec2(c.x - k + px(2), c.y - k), ImVec2(c.x + k, c.y + k - px(2)), ic, px(1.5f), 0, lw);
            dl->AddRectFilled(ImVec2(c.x - k, c.y - k + px(2)), ImVec2(c.x + k - px(2), c.y + k), ImGui::GetColorU32(C_BG), px(1.5f));
            dl->AddRect(ImVec2(c.x - k, c.y - k + px(2)), ImVec2(c.x + k - px(2), c.y + k), ic, px(1.5f), 0, lw);
        } else dl->AddRect(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), ic, px(1.5f), 0, lw);
    }
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f, 0.24f, 0.24f, 0.90f));
    ImGui::SetCursorScreenPos(ImVec2(bx + 2 * bw, o.y));
    if (ImGui::Button("##close", ImVec2(bw, H))) done = true;
    { const ImVec2 c(bx + bw * 2.5f, cy); dl->AddLine(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), ic, lw); dl->AddLine(ImVec2(c.x - k, c.y + k), ImVec2(c.x + k, c.y - k), ic, lw); }
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
}

static bool save_backbuffer(const std::string & path) {           // the window as a PNG (test hook --shot)
    ID3D11Texture2D * back = nullptr, * stg = nullptr;
    if (FAILED(g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back))) || !back) return false;
    D3D11_TEXTURE2D_DESC d;
    back->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0;
    bool ok = false;
    if (SUCCEEDED(g_pd3dDevice->CreateTexture2D(&d, nullptr, &stg)) && stg) {
        g_pd3dDeviceContext->CopyResource(stg, back);
        D3D11_MAPPED_SUBRESOURCE mp;
        if (SUCCEEDED(g_pd3dDeviceContext->Map(stg, 0, D3D11_MAP_READ, 0, &mp))) {
            fluximg::Rgb im;
            im.w = (int) d.Width; im.h = (int) d.Height;
            im.px.resize((size_t) im.w * im.h * 3);
            for (int y = 0; y < im.h; y++) {
                const uint8_t * s = (const uint8_t *) mp.pData + (size_t) y * mp.RowPitch;
                for (int x = 0; x < im.w; x++) { im.px[((size_t) y * im.w + x) * 3] = s[x * 4]; im.px[((size_t) y * im.w + x) * 3 + 1] = s[x * 4 + 1]; im.px[((size_t) y * im.w + x) * 3 + 2] = s[x * 4 + 2]; }
            }
            g_pd3dDeviceContext->Unmap(stg, 0);
            std::thread t([&]() { try { fluximg::save_png(path, im); ok = true; } catch (...) {} });
            t.join();
        }
        stg->Release();
    }
    back->Release();
    return ok;
}

// ════════════════════════ main ════════════════════════
int main(int argc, char ** argv) {
    std::string shot, demo, demo_image, demo_swap;
    int lang_arg = -1, shot_w = 0, shot_h = 0;
    bool demo_cpu = false;
    for (int i = 1; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--shot" && i + 1 < argc) shot = argv[++i];
        else if (s == "--demo-run" && i + 1 < argc) demo = argv[++i];
        else if (s == "--demo-cpu") demo_cpu = true;
        else if (s == "--demo-image" && i + 1 < argc) demo_image = argv[++i];
        else if (s == "--demo-swap" && i + 1 < argc) demo_swap = argv[++i];          // with --demo-image: change the picture mid-frame
        else if (s == "--lang" && i + 1 < argc) lang_arg = std::string(argv[++i]) == "en" ? 1 : 0;
        else if (s == "--window" && i + 2 < argc) { shot_w = atoi(argv[++i]); shot_h = atoi(argv[++i]); }
    }
    WNDCLASSEXW wc{ sizeof(wc), CS_CLASSDC, WndProc, 0, 0, GetModuleHandle(nullptr), nullptr, LoadCursor(nullptr, IDC_ARROW), nullptr, nullptr, APP_CLASS_W, nullptr };
    wc.hIcon = (HICON) LoadImageW(nullptr, to_w(app_dir() + "\\" APP_ICON).c_str(), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);
    wc.hIconSm = (HICON) LoadImageW(nullptr, to_w(app_dir() + "\\" APP_ICON).c_str(), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_LOADFROMFILE);
    RegisterClassExW(&wc);
    g_scale = GetDpiForSystem() / 96.0f;
    RECT wa{ 0, 0, 1280, 1024 };
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
    const int waW = wa.right - wa.left, waH = wa.bottom - wa.top;
    const int winW = shot_w ? shot_w : (std::min)((int) (1400 * g_scale), waW - (int) (24 * g_scale));
    const int winH = shot_h ? shot_h : (std::min)((int) (900 * g_scale), waH - (int) (24 * g_scale));
    const int winX = wa.left + (waW - winW) / 2, winY = wa.top + (waH - winH) / 2;
    HWND hWnd = CreateWindowExW(WS_EX_APPWINDOW | WS_EX_ACCEPTFILES, wc.lpszClassName, APP_NAME_W,
                                WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU,
                                winX, winY, winW, winH, nullptr, nullptr, wc.hInstance, nullptr);
    {
        const DWORD round = 2;
        DwmSetWindowAttribute(hWnd, 33, &round, sizeof(round));
        const BOOL dark = TRUE;
        DwmSetWindowAttribute(hWnd, 20, &dark, sizeof(dark));
        const COLORREF edge = RGB(84, 90, 101);
        DwmSetWindowAttribute(hWnd, 34, &edge, sizeof(edge));
    }
    g_scale = GetDpiForWindow(hWnd) / 96.0f;
    if (!CreateDeviceD3D(hWnd)) { CleanupDeviceD3D(); UnregisterClassW(wc.lpszClassName, wc.hInstance); return 1; }
    ShowWindow(hWnd, shot.empty() ? SW_SHOWDEFAULT : SW_SHOWNOACTIVATE);
    UpdateWindow(hWnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();
    io.IniFilename = nullptr;
    {
        std::string fonts = known_folder(FOLDERID_Fonts);
        if (fonts.empty()) fonts = "C:\\Windows\\Fonts";
        g_font = io.Fonts->AddFontFromFileTTF((fonts + "\\segoeui.ttf").c_str(), FS);
        if (!g_font) g_font = io.Fonts->AddFontDefault();
        g_semi = io.Fonts->AddFontFromFileTTF((fonts + "\\seguisb.ttf").c_str(), FS);
        if (!g_semi) g_semi = g_font;
    }
    apply_style();
    ImGui_ImplWin32_Init(hWnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    static App app;
    app.find_things();
    app.settings_load();
    if (lang_arg >= 0) g_lang = lang_arg;
    app.find_things();                                     // again: the messages in the chosen language
    app.films_load();
    app.preview_meta_load();
    app.demo_cpu = demo_cpu;
    app.test_run = !shot.empty() || !demo.empty();
    if (!demo.empty()) snprintf(app.prompt, sizeof app.prompt, "%s", demo.c_str());
    if (!demo_image.empty()) app.set_picture(demo_image);

    bool done = false, quit_asked = false, demo_started = false;
    int frame = 0, frames_after = 0, exit_code = 0;
    while (!done) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); if (msg.message == WM_QUIT) quit_asked = true; }
        if (IsIconic(hWnd) && shot.empty()) { Sleep(120); app.pump(); if (quit_asked && !app.running) break; continue; }
        if (GetForegroundWindow() != hWnd && shot.empty()) Sleep(30);
        if (g_new_dpi) { g_scale = g_new_dpi / 96.0f; g_new_dpi = 0; apply_style(); }
        for (const std::string & f : g_dropped) if (!app.running && is_picture_file(f)) { app.set_picture(f); break; }
        g_dropped.clear();
        app.pump();
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        const ImGuiViewport * vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin(APP_NAME, nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
                                               | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar();
        bool close_clicked = false;
        draw_titlebar(hWnd, close_clicked, app);
        if (close_clicked) quit_asked = true;
        const float mg = px(18);
        ImGui::SetCursorPos(ImVec2(mg, (float) g_cap_h + px(2)));
        ImGui::BeginChild("##page", ImVec2(ImGui::GetWindowSize().x - 2 * mg, ImGui::GetWindowSize().y - g_cap_h - px(2) - mg), 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            const ImVec2 av = ImGui::GetContentRegionAvail();
            const float leftw = (std::min)(px(480), av.x * 0.42f);
            ImGui::BeginChild("##left", ImVec2(leftw, av.y), 0, ImGuiWindowFlags_NoScrollbar);
            draw_left(app, hWnd);
            ImGui::EndChild();
            ImGui::SameLine(0, px(16));
            ImGui::BeginChild("##right", ImVec2(0, av.y), 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            draw_right(app);
            ImGui::EndChild();
        }
        ImGui::EndChild();
        if (quit_asked && app.running) {                   // never leave an engine behind: the window waits for the film
            const char * t = tr("Avslutar när filmen är klar …", "Closing when the film is done …");
            const ImVec2 ts = ImGui::CalcTextSize(t);
            ImGui::GetForegroundDrawList()->AddText(ImVec2(vp->Pos.x + px(200), vp->Pos.y + (g_cap_h - ts.y) * 0.5f), ImGui::GetColorU32(C_WARN), t);
        }
        ImGui::End();
        ImGui::Render();
        const float clr[4] = { C_BG.x, C_BG.y, C_BG.z, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        frame++;
        if (frame == 10 && !demo_swap.empty()) g_test_swap = demo_swap;       // drawn on frame 11, inside the card
        if (!demo.empty() && !demo_started && frame > 5) {
            demo_started = true;
            app.start();
            if (!app.running) { std::lock_guard<std::mutex> l(app.m); fprintf(stderr, "demo: nothing started (%s%s)\n", app.error.c_str(), app.not_ready.c_str()); exit_code = 3; quit_asked = true; }
        }
        if (!shot.empty()) {
            const bool settled = demo.empty() ? frame > 40 : (demo_started && !app.running && ++frames_after > 30);
            if (!demo.empty() && app.running && now_s() - app.t0 > 6.0 && !file_exists(shot + ".mid.png")) save_backbuffer(shot + ".mid.png");
            if (settled) {
                if (!save_backbuffer(shot)) exit_code = 4;
                std::lock_guard<std::mutex> l(app.m);
                if (!app.error.empty()) { fprintf(stderr, "demo: %s\n", app.error.c_str()); exit_code = 5; }
                quit_asked = true;
            }
        }
        g_pSwapChain->Present(1, 0);
        release_frame_end();                               // textures given up during this frame: no draw list holds them now
        if (quit_asked && !app.running) done = true;
    }
    app.settings_save();
    if (app.th.joinable()) app.th.join();
    app.drop_preview();
    app.pic_thumb.drop();
    release_frame_end();                                   // no frame is drawn any more
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hWnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return exit_code;
}
