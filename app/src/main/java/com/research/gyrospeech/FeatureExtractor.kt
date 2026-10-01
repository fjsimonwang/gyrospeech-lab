package com.research.gyrospeech

import kotlin.math.ln
import kotlin.math.sqrt

/**
 * 把一个传感器信号窗口变成定长特征向量，供分类器使用。
 *
 * 设计要点（受 <100Hz 有效带宽约束）：
 *  - 取 FFT 幅度谱，压成 32 个频带能量（对数刻度）—— 描述"频谱形状"，对说话人/性别敏感。
 *  - 追加 4 个派生特征：谱质心、谱带宽、主频、低/高频能量比。
 *  - 对整条向量做 L2 归一化 —— 抵消响度差异，只保留"形状"，让近场/远场都可比。
 */
object FeatureExtractor {

    const val DIM = 36
    private const val N_BINS = 32

    /** 窗口的 RMS（去均值后），用作有声/静音门限。 */
    fun energy(window: FloatArray): Float {
        if (window.isEmpty()) return 0f
        var mean = 0f
        for (v in window) mean += v
        mean /= window.size
        var s = 0f
        for (v in window) { val d = v - mean; s += d * d }
        return sqrt(s / window.size)
    }

    /**
     * @param window 时域幅度信号（长度应为 2 的幂，如 256）
     * @param sampleRateHz 当前真实采样率，用于把频带换算成 Hz
     */
    fun extract(window: FloatArray, sampleRateHz: Float): FloatArray {
        val spec = Fft.magnitudeSpectrum(window)   // 长度 = window/2
        val half = spec.size
        val out = FloatArray(DIM)

        // 32 个对数频带能量
        val per = (half / N_BINS).coerceAtLeast(1)
        for (b in 0 until N_BINS) {
            var acc = 0f
            var cnt = 0
            val start = b * per
            val end = if (b == N_BINS - 1) half else (start + per)
            for (i in start until end.coerceAtMost(half)) { acc += spec[i]; cnt++ }
            val avg = if (cnt > 0) acc / cnt else 0f
            out[b] = ln(1f + avg)
        }

        // 派生特征
        val hzPerBin = if (sampleRateHz > 0) sampleRateHz / (2f * half) else 1f
        var sumMag = 0f
        var centroidNum = 0f
        var dominantBin = 0
        var dominantVal = 0f
        var lowE = 0f   // 低半带能量
        var highE = 0f  // 高半带能量
        for (i in 0 until half) {
            val m = spec[i]
            sumMag += m
            centroidNum += i * m
            if (m > dominantVal) { dominantVal = m; dominantBin = i }
            if (i < half / 2) lowE += m else highE += m
        }
        val centroidBin = if (sumMag > 0) centroidNum / sumMag else 0f
        var spread = 0f
        if (sumMag > 0) {
            var num = 0f
            for (i in 0 until half) { val d = i - centroidBin; num += d * d * spec[i] }
            spread = sqrt(num / sumMag)
        }
        val nyq = sampleRateHz / 2f
        out[N_BINS + 0] = if (nyq > 0) (centroidBin * hzPerBin) / nyq else 0f          // 归一化谱质心
        out[N_BINS + 1] = if (half > 0) spread / half else 0f                          // 归一化带宽
        out[N_BINS + 2] = if (nyq > 0) (dominantBin * hzPerBin) / nyq else 0f          // 归一化主频（近似基频）
        out[N_BINS + 3] = if (highE + lowE > 0) lowE / (highE + lowE) else 0f          // 低/高能量比

        // L2 归一化
        var norm = 0f
        for (v in out) norm += v * v
        norm = sqrt(norm)
        if (norm > 1e-6f) for (i in out.indices) out[i] /= norm
        return out
    }

    /** 主频估计（Hz），用于 UI 显示（男声≈85–155，女声≈165–255，受采样率封顶影响）。 */
    fun dominantHz(window: FloatArray, sampleRateHz: Float): Float {
        val spec = Fft.magnitudeSpectrum(window)
        var bin = 0; var mx = 0f
        for (i in spec.indices) if (spec[i] > mx) { mx = spec[i]; bin = i }
        return if (sampleRateHz > 0) bin * sampleRateHz / (2f * spec.size) else 0f
    }
}
