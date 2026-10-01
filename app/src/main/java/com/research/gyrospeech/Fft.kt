package com.research.gyrospeech

import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** 极简 radix-2 Cooley-Tukey FFT，用于把窄带传感器信号变成幅度谱。 */
object Fft {

    /** 就地复数 FFT。re/im 长度必须是 2 的幂。 */
    private fun transform(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        if (n == 1) return
        // 位反转置换
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
        }
        // 蝶形运算
        var len = 2
        while (len <= n) {
            val ang = -2.0 * Math.PI / len
            val wRe = cos(ang)
            val wIm = sin(ang)
            var i = 0
            while (i < n) {
                var curRe = 1.0
                var curIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
                    val vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
                    re[i + k] = uRe + vRe
                    im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe
                    im[i + k + len / 2] = uIm - vIm
                    val nRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = nRe
                }
                i += len
            }
            len = len shl 1
        }
    }

    /**
     * 输入实数信号（会先去均值 + 汉宁窗），返回单边幅度谱（长度 n/2）。
     */
    fun magnitudeSpectrum(input: FloatArray): FloatArray {
        val n = input.size
        val re = DoubleArray(n)
        val im = DoubleArray(n)
        // 去直流
        var mean = 0.0
        for (v in input) mean += v
        mean /= n
        // 汉宁窗
        for (i in 0 until n) {
            val w = 0.5 - 0.5 * cos(2.0 * Math.PI * i / (n - 1))
            re[i] = (input[i] - mean) * w
        }
        transform(re, im)
        val half = n / 2
        val mag = FloatArray(half)
        for (i in 0 until half) {
            mag[i] = sqrt(re[i] * re[i] + im[i] * im[i]).toFloat() / half
        }
        return mag
    }
}
