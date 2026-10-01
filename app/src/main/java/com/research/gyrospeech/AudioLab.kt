package com.research.gyrospeech

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaPlayer
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * M4 音频引擎：
 *  1) playProbe —— 用扬声器外放一段"类语音"探测音（多载波 + 慢包络调制 + 停顿），
 *     给加速度计一个结构化、可复现的声源，制造机械耦合。
 *  2) reconstructToWav —— 把加速度信号按帧提取"响度包络 + 主频(音高)"，
 *     以连续相位重合成到 16kHz WAV，得到可听、跟随原音韵律/音高的重建（DSP 基线，非逐字）。
 *  3) play —— 播放任意 WAV。
 */
object AudioLab {

    private const val OUT_RATE = 16000

    // ---------- 探测音播放 ----------
    /** 阻塞式播放，时长 durationMs。应在后台线程调用。 */
    fun playProbe(context: Context, durationMs: Int) {
        val rate = 44100
        val total = rate * durationMs / 1000
        val buf = ShortArray(total)
        // 三个载波，模拟语音共振峰；被 4Hz 慢包络调制，并每隔约 0.8s 制造停顿
        val f = doubleArrayOf(180.0, 350.0, 620.0)
        for (n in 0 until total) {
            val t = n.toDouble() / rate
            val env = (0.5 + 0.5 * sin(2 * PI * 4.0 * t))   // 4Hz 节奏
            val gate = if ((t % 0.8) < 0.6) 1.0 else 0.0     // 停顿
            var s = 0.0
            for (fc in f) s += sin(2 * PI * fc * t)
            s = s / f.size * env * gate
            buf[n] = (s * 0.9 * Short.MAX_VALUE).toInt().toShort()
        }

        // 尽量把媒体音量调高以增强耦合
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val maxVol = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        try { am.setStreamVolume(AudioManager.STREAM_MUSIC, maxVol, 0) } catch (_: Exception) {}

        val minBuf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(rate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuf, total * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        track.write(buf, 0, buf.size)
        track.play()
        // 等待播放完成
        Thread.sleep((durationMs + 200).toLong())
        try { track.stop(); track.release() } catch (_: Exception) {}
    }

    // ---------- 加速度 → 可听 WAV 重建 ----------
    /**
     * @param accel 加速度幅度时序（去均值前的 |a|）
     * @param rateHz 采集真实采样率
     * @return 生成的 WAV 文件（16kHz 单声道）
     */
    fun reconstructToWav(accel: FloatArray, rateHz: Float, outFile: File): Float {
        val rate = if (rateHz > 1f) rateHz else 200f
        val n = accel.size
        if (n < 16) { writeWav(ShortArray(0), OUT_RATE, outFile); return 0f }

        // 去均值
        var mean = 0f
        for (v in accel) mean += v
        mean /= n
        val sig = FloatArray(n) { accel[it] - mean }

        // 分帧：帧长≈50ms，跳步≈25ms
        val frameLen = (rate * 0.05f).toInt().coerceAtLeast(8)
        val hop = (frameLen / 2).coerceAtLeast(1)
        val nFrames = ((n - frameLen) / hop).coerceAtLeast(1)

        val amp = FloatArray(nFrames)
        val pitch = FloatArray(nFrames)
        // FFT 帧补零到 2 的幂
        var pow = 1; while (pow < frameLen) pow = pow shl 1
        val fftBuf = FloatArray(pow)

        var maxAmp = 1e-6f
        for (k in 0 until nFrames) {
            val start = k * hop
            var rms = 0f
            for (i in 0 until frameLen) { val v = sig[start + i]; rms += v * v; fftBuf[i] = v }
            for (i in frameLen until pow) fftBuf[i] = 0f
            rms = sqrt(rms / frameLen)
            amp[k] = rms
            if (rms > maxAmp) maxAmp = rms
            // 主频
            val spec = Fft.magnitudeSpectrum(fftBuf)
            var bin = 1; var mx = 0f
            for (i in 1 until spec.size) if (spec[i] > mx) { mx = spec[i]; bin = i }
            pitch[k] = bin * rate / (2f * spec.size) // 0..rate/2 Hz
        }
        for (k in 0 until nFrames) amp[k] /= maxAmp

        // 合成：连续相位，逐样本插值帧参数；把 <100Hz 音高上移到清晰可听区
        val durSec = n / rate
        val outN = (durSec * OUT_RATE).toInt().coerceAtLeast(1)
        val out = FloatArray(outN)
        var phase = 0.0
        val pitchScale = 6f   // 100Hz -> 600Hz，进入清晰可听区
        for (t in 0 until outN) {
            val fpos = t.toFloat() / OUT_RATE / hop * rate // 对应帧位置
            val k0 = fpos.toInt().coerceIn(0, nFrames - 1)
            val k1 = (k0 + 1).coerceAtMost(nFrames - 1)
            val frac = (fpos - k0).coerceIn(0f, 1f)
            val a = amp[k0] * (1 - frac) + amp[k1] * frac
            val pHz = (pitch[k0] * (1 - frac) + pitch[k1] * frac) * pitchScale
            val fClamped = pHz.coerceIn(120f, 3000f)
            phase += 2 * PI * fClamped / OUT_RATE
            out[t] = (a * sin(phase)).toFloat()
        }

        // 归一化 + 转 int16
        var pk = 1e-6f
        for (v in out) { val av = if (v < 0) -v else v; if (av > pk) pk = av }
        val pcm = ShortArray(outN)
        val g = 0.9f / pk
        for (i in 0 until outN) pcm[i] = (out[i] * g * Short.MAX_VALUE).toInt().toShort()
        writeWav(pcm, OUT_RATE, outFile)
        return durSec
    }

    // ---------- WAV 写入 ----------
    private fun writeWav(pcm: ShortArray, sampleRate: Int, file: File) {
        val byteRate = sampleRate * 2
        val dataLen = pcm.size * 2
        val raf = RandomAccessFile(file, "rw")
        raf.setLength(0)
        fun w(s: String) = raf.writeBytes(s)
        fun i32(v: Int) { raf.write(v and 0xff); raf.write((v shr 8) and 0xff); raf.write((v shr 16) and 0xff); raf.write((v shr 24) and 0xff) }
        fun i16(v: Int) { raf.write(v and 0xff); raf.write((v shr 8) and 0xff) }
        w("RIFF"); i32(36 + dataLen); w("WAVE")
        w("fmt "); i32(16); i16(1); i16(1); i32(sampleRate); i32(byteRate); i16(2); i16(16)
        w("data"); i32(dataLen)
        val bytes = ByteArray(dataLen)
        for (i in pcm.indices) { bytes[i * 2] = (pcm[i].toInt() and 0xff).toByte(); bytes[i * 2 + 1] = ((pcm[i].toInt() shr 8) and 0xff).toByte() }
        raf.write(bytes)
        raf.close()
    }

    // ---------- 播放 WAV ----------
    private var player: MediaPlayer? = null
    fun play(file: File) {
        try {
            player?.release()
            player = MediaPlayer().apply {
                setDataSource(file.absolutePath)
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build()
                )
                prepare(); start()
            }
        } catch (_: Exception) {}
    }
}
