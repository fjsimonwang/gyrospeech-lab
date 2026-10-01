package com.research.gyrospeech

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.io.File
import java.io.Writer
import kotlin.math.sqrt

/**
 * 采集单个运动传感器（陀螺仪或加速度计）的三轴数据。
 * - 维护一个环形缓冲区供可视化（幅度信号）。
 * - 测量真实到达采样率（Hz）。
 * - 可选：把带时间戳的原始三轴写入 CSV，供离线训练用。
 */
class SensorRecorder(context: Context) : SensorEventListener {

    enum class Kind(val type: Int, val label: String) {
        GYRO(Sensor.TYPE_GYROSCOPE, "Gyroscope"),
        ACCEL(Sensor.TYPE_ACCELEROMETER, "Accelerometer")
    }

    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val appContext = context.applicationContext

    val bufferSize = 512
    private val ring = FloatArray(bufferSize)
    private var writeIdx = 0

    @Volatile var measuredHz: Float = 0f
        private set
    @Volatile var sampleCount: Long = 0L
        private set

    private var lastTimestampNs = 0L
    private var rateAccum = 0.0
    private var rateSamples = 0

    private var csvWriter: Writer? = null
    @Volatile var recording = false
        private set
    var lastFile: File? = null
        private set

    // M4：把幅度样本连续采集到一个可增长缓冲，供重建/成对数据
    private val captureList = ArrayList<Float>()
    @Volatile var capturing = false
        private set

    fun startCapture() {
        synchronized(captureList) { captureList.clear() }
        capturing = true
    }

    /** 停止并返回本次采集的幅度时序。 */
    fun stopCapture(): FloatArray {
        capturing = false
        synchronized(captureList) { return captureList.toFloatArray() }
    }

    fun captureCount(): Int = synchronized(captureList) { captureList.size }

    var currentKind: Kind = Kind.GYRO
        private set

    fun hasSensor(kind: Kind): Boolean = sm.getDefaultSensor(kind.type) != null

    fun start(kind: Kind) {
        stop(keepFile = true)
        currentKind = kind
        val sensor = sm.getDefaultSensor(kind.type) ?: return
        // SENSOR_DELAY_FASTEST：请求硬件最高速率（Android 12+ 未持权限时系统封顶 200Hz）
        sm.registerListener(this, sensor, SensorManager.SENSOR_DELAY_FASTEST)
        lastTimestampNs = 0L
        rateAccum = 0.0
        rateSamples = 0
    }

    fun stop(keepFile: Boolean = false) {
        sm.unregisterListener(this)
        if (!keepFile) stopRecording()
    }

    /** 快照当前波形（按时间顺序展开的环形缓冲区）。 */
    fun snapshot(out: FloatArray) {
        val n = out.size.coerceAtMost(bufferSize)
        synchronized(ring) {
            for (i in 0 until n) {
                val idx = (writeIdx + i) % bufferSize
                out[i] = ring[idx]
            }
        }
    }

    fun startRecording(): File {
        val dir = appContext.getExternalFilesDir(null) ?: appContext.filesDir
        val f = File(dir, "sensor_${currentKind.name}_${System.currentTimeMillis()}.csv")
        val w = f.bufferedWriter()
        w.write("timestamp_ns,x,y,z\n")
        csvWriter = w
        lastFile = f
        recording = true
        return f
    }

    fun stopRecording() {
        recording = false
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (_: Exception) {
        }
        csvWriter = null
    }

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]

        // 幅度信号（用于可视化；FFT 内部会再去直流）
        val mag = sqrt(x * x + y * y + z * z)
        synchronized(ring) {
            ring[writeIdx] = mag
            writeIdx = (writeIdx + 1) % bufferSize
        }
        if (capturing) synchronized(captureList) { captureList.add(mag) }
        sampleCount++

        // 真实采样率（对相邻事件间隔做指数平滑）
        if (lastTimestampNs != 0L) {
            val dt = (event.timestamp - lastTimestampNs) / 1e9
            if (dt > 0) {
                rateAccum += 1.0 / dt
                rateSamples++
                if (rateSamples >= 20) {
                    measuredHz = (rateAccum / rateSamples).toFloat()
                    rateAccum = 0.0
                    rateSamples = 0
                }
            }
        }
        lastTimestampNs = event.timestamp

        // 写 CSV
        if (recording) {
            try {
                csvWriter?.write("${event.timestamp},$x,$y,$z\n")
            } catch (_: Exception) {
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
