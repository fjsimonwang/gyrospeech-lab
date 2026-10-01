package com.research.gyrospeech

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.exp

/**
 * 端上最近质心分类器（nearest-centroid，余弦相似度）。
 *
 * - 每个类别保存若干条 L2 归一化特征向量（录入阶段采集）。
 * - train() 计算各类质心。
 * - predict() 用余弦相似度 + softmax 输出各类置信度。
 * - 模型持久化到 filesDir 的 JSON，重启后仍在。
 *
 * 之所以不用大型神经网络：没有可用的公开"陀螺仪语音"数据集，
 * 必须靠用户自己在本机录入的少量样本训练，最近质心在小样本下最稳、可解释、零依赖。
 */
class Classifier(context: Context, private val modelKey: String) {

    private val file = File(context.filesDir, "model_$modelKey.json")
    private val samples = LinkedHashMap<String, MutableList<FloatArray>>()
    private val centroids = LinkedHashMap<String, FloatArray>()
    private val softmaxTemp = 12f

    init { load() }

    data class Prediction(val best: String, val confidence: Float, val probs: LinkedHashMap<String, Float>)

    fun labels(): List<String> = samples.keys.toList()
    fun sampleCount(label: String): Int = samples[label]?.size ?: 0
    fun readyClasses(): Int = samples.count { it.value.isNotEmpty() }
    fun isReady(): Boolean = readyClasses() >= 2

    fun ensureLabel(label: String) { samples.getOrPut(label) { mutableListOf() } }

    fun addSample(label: String, vec: FloatArray) {
        samples.getOrPut(label) { mutableListOf() }.add(vec.copyOf())
    }

    fun clearLabel(label: String) {
        samples[label]?.clear()
        centroids.remove(label)
        save()
    }

    fun clearAll() {
        samples.clear()
        centroids.clear()
        save()
    }

    /** 重新计算质心并持久化。 */
    fun train() {
        centroids.clear()
        for ((label, list) in samples) {
            if (list.isEmpty()) continue
            val dim = list[0].size
            val c = FloatArray(dim)
            for (v in list) for (i in 0 until dim) c[i] += v[i]
            for (i in 0 until dim) c[i] /= list.size
            // 质心再归一化，便于余弦相似度
            var n = 0f
            for (x in c) n += x * x
            n = kotlin.math.sqrt(n)
            if (n > 1e-6f) for (i in c.indices) c[i] /= n
            centroids[label] = c
        }
        save()
    }

    fun predict(vec: FloatArray): Prediction? {
        if (centroids.size < 2) return null
        val sims = LinkedHashMap<String, Float>()
        for ((label, c) in centroids) {
            var dot = 0f
            val n = minOf(c.size, vec.size)
            for (i in 0 until n) dot += c[i] * vec[i]
            sims[label] = dot // 两者均已 L2 归一化 → dot = 余弦相似度
        }
        // softmax
        var maxSim = Float.NEGATIVE_INFINITY
        for (s in sims.values) if (s > maxSim) maxSim = s
        var sum = 0f
        val probs = LinkedHashMap<String, Float>()
        for ((label, s) in sims) { val e = exp(((s - maxSim) * softmaxTemp).toDouble()).toFloat(); probs[label] = e; sum += e }
        for (label in probs.keys) probs[label] = probs[label]!! / sum
        val best = probs.maxByOrNull { it.value }!!
        return Prediction(best.key, best.value, probs)
    }

    // ---------- 持久化 ----------
    private fun save() {
        try {
            val root = JSONObject()
            for ((label, list) in samples) {
                val arr = JSONArray()
                for (v in list) {
                    val row = JSONArray()
                    for (x in v) row.put(x.toDouble())
                    arr.put(row)
                }
                root.put(label, arr)
            }
            file.writeText(root.toString())
        } catch (_: Exception) {}
    }

    private fun load() {
        try {
            if (!file.exists()) return
            val root = JSONObject(file.readText())
            val keys = root.keys()
            while (keys.hasNext()) {
                val label = keys.next()
                val arr = root.getJSONArray(label)
                val list = mutableListOf<FloatArray>()
                for (i in 0 until arr.length()) {
                    val row = arr.getJSONArray(i)
                    val v = FloatArray(row.length())
                    for (j in 0 until row.length()) v[j] = row.getDouble(j).toFloat()
                    list.add(v)
                }
                samples[label] = list
            }
            train()
        } catch (_: Exception) {}
    }
}
