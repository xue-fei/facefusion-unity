using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

public class FaceDetector : MonoBehaviour
{
    public ModelAsset modelAsset;
    public Texture2D inputTexture;

    [Range(0.1f, 1.0f)]
    public float scoreThreshold = 0.5f;
    [Range(0.1f, 1.0f)]
    public float nmsThreshold = 0.4f;

    private readonly string[] SCORE_NAMES = { "448", "471", "494" };
    private readonly string[] BBOX_NAMES = { "451", "474", "497" };
    private readonly string[] LM_NAMES = { "454", "477", "500" };

    const int INPUT_SIZE = 640;

    void Start()
    {
        if (modelAsset == null || inputTexture == null)
        {
            Debug.LogError("Missing model or input texture.");
            return;
        }

        var model = ModelLoader.Load(modelAsset);
        using var worker = new Worker(model, BackendType.GPUCompute);

        RunDetection(worker, inputTexture);
    }

    void RunDetection(Worker worker, Texture2D texture)
    {
        Tensor<float> input = Preprocess(texture);
        worker.Schedule(input); 

        var allFaces = new List<FaceResult>();

        for (int i = 0; i < 3; i++)
        {
            var scoreTensor = worker.PeekOutput(SCORE_NAMES[i]).ReadbackAndClone() as Tensor<float>;
            var bboxTensor = worker.PeekOutput(BBOX_NAMES[i]).ReadbackAndClone() as Tensor<float>;
            var lmTensor = worker.PeekOutput(LM_NAMES[i]).ReadbackAndClone() as Tensor<float>;

            // 直接解析 [N, C] 格式
            var faces = DecodeFlattened(scoreTensor, bboxTensor, lmTensor);
            allFaces.AddRange(faces);

            scoreTensor.Dispose();
            bboxTensor.Dispose();
            lmTensor.Dispose();
        }

        // 映射回原图
        float scaleX = (float)texture.width / INPUT_SIZE;
        float scaleY = (float)texture.height / INPUT_SIZE;
        for (int i = 0; i < allFaces.Count; i++)
        {
            var f = allFaces[i];
            f.x *= scaleX;
            f.y *= scaleY;
            f.width *= scaleX;
            f.height *= scaleY;
            for (int j = 0; j < 5; j++)
            {
                f.landmarks[j].x *= scaleX;
                f.landmarks[j].y *= scaleY;
            }
            allFaces[i] = f;
        }

        // NMS
        allFaces = ApplyNMS(allFaces, nmsThreshold);

        foreach (var face in allFaces)
        {
            if (face.score >= scoreThreshold)
            {
                Debug.Log($"✅ Face: x={face.x:F1}, y={face.y:F1}, w={face.width:F1}, h={face.height:F1}, score={face.score:F3}");
            }
        }

        input.Dispose();
    }

    Tensor<float> Preprocess(Texture2D tex)
    {
        // Resize to 640x640
        RenderTexture rt = RenderTexture.GetTemporary(INPUT_SIZE, INPUT_SIZE);
        Graphics.Blit(tex, rt);

        Texture2D resized = new Texture2D(INPUT_SIZE, INPUT_SIZE, TextureFormat.RGB24, false);
        RenderTexture.active = rt;
        resized.ReadPixels(new Rect(0, 0, INPUT_SIZE, INPUT_SIZE), 0, 0);
        resized.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = resized.GetPixels32();
        float[] data = new float[3 * INPUT_SIZE * INPUT_SIZE];
        for (int i = 0; i < pixels.Length; i++)
        {
            // 尝试 [0,1] 归一化（若检测不到，改用 pixels[i].r 直接传 0~255）
            data[i] = pixels[i].r / 255.0f;
            data[i + INPUT_SIZE * INPUT_SIZE] = pixels[i].g / 255.0f;
            data[i + 2 * INPUT_SIZE * INPUT_SIZE] = pixels[i].b / 255.0f;
        }

        return new Tensor<float>(new TensorShape(1, 3, INPUT_SIZE, INPUT_SIZE), data);
    }

    // ✅ 核心：解析 [N, C] 格式输出
    List<FaceResult> DecodeFlattened(Tensor<float> scores, Tensor<float> bboxes, Tensor<float> landmarks)
    {
        var results = new List<FaceResult>();
        long N = scores.shape[0]; // 检测数量 
        Debug.Log(N);
        for (int i = 0; i < N; i++)
        {
            float score = scores[i, 0]; // 注意：这个 score 可能已经是 sigmoid 后的值！

            // ⚠️ 重要：有些导出模型输出的是 logits（需 sigmoid），有些是 prob（无需）
            // 先假设是 prob（0~1）。如果 score 全 >0.9，可能是 logits → 需 sigmoid
            // 如果 score 全 ≈0.5，可能是 logits → 取消下面注释：
            //score = 1.0f / (1.0f + Mathf.Exp(-score));

            if (score < scoreThreshold) continue;

            // bbox: 通常为 [x1, y1, x2, y2]（绝对坐标，640x640 空间）
            float x1 = bboxes[i, 0];
            float y1 = bboxes[i, 1];
            float x2 = bboxes[i, 2];
            float y2 = bboxes[i, 3];

            // 转为 (x, y, w, h)
            float x = x1;
            float y = y1;
            float w = x2 - x1;
            float h = y2 - y1;

            // landmarks: [x0,y0,x1,y1,...]
            var lms = new Vector2[5];
            for (int k = 0; k < 5; k++)
            {
                lms[k] = new Vector2(landmarks[i, k * 2 + 0], landmarks[i, k * 2 + 1]);
            }

            results.Add(new FaceResult
            {
                x = x,
                y = y,
                width = w,
                height = h,
                score = score,
                landmarks = lms
            });
        }
        return results;
    }

    List<FaceResult> ApplyNMS(List<FaceResult> faces, float iouThreshold)
    {
        faces.Sort((a, b) => b.score.CompareTo(a.score));
        var keep = new List<FaceResult>();
        var suppressed = new bool[faces.Count];

        for (int i = 0; i < faces.Count; i++)
        {
            if (suppressed[i]) continue;
            keep.Add(faces[i]);
            for (int j = i + 1; j < faces.Count; j++)
            {
                if (suppressed[j]) continue;
                float iou = ComputeIOU(faces[i], faces[j]);
                if (iou > iouThreshold)
                    suppressed[j] = true;
            }
        }
        return keep;
    }

    float ComputeIOU(FaceResult a, FaceResult b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.x + a.width, b.x + b.width);
        float y2 = Mathf.Min(a.y + a.height, b.y + b.height);

        if (x2 <= x1 || y2 <= y1) return 0;

        float inter = (x2 - x1) * (y2 - y1);
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float union = areaA + areaB - inter;
        return inter / union;
    }
}

[System.Serializable]
public struct FaceResult
{
    public float x, y, width, height, score;
    public Vector2[] landmarks;
}