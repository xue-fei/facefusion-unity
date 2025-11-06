using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using Unity.Mathematics;
using UnityEngine;

public class FaceDetection : MonoBehaviour
{
    public Texture2D inputImage;
    public ModelAsset detectionModelAsset; // det_10g.onnx

    const int INPUT_SIZE = 640;
    static readonly int[] STRIDES = { 8, 16, 32 };

    void Start()
    {
        if (inputImage == null || detectionModelAsset == null)
        {
            Debug.LogError("Missing input image or model.");
            return;
        }

        var faces = RunFaceDetection(inputImage);
        Debug.Log($"Detected {faces.Count} faces.");
        foreach (var face in faces)
        {
            Debug.Log($"Score: {face.score:F3}, BBox: {face.bbox}");
        }
    }

    List<FaceResult> RunFaceDetection(Texture2D image)
    {
        var model = ModelLoader.Load(detectionModelAsset);
        using var worker = new Worker(model, BackendType.GPUCompute);

        // 1. 预处理：Resize + 归一化到 [0,1]
        var resized = ResizeTexture(image, INPUT_SIZE, INPUT_SIZE);
        var inputTensor = TextureToTensorNCHW(resized, 255f); // 注意：det_10g 通常输入 [0,255]，但有些版本用 [0,1]
        worker.SetInput(0, inputTensor);
        worker.Schedule();

        // 2. 获取9个输出（3层 × 3输出：score, bbox, kps）
        var outputs = new Tensor<float>[9];
        for (int i = 0; i < 9; i++)
        {
            outputs[i] = worker.PeekOutput(i) as Tensor<float>;
        }

        var allScores = new List<float>();
        var allBoxes = new List<float4>();
        var allKpss = new List<float2[]>();

        // 3. 后处理每层输出
        for (int i = 0; i < 3; i++)
        {
            var scoreTensor = outputs[i * 3 + 0]; // [1, 2, H, W]
            var bboxTensor = outputs[i * 3 + 1]; // [1, 4, H, W]
            Debug.Log($"bboxTensor shape: {bboxTensor.shape}");
            Debug.Log($"bboxTensor data count: {bboxTensor.count}");
            var kpsTensor = outputs[i * 3 + 2]; // [1, 10, H, W]

            int stride = STRIDES[i];
            int H = INPUT_SIZE / stride;
            int W = H;

            var anchorCenters = GenerateAnchorCenters(H, W, stride); // 长度 = H * W * 2

            var boxes = Distance2BBox(anchorCenters, bboxTensor.DownloadToArray(), H, W, stride);
            var kpss = Distance2Kps(anchorCenters, kpsTensor.DownloadToArray(), H, W, stride);
            var scores = scoreTensor.DownloadToArray(); // NCHW: [1,2,H,W]

            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    int idx = h * W + w;
                    // score 是 [1,2,H,W] → 第 1 个通道是背景，第 2 个是人脸置信度
                    // 所以取 channel=1
                    int scoreIdx = ((0 * 2 + 1) * H + h) * W + w; // 取人脸分数（不是背景）
                    float score = scores[scoreIdx];
                    if (score > 0.5f)
                    {
                        allScores.Add(score);
                        allBoxes.Add(boxes[idx]);
                        allKpss.Add(kpss[idx]);
                    }
                }
            }
        }

        // 4. 映射回原图坐标
        var results = new List<FaceResult>();
        float scaleX = (float)image.width / INPUT_SIZE;
        float scaleY = (float)image.height / INPUT_SIZE;

        for (int i = 0; i < allScores.Count; i++)
        {
            var box = allBoxes[i];
            var kps = allKpss[i];

            Rect bbox = new Rect(
                box.x * scaleX, box.y * scaleY,
                (box.z - box.x) * scaleX, (box.w - box.y) * scaleY
            );

            float2[] landmarks5 = new float2[5];
            for (int j = 0; j < 5; j++)
            {
                landmarks5[j] = new float2(kps[j].x * scaleX, kps[j].y * scaleY);
            }

            results.Add(new FaceResult
            {
                score = allScores[i],
                bbox = bbox,
                landmarks5 = landmarks5
            });
        }

        // 5. 可选：加 NMS（非极大值抑制）
        results = ApplyNMS(results, 0.4f);

        return results;
    }



    // ========== 工具函数 ==========
    public class FaceResult
    {
        public float score;
        public Rect bbox;
        public float2[] landmarks5;
    }

    float2[] GenerateAnchorCenters(int H, int W, int stride)
    {
        var centers = new float2[H * W];
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                centers[i * W + j] = new float2((j + 0.5f) * stride, (i + 0.5f) * stride);
            }
        }
        return centers;
    }

    float4[] Distance2BBox(float2[] anchorCenters, float[] rawBBoxes, int H, int W, int stride)
    {
        var boxes = new float4[anchorCenters.Length]; // 1 anchor per location
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                int idx = i * W + j;
                float2 center = anchorCenters[idx];

                // NCHW: [1,4,H,W] → channel 0~3
                int baseIdx = ((0 * 4 + 0) * H + i) * W + j; // dx1
                float dx1 = rawBBoxes[baseIdx];
                float dy1 = rawBBoxes[baseIdx + 1 * H * W];   // channel 1
                float dx2 = rawBBoxes[baseIdx + 2 * H * W];   // channel 2
                float dy2 = rawBBoxes[baseIdx + 3 * H * W];   // channel 3

                boxes[idx] = new float4(
                    center.x - dx1,
                    center.y - dy1,
                    center.x + dx2,
                    center.y + dy2
                );
            }
        }
        return boxes;
    }

    float2[][] Distance2Kps(float2[] anchorCenters, float[] rawKps, int H, int W, int stride)
    {
        var kpss = new float2[anchorCenters.Length][];
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++)
            {
                int idx = i * W + j;
                float2 center = anchorCenters[idx];
                var kps = new float2[5];
                for (int p = 0; p < 5; p++)
                {
                    // channel: 2*p (x), 2*p+1 (y)
                    int cx = ((0 * 10 + 2 * p + 0) * H + i) * W + j;
                    int cy = ((0 * 10 + 2 * p + 1) * H + i) * W + j;
                    float kpx = rawKps[cx];
                    float kpy = rawKps[cy];
                    kps[p] = center + new float2(kpx, kpy);
                }
                kpss[idx] = kps;
            }
        }
        return kpss;
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
                float iou = IOU(faces[i].bbox, faces[j].bbox);
                if (iou > iouThreshold) suppressed[j] = true;
            }
        }
        return keep;
    }

    float IOU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.x + a.width, b.x + b.width);
        float y2 = Mathf.Min(a.y + a.height, b.y + b.height);
        if (x2 <= x1 || y2 <= y1) return 0;
        float inter = (x2 - x1) * (y2 - y1);
        float union = a.width * a.height + b.width * b.height - inter;
        return inter / union;
    }

    // Texture 工具（需自行实现 ResizeTexture 和 TextureToTensorNCHW）
    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(source, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D result = new Texture2D(width, height);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    Tensor<float> TextureToTensorNCHW(Texture2D tex, float divideBy = 255f)
    {
        Color32[] pixels = tex.GetPixels32();
        int w = tex.width;
        int h = tex.height;
        float[] data = new float[3 * h * w];

        // 注意：Unity Texture 是从左下角开始，但通常 CNN 对此不敏感
        // 如果方向不对，可加 FlipY 逻辑
        for (int i = 0; i < pixels.Length; i++)
        {
            // NCHW: Channel first
            data[i] = pixels[i].r / divideBy; // R -> channel 0
            data[i + w * h] = pixels[i].g / divideBy; // G -> channel 1
            data[i + 2 * w * h] = pixels[i].b / divideBy; // B -> channel 2
        }

        // ✅ 正确创建 Tensor：shape 先，data 后
        var shape = new TensorShape(1, 3, h, w);
        return new Tensor<float>(shape, data);
    }
}