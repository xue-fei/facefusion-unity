using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

public class FaceDetectionAndAlignment : MonoBehaviour
{
    public Texture2D inputImage;           // 原始输入图像
    public ModelAsset detectionModelAsset;      // det_10g.onnx
    public ModelAsset landmarkModelAsset;       // 2d106det.onnx

    public RawImage debugOutput;           // 用于显示对齐后人脸

    public Model detectionModel;
    public Worker detectionWorker;
    public Model landmarkModel;
    public Worker landmarkWorker;

    [Serializable]
    public class FaceResult
    {
        public float score;
        public Rect bbox;
        public float2[] landmarks5; // 5 points: LE, RE, Nose, LM, RM
    }

    // 标准 5 点（128x128）
    static readonly float2[] s_standard5Points = {
        new float2(38.2946f, 51.6963f),
        new float2(73.5318f, 51.5014f),
        new float2(56.0252f, 71.7366f),
        new float2(41.5493f, 92.3655f),
        new float2(70.7299f, 92.2041f)
    };

    [Serializable]
    public class DetectedFace
    {
        public Rect bbox;
        public float2[] landmarks106;
        public float2[] landmarks5; // 从106中提取
    }

    void Start()
    {
        if (inputImage == null) return;

        detectionModel = ModelLoader.Load(detectionModelAsset);
        detectionWorker = new Worker(detectionModel, BackendType.GPUCompute);

        landmarkModel = ModelLoader.Load(landmarkModelAsset);
        landmarkWorker = new Worker(landmarkModel, BackendType.GPUCompute);

        var faces = DetectAndAlignFaces(inputImage);
        if (faces.Count > 0)
        {
            var aligned = AlignFace(inputImage, faces[0].landmarks5, 128);
            debugOutput.texture = aligned;
        }
        else
        {
            Debug.Log("no face");
        }
    }

    List<DetectedFace> DetectAndAlignFaces(Texture2D image)
    {
        var faces = new List<DetectedFace>();

        // Step 1: 人脸检测
        var detections = RunFaceDetection(image);
        foreach (var detection in detections)
        {
            Debug.Log(detection.score + " " + detection.bbox + " " + detection.score);
        }
        foreach (var det in detections)
        {
            // Step 2: 关键点检测（裁剪人脸区域）
            var cropped = CropFace(image, det.bbox, 192);
            var landmarks106 = RunLandmarkDetection(cropped);

            // Step 3: 从106点中提取5点
            var landmarks5 = Extract5Points(landmarks106, det.bbox, image.width, image.height);

            faces.Add(new DetectedFace
            {
                bbox = det.bbox,
                landmarks106 = landmarks106,
                landmarks5 = landmarks5
            });
        }

        return faces;
    }

    const int inputSize = 640;
    // ===== 1. 人脸检测 =====
    List<FaceResult> RunFaceDetection(Texture2D image)
    {
        // 预处理：缩放到 640x640，归一化 [0,1]
        var resized = Tools.ResizeTexture(image, inputSize, inputSize);
        var inputTensor = Tools.TextureToTensorNCHW(resized, 255f); // [1,3,640,640], [0,1]
        detectionWorker.SetInput(0, inputTensor);
        detectionWorker.Schedule();
        var outputs = new Tensor<float>[9];
        for (int i = 0; i < 9; i++)
        {
            outputs[i] = detectionWorker.PeekOutput(i) as Tensor<float>;
            outputs[i].ReadbackAndClone();
        }
        var allScores = new List<float>();
        var allBoxes = new List<float4>();
        var allKpss = new List<float2[]>();

        int[] strides = { 8, 16, 32 };
        for (int i = 0; i < 3; i++)
        {
            var scores = outputs[i * 3 + 0];
            var bboxes = outputs[i * 3 + 1];
            var kpss = outputs[i * 3 + 2];

            int stride = strides[i];
            int H = inputSize / stride; // e.g., 80, 40, 20
            int W = H;

            var anchorCenters = GenerateAnchorCenters(H, W, stride); // 长度 = H*W*2

            // ✅ 传入 H, W
            var boxes = Distance2BBox(anchorCenters, bboxes.DownloadToArray(), H, W, stride);
            var keypoints = Distance2Kps(anchorCenters, kpss.DownloadToArray(), H, W, stride);

            // scores: [1, 2, H, W]
            var scoreArray = scores.DownloadToArray();
            for (int a = 0; a < 2; a++)
            {
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        int idx = (a * H + h) * W + w;
                        float score = scoreArray[(a * H + h) * W + w]; // 注意：scores 也是 NCHW
                        if (score > 0.5f)
                        {
                            allScores.Add(score);
                            allBoxes.Add(boxes[idx]);
                            allKpss.Add(keypoints[idx]);
                        }
                    }
                }
            }
        }

        // 转换为 Unity 坐标（640x640 -> 原图）
        var results = new List<FaceResult>();
        float scaleX = (float)image.width / inputSize;
        float scaleY = (float)image.height / inputSize;

        for (int i = 0; i < allScores.Count; i++)
        {
            var score = allScores[i];
            var box = allBoxes[i];
            var kps = allKpss[i];

            // bbox
            Rect bbox = new Rect(
                box.x * scaleX,
                box.y * scaleY,
                (box.z - box.x) * scaleX,
                (box.w - box.y) * scaleY
            );

            // 5 points (kps 是 10 个值: x0,y0,x1,y1,...x4,y4)
            float2[] lm5 = new float2[5];
            for (int j = 0; j < 5; j++)
            {
                lm5[j] = new float2(
                    kps[j].x * scaleX,
                    kps[j].y * scaleY
                );
            }

            results.Add(new FaceResult { score = score, bbox = bbox, landmarks5 = lm5 });
        }

        return results;
    }

    // ===== 2. 关键点检测 =====
    float2[] RunLandmarkDetection(Texture2D faceCrop)
    {
        const int inputSize = 192;

        var resized = Tools.ResizeTexture(faceCrop, inputSize, inputSize);
        var inputTensor = Tools.TextureToTensorNCHW(resized, 1f); // 注意：2d106det 通常用 [0,255]

        landmarkWorker.SetInput("data", inputTensor);
        landmarkWorker.Schedule();
        var output = landmarkWorker.PeekOutput() as Tensor<float>;
        //var fgrAwaiter = output.ReadbackAndCloneAsync().GetAwaiter();
        //while (!fgrAwaiter.IsCompleted)
        //{

        //}
        var data = output.DownloadToArray();

        var points = new float2[106];
        for (int i = 0; i < 106; i++)
        {
            // 输出是 [x, y] 像素坐标（相对于 192x192）
            points[i] = new float2(data[i * 2], data[i * 2 + 1]);
        }

        // 将关键点映射回原始裁剪区域（192x192 -> faceCrop 尺寸）
        float scaleX = (float)faceCrop.width / inputSize;
        float scaleY = (float)faceCrop.height / inputSize;
        for (int i = 0; i < 106; i++)
        {
            points[i] = new float2(points[i].x * scaleX, points[i].y * scaleY);
        }

        return points;
    }

    // ===== 3. 从106点提取5点 =====
    float2[] Extract5Points(float2[] lmk106, Rect faceRect, int imgW, int imgH)
    {
        // 106点中对应5点的索引（参考 InsightFace）
        int[] idxMap = { 38, 88, 81, 48, 95 }; // LE, RE, Nose, LM, RM

        var points = new float2[5];
        for (int i = 0; i < 5; i++)
        {
            // lmk106 是相对于裁剪后人脸的坐标，需加 faceRect 偏移
            points[i] = new float2(
                faceRect.x + lmk106[idxMap[i]].x,
                faceRect.y + lmk106[idxMap[i]].y
            );
        }
        return points;
    }

    // ===== 4. 仿射对齐 =====
    Texture2D AlignFace(Texture2D srcImage, float2[] src5Points, int outSize = 128)
    {
        // 计算仿射变换矩阵（从 src5Points -> standard5Points）
        var M = SimilarityTransform(src5Points, s_standard5Points);

        // 创建输出纹理
        var aligned = new Texture2D(outSize, outSize, TextureFormat.RGB24, false);
        var pixels = new Color32[outSize * outSize];

        // 逆变换：对输出每个像素，映射回原图
        for (int y = 0; y < outSize; y++)
        {
            for (int x = 0; x < outSize; x++)
            {
                // [x, y, 1] -> 原图坐标
                float srcX = M.c0.x * x + M.c1.x * y + M.c2.x;
                float srcY = M.c0.y * x + M.c1.y * y + M.c2.y;

                // 双线性插值采样
                Color32 color = BilinearSample(srcImage, srcX, srcY);
                pixels[y * outSize + x] = color;
            }
        }

        aligned.SetPixels32(pixels);
        aligned.Apply();
        return aligned;
    }

    // 计算相似变换矩阵（旋转+缩放+平移）
    float3x3 SimilarityTransform(float2[] srcPoints, float2[] dstPoints)
    {
        // 使用最小二乘法求解仿射变换（仅相似变换）
        // 简化：使用前3点计算（足够稳定）
        float2 srcCenter = (srcPoints[0] + srcPoints[1]) * 0.5f;
        float2 dstCenter = (dstPoints[0] + dstPoints[1]) * 0.5f;

        float srcDx = srcPoints[1].x - srcPoints[0].x;
        float srcDy = srcPoints[1].y - srcPoints[0].y;
        float dstDx = dstPoints[1].x - dstPoints[0].x;
        float dstDy = dstPoints[1].y - dstPoints[0].y;

        float srcScale = Mathf.Sqrt(srcDx * srcDx + srcDy * srcDy);
        float dstScale = Mathf.Sqrt(dstDx * dstDx + dstDy * dstDy);
        float scale = dstScale / srcScale;

        float srcAngle = Mathf.Atan2(srcDy, srcDx);
        float dstAngle = Mathf.Atan2(dstDy, dstDx);
        float angle = dstAngle - srcAngle;

        float cosA = Mathf.Cos(angle) * scale;
        float sinA = Mathf.Sin(angle) * scale;

        // 构建变换矩阵
        var M = new float3x3(
            new float3(cosA, sinA, 0),
            new float3(-sinA, cosA, 0),
            new float3(dstCenter.x - (cosA * srcCenter.x - sinA * srcCenter.y),
                       dstCenter.y - (sinA * srcCenter.x + cosA * srcCenter.y), 1)
        );
        return M;
    }

    // 双线性采样
    Color32 BilinearSample(Texture2D tex, float x, float y)
    {
        int w = tex.width;
        int h = tex.height;

        if (x < 0 || x >= w - 1 || y < 0 || y >= h - 1)
            return new Color32(0, 0, 0, 255);

        int x0 = (int)Mathf.Floor(x);
        int y0 = (int)Mathf.Floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = x - x0;
        float wy = y - y0;

        Color32 c00 = tex.GetPixel(x0, y0);
        Color32 c10 = tex.GetPixel(x1, y0);
        Color32 c01 = tex.GetPixel(x0, y1);
        Color32 c11 = tex.GetPixel(x1, y1);

        Color32 top = LerpColor(c00, c10, wx);
        Color32 bottom = LerpColor(c01, c11, wx);
        return LerpColor(top, bottom, wy);
    }

    Color32 LerpColor(Color32 a, Color32 b, float t)
    {
        return new Color32(
            (byte)(a.r + t * (b.r - a.r)),
            (byte)(a.g + t * (b.g - a.g)),
            (byte)(a.b + t * (b.b - a.b)),
            (byte)(a.a + t * (b.a - a.a))
        );
    }

    // 工具函数：裁剪人脸区域（CPU 实现）
    Texture2D CropFace(Texture2D src, Rect rect, int size)
    {
        // 扩展边界框
        float cx = rect.x + rect.width / 2f;
        float cy = rect.y + rect.height / 2f;
        float scale = 1.3f;
        float w = rect.width * scale;
        float h = rect.height * scale;

        Rect cropRect = new Rect(cx - w / 2f, cy - h / 2f, w, h);
        // 限制在图像内
        cropRect.x = Mathf.Clamp(cropRect.x, 0, src.width);
        cropRect.y = Mathf.Clamp(cropRect.y, 0, src.height);
        cropRect.width = Mathf.Clamp(cropRect.width, 0, src.width - cropRect.x);
        cropRect.height = Mathf.Clamp(cropRect.height, 0, src.height - cropRect.y);

        int cropW = Mathf.Max(1, Mathf.RoundToInt(cropRect.width));
        int cropH = Mathf.Max(1, Mathf.RoundToInt(cropRect.height));

        // 裁剪像素
        Color[] pixels = src.GetPixels(
            Mathf.RoundToInt(cropRect.x),
            Mathf.RoundToInt(cropRect.y),
            cropW,
            cropH
        );

        Texture2D cropped = new Texture2D(cropW, cropH, TextureFormat.RGB24, false);
        cropped.SetPixels(pixels);
        cropped.Apply();

        // 缩放到目标尺寸
        return Tools.ResizeTexture(cropped, size, size);
    }

    // 生成 anchor centers: (H*W*2, 2)
    float2[] GenerateAnchorCenters(int H, int W, int stride)
    {
        var centers = new List<float2>();
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                float cy = (h + 0.5f) * stride;
                float cx = (w + 0.5f) * stride;
                // 重复两次（num_anchors=2）
                centers.Add(new float2(cx, cy));
                centers.Add(new float2(cx, cy));
            }
        }
        return centers.ToArray();
    }

    // distance2kps: 输入 distance [N,10], 输出 5 个点 [(x,y), ...]
    float2[][] Distance2Kps(float2[] centers, float[] preds, int H, int W, int stride)
    {
        int numAnchors = 2;
        int total = H * W * numAnchors;
        var kpss = new float2[total][];

        // preds shape: [1, 20, H, W] → 20 = numAnchors * 10 (5 points × 2)
        for (int a = 0; a < numAnchors; a++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    int idx = (a * H + h) * W + w;
                    float cx = centers[idx].x;
                    float cy = centers[idx].y;

                    var points = new float2[5];
                    int baseChan = a * 10;
                    for (int p = 0; p < 5; p++)
                    {
                        float dx = preds[(baseChan + p * 2 + 0) * H * W + h * W + w] * stride;
                        float dy = preds[(baseChan + p * 2 + 1) * H * W + h * W + w] * stride;
                        points[p] = new float2(cx + dx, cy + dy);
                    }
                    kpss[idx] = points;
                }
            }
        }
        return kpss;
    }

    // distance2bbox: 输入 distance [N,4], 输出 [x1,y1,x2,y2]
    float4[] Distance2BBox(float2[] centers, float[] preds, int H, int W, int stride)
    {
        int numAnchors = 2;
        int total = H * W * numAnchors;
        var boxes = new float4[total];

        // preds shape: [1, 8, H, W] → 8 = numAnchors * 4
        // 所以 preds 有 8 个通道：C0~C7
        // C0~C3: anchor0 (dx1, dy1, dx2, dy2)
        // C4~C7: anchor1 (dx1, dy1, dx2, dy2)

        for (int a = 0; a < numAnchors; a++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    int idx = (a * H + h) * W + w; // centers 的索引
                    float cx = centers[idx].x;
                    float cy = centers[idx].y;

                    // 通道偏移：每个 anchor 占 4 个通道
                    int baseChan = a * 4;
                    // 通道内的空间偏移：C × H × W + h × W + w
                    float dx1 = preds[(baseChan + 0) * H * W + h * W + w] * stride;
                    float dy1 = preds[(baseChan + 1) * H * W + h * W + w] * stride;
                    float dx2 = preds[(baseChan + 2) * H * W + h * W + w] * stride;
                    float dy2 = preds[(baseChan + 3) * H * W + h * W + w] * stride;

                    float x1 = cx - dx1;
                    float y1 = cy - dy1;
                    float x2 = cx + dx2;
                    float y2 = cy + dy2;

                    boxes[idx] = new float4(x1, y1, x2, y2);
                }
            }
        }
        return boxes;
    }

    private void OnDestroy()
    {
        detectionWorker?.Dispose();
        landmarkWorker?.Dispose();
    }
}