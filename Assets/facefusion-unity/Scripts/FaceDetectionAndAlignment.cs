using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using Unity.Mathematics; 
using UnityEngine;
using UnityEngine.UI;

public class FaceDetectionAndAlignment : MonoBehaviour
{
    public Texture2D inputImage;           // 原始输入图像
    public ModelAsset detectionModel;      // det_10g.onnx
    public ModelAsset landmarkModel;       // 2d106det.onnx

    public RawImage debugOutput;           // 用于显示对齐后人脸

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
        var faces = DetectAndAlignFaces(inputImage);
        if (faces.Count > 0)
        {
            var aligned = AlignFace(inputImage, faces[0].landmarks5, 128);
            debugOutput.texture = aligned;
        }
    }

    List<DetectedFace> DetectAndAlignFaces(Texture2D image)
    {
        var faces = new List<DetectedFace>();

        // Step 1: 人脸检测
        var detections = RunFaceDetection(image);
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

    // ===== 1. 人脸检测 =====
    List<(Rect bbox, float score)> RunFaceDetection(Texture2D image)
    {
        const int inputSize = 640;
        var model = ModelLoader.Load(detectionModel);
        using var worker = new Worker( model, BackendType.GPUCompute);

        // 预处理：缩放到 640x640，归一化 [0,1]
        var resized = ResizeTexture(image, inputSize, inputSize);
        var inputTensor = TextureToTensorNCHW(resized, 1f); // [1,3,640,640], [0,1]
        worker.SetInput("input", inputTensor);
        worker.Schedule();
        var output = worker.PeekOutput("output") as Tensor<float>;
        var scores = output.DownloadToArray();

        // 后处理：解析输出（简化版，仅取高分框）
        var results = new List<(Rect, float)>();
        int numAnchors = output.shape[1]; // e.g., 16800

        for (int i = 0; i < numAnchors; i++)
        {
            float score = scores[i * 15 + 4]; // 第5个是置信度
            if (score > 0.7f)
            {
                float x1 = scores[i * 15 + 0];
                float y1 = scores[i * 15 + 1];
                float x2 = scores[i * 15 + 2];
                float y2 = scores[i * 15 + 3];

                // 转换为原始图像坐标（注意：模型输入是 640x640）
                float scaleX = (float)image.width / inputSize;
                float scaleY = (float)image.height / inputSize;

                Rect bbox = new Rect(
                    x1 * scaleX, y1 * scaleY,
                    (x2 - x1) * scaleX, (y2 - y1) * scaleY
                );

                results.Add((bbox, score));
            }
        }

        // 可加 NMS（此处省略）
        return results;
    }

    // ===== 2. 关键点检测 =====
    float2[] RunLandmarkDetection(Texture2D faceCrop)
    {
        const int inputSize = 192;
        var model = ModelLoader.Load(landmarkModel);
        using var worker = new Worker(model,BackendType.GPUCompute );

        var resized = ResizeTexture(faceCrop, inputSize, inputSize);
        var inputTensor = TextureToTensorNCHW(resized, 255f); // 注意：2d106det 通常用 [0,255]

        worker.SetInput("input", inputTensor);
        worker.Schedule();
        var output = worker.PeekOutput("output") as Tensor<float>;
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

    // 工具函数：缩放纹理
    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        var rt = RenderTexture.GetTemporary(width, height);
        Graphics.Blit(source, rt);
        var dest = new Texture2D(width, height);
        RenderTexture.active = rt;
        dest.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        dest.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);
        return dest;
    }

    // 工具函数：Texture -> Tensor (NCHW)
    Tensor<float> TextureToTensorNCHW(Texture2D tex, float div)
    {
        var pixels = tex.GetPixels32();
        int w = tex.width, h = tex.height;
        var data = new float[1 * 3 * h * w];
        int i = 0;
        for (int y = h - 1; y >= 0; y--)
        {
            for (int x = 0; x < w; x++)
            {
                var p = pixels[y * w + x];
                data[i++] = p.r / div;
                data[i++] = p.g / div;
                data[i++] = p.b / div;
            }
        }
        return new Tensor<float>(new TensorShape(1, 3, h, w), data);
    }

    // 工具函数：裁剪人脸区域
    Texture2D CropFace(Texture2D src, Rect rect, int size)
    {
        // 扩展 bbox 以包含更多上下文（可选）
        float cx = rect.x + rect.width / 2;
        float cy = rect.y + rect.height / 2;
        float scale = 1.3f;
        float w = rect.width * scale;
        float h = rect.height * scale;

        Rect cropRect = new Rect(cx - w / 2, cy - h / 2, w, h);
        cropRect.x = Mathf.Max(0, cropRect.x);
        cropRect.y = Mathf.Max(0, cropRect.y);
        cropRect.width = Mathf.Min(src.width - cropRect.x, cropRect.width);
        cropRect.height = Mathf.Min(src.height - cropRect.y, cropRect.height);

        var rt = RenderTexture.GetTemporary(size, size);
        var prev = RenderTexture.active;
        RenderTexture.active = rt;

        GL.PushMatrix();
        GL.LoadPixelMatrix(0, size, size, 0);
        Graphics.DrawTexture(new Rect(0, 0, size, size), src, new Rect(
            cropRect.x / src.width, 
            cropRect.y / src.height, 
            cropRect.width / src.width, 
            cropRect.height / src.height));
        GL.PopMatrix();

        var cropped = new Texture2D(size, size);
        cropped.ReadPixels(new Rect(0, 0, size, size), 0, 0);
        cropped.Apply();

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);
        return cropped;
    }
}