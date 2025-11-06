using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

public class FaceDetector : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Model runtimeModel;
    private Worker worker;

    public Texture2D texture;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        Debug.Log("=== 模型输出层信息 ===");
        foreach (var o in runtimeModel.outputs)
        {
            Debug.Log($"Output: {o.name}");
        }
        RunModel(texture);
    }

    public void RunModel(Texture2D inputImage)
    {
        Tensor<float> input = Preprocess(inputImage);

        worker.Schedule(input);

        // 获取所有9个输出
        Tensor<float> score1 = worker.PeekOutput("448").ReadbackAndClone() as Tensor<float>;
        Tensor<float> score2 = worker.PeekOutput("471").ReadbackAndClone() as Tensor<float>;
        Tensor<float> score3 = worker.PeekOutput("494").ReadbackAndClone() as Tensor<float>;

        Tensor<float> bbox1 = worker.PeekOutput("451").ReadbackAndClone() as Tensor<float>; ;
        Tensor<float> bbox2 = worker.PeekOutput("474").ReadbackAndClone() as Tensor<float>;
        Tensor<float> bbox3 = worker.PeekOutput("497").ReadbackAndClone() as Tensor<float>;

        Tensor<float> landmark1 = worker.PeekOutput("454").ReadbackAndClone() as Tensor<float>;
        Tensor<float> landmark2 = worker.PeekOutput("477").ReadbackAndClone() as Tensor<float>;
        Tensor<float> landmark3 = worker.PeekOutput("500").ReadbackAndClone() as Tensor<float>;

        // 解析结果
        var faces = DecodeOutputs(score1, bbox1, landmark1);
        faces.AddRange(DecodeOutputs(score2, bbox2, landmark2));
        faces.AddRange(DecodeOutputs(score3, bbox3, landmark3));

        // 打印结果
        foreach (var face in faces)
        {
            if (face.score > 0.6f)
            {
                Debug.Log($"Face: x={face.x}, y={face.y}, w={face.w}, h={face.h}, score={face.score}");
            }
        }

        input.Dispose();
        score1.Dispose(); score2.Dispose(); score3.Dispose();
        bbox1.Dispose(); bbox2.Dispose(); bbox3.Dispose();
        landmark1.Dispose(); landmark2.Dispose(); landmark3.Dispose();
    }

    private List<FaceBox> DecodeOutputs(Tensor<float> score, Tensor<float> bbox, Tensor<float> landmark)
    {
        List<FaceBox> result = new List<FaceBox>();
        int num = score.shape[1]; // 每层 anchor 数量
        Debug.Log("num:" + num);
        for (int i = 0; i < num; i++)
        {
            float conf = score[0, i, 0];
            Debug.Log("score:" + conf);
            //if (conf < 0.5f) continue;

            float x = bbox[0, i, 0];
            float y = bbox[0, i, 1];
            float w = bbox[0, i, 2];
            float h = bbox[0, i, 3];

            // landmark 可选
            float[] lm = new float[10];
            for (int j = 0; j < 10; j++)
                lm[j] = landmark[0, i, j];

            result.Add(new FaceBox(x, y, w, h, conf, lm));
        }
        return result;
    }

    private Tensor<float> Preprocess(Texture2D input)
    {
        const int SIZE = 128; // BlazeFace 默认 128x128
        Texture2D resized = new Texture2D(SIZE, SIZE);
        Graphics.ConvertTexture(input, resized);

        float[] floatValues = new float[SIZE * SIZE * 3];
        Color32[] pixels = resized.GetPixels32();
        for (int i = 0; i < pixels.Length; i++)
        {
            floatValues[i * 3 + 0] = pixels[i].r / 255.0f;
            floatValues[i * 3 + 1] = pixels[i].g / 255.0f;
            floatValues[i * 3 + 2] = pixels[i].b / 255.0f;
        }
        var shape = new TensorShape(1, 3, SIZE, SIZE);
        return new Tensor<float>(shape, floatValues);
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}

public class FaceBox
{
    public float x, y, w, h, score;
    public float[] landmarks;

    public FaceBox(float x, float y, float w, float h, float score, float[] landmarks)
    {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
        this.score = score;
        this.landmarks = landmarks;
    }
}
