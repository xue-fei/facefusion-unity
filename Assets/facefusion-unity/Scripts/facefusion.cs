using Mono.Cecil;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

public class facefusion : MonoBehaviour
{
    public ModelAsset modelAsset;
    private Worker worker;
    private Model model;
    public Texture2D sourceFace;  // 源人脸图片 
    public Texture2D targetImage; // 目标图片
    public RenderTexture result;

    private Tensor<float> source, target;

    private void Awake()
    {
        model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.GPUCompute);
        source = new Tensor<float>(new TensorShape(1, 512), new float[] { 0.0f });
        target = new Tensor<float>(new TensorShape(1, 3, 128, 128), new float[] { 0.0f });

        TextureConverter.ToTensor(sourceFace, source, new TextureTransform());
        TextureConverter.ToTensor(targetImage, target, new TextureTransform());
         
        worker.SetInput(0, target);
        worker.SetInput(1, source);
        worker.Schedule();

        // 获取输出[citation:2]
        Tensor<float> resultTensor = worker.PeekOutput() as Tensor<float>;

        TextureConverter.RenderToTexture(resultTensor, result);

        // 释放张量资源
        source.Dispose();
        target.Dispose();
        resultTensor.Dispose();
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnDestroy()
    {
        // 清理worker[citation:2]
        worker?.Dispose();
    }
}