using Unity.InferenceEngine;
using UnityEngine;

public class Tools
{ 
    /// <summary>
    /// 工具函数：缩放纹理
    /// </summary>
    /// <param name="source"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <returns></returns>
    public static Texture2D ResizeTexture(Texture2D source, int width, int height)
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

    /// <summary>
    /// 工具函数：Texture -> Tensor (NCHW)
    /// </summary>
    /// <param name="tex"></param>
    /// <param name="div"></param>
    /// <returns></returns>
    public static Tensor<float> TextureToTensorNCHW(Texture2D tex, float div)
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
}