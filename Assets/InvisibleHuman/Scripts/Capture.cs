using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

using OpenCvSharp;
using OpenCvSharp.Util;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

namespace InvisibleHuman.CaptureFunction
{
    public class Capture : MonoBehaviour
    {
        [SerializeField] private ARCameraManager CameraManager;
        [SerializeField] private AROcclusionManager OcclusionManager;

        [Space]
        [SerializeField] private RawImage RGB_Image;
        [SerializeField] private RawImage HumanStencil_Image;
        [SerializeField] private RawImage Inpaint_Image;

        [Space]
        [SerializeField] private Material HumanStencil_Material;

        private Texture2D RGB_Texture;
        private Texture2D Stencil_Texture;
        private RenderTexture HumanStencil_RT;

        private int width = 640, height = 480;

        private CancellationTokenSource tokenSource;

        #region Runtime
        private void Start()
        {
            CameraManager.frameReceived += OnARCameraFrameReceived;

            tokenSource = new CancellationTokenSource();
            var cancelToken = tokenSource.Token;
            _ = CaptureLoop(cancelToken);
        }

        private void OnDestroy()
        {
            CameraManager.frameReceived -= OnARCameraFrameReceived;
            tokenSource.Cancel();
        }

        private void Update()
        {
            #region Get Human Stencil
            var currentStencil = OcclusionManager.humanStencilTexture;

            if (currentStencil == null) return;

            if (HumanStencil_RT == null)
            {
                HumanStencil_RT = RenderTexture.GetTemporary(currentStencil.width, currentStencil.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                HumanStencil_RT.Create();
            }

            if (Stencil_Texture == null)
            {
                Stencil_Texture = new Texture2D(currentStencil.width, currentStencil.height);
            }

            Graphics.Blit(currentStencil, HumanStencil_RT, HumanStencil_Material);

            var currentRT = RenderTexture.active;
            RenderTexture.active = HumanStencil_RT;
            Stencil_Texture.ReadPixels(new UnityEngine.Rect(0, 0, Stencil_Texture.width, Stencil_Texture.height), 0, 0);
            Stencil_Texture.Apply();

            RenderTexture.active = currentRT;
            #endregion
        }

        private async Task CaptureLoop(CancellationToken cancelToken)
        {
            byte[] dilateArray =
                {   1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, };

            Texture2D stencilViewTexture = new Texture2D(width,height);
            Texture2D rgbViewTexture = new Texture2D(width, height);
            Texture2D inpaintViewTexture = new Texture2D(width, height);
            Texture2D floodFillTexture = new Texture2D(width, height);

            HumanStencil_Image.texture = stencilViewTexture;
            RGB_Image.texture = rgbViewTexture;
            // Inpaint_Image.texture = inpaintViewTexture;
            Inpaint_Image.texture = floodFillTexture;

            while (!cancelToken.IsCancellationRequested)
            {
                await Task.Delay(10);

                if (RGB_Texture == null || Stencil_Texture == null) continue;

                using (Mat stencilMat = OpenCvSharp.Unity.TextureToMat(Stencil_Texture))
                using (Mat rgbMat = OpenCvSharp.Unity.TextureToMat(RGB_Texture))
                using (Mat inpaintMat = new Mat())
                {

                    #region stencil texture
                    Cv2.CvtColor(stencilMat, stencilMat, ColorConversionCodes.BGR2GRAY);
                    // Cv2.Dilate(stencilMat, stencilMat, InputArray.Create(dilateArray));
                    Cv2.Resize(stencilMat, stencilMat, new OpenCvSharp.Size(width, height));
                    stencilViewTexture = OpenCvSharp.Unity.MatToTexture(stencilMat, stencilViewTexture);
                    #endregion

                    #region floodfill texture
                    Mat baseMat = OpenCvSharp.Unity.TextureToMat(Stencil_Texture);
                    Cv2.CvtColor(baseMat,baseMat, ColorConversionCodes.BGR2GRAY);

                    Point[][] contours;
			        HierarchyIndex[] hierarchy;
			        Cv2.FindContours (baseMat, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple, null);
                    
                    Cv2.Dilate(baseMat, baseMat, InputArray.Create(dilateArray));
                    Cv2.GaussianBlur (baseMat, baseMat, new Size(5,5),0);
                    Cv2.Threshold(baseMat, baseMat, 220,255,ThresholdTypes.Binary);

                    UnityEngine.Debug.Log("Chegou até aqui baseMat ROWS---->:"+baseMat.Rows);
                    UnityEngine.Debug.Log("Chegou até aqui baseMat COLS--->:"+baseMat.Cols);

                    IntPtr m_rows= ((IntPtr)baseMat.Size().Height)+2;
                    IntPtr m_cols = ((IntPtr)baseMat.Size().Width)+2;

                    Mat mask = new Mat((int)m_cols, (int)m_rows, MatType.CV_32F);

                    // mask.rows == size.height+2 && mask.cols == size.width+2
                    // Cv2.FloodFill(baseMat,baseMat, new Point (0,0), new Scalar (0,0,0));
                    // Cv2.Resize(baseMat,baseMat, new OpenCvSharp.Size(width, height));
                    // Cv2.Resize(baseMat,baseMat, new OpenCvSharp.Size(baseMat.Size().Width+2, baseMat.Size().Height+2));
                    // Cv2.Resize(baseMat,baseMat, new OpenCvSharp.Size(width+2, height+2));


                    UnityEngine.Debug.Log("Resized BASE ROWS---->:"+baseMat.Rows);
                    UnityEngine.Debug.Log("Resized BASE COLS--->:"+baseMat.Cols);

                    Cv2.FloodFill(baseMat,mask, new Point (0,0), new Scalar (0,0,0));

                    floodFillTexture = OpenCvSharp.Unity.MatToTexture(baseMat, floodFillTexture);

                    #endregion

                    #region rgb texture
                    Cv2.Resize(rgbMat, rgbMat, new OpenCvSharp.Size(width, height));
                    Cv2.Flip(rgbMat, rgbMat, FlipMode.Y);
                    rgbViewTexture = OpenCvSharp.Unity.MatToTexture(rgbMat, rgbViewTexture);
                    #endregion


                    #region inpaint
                    Cv2.Inpaint(rgbMat, stencilMat, inpaintMat, 3, InpaintMethod.NS);
                    inpaintViewTexture = OpenCvSharp.Unity.MatToTexture(inpaintMat, inpaintViewTexture);
                    #endregion

                    stencilMat.Dispose();
                    baseMat.Dispose();
                    rgbMat.Dispose();
                    inpaintMat.Dispose();
                }
            }

        }
        #endregion

        #region ARCamera Callback
        unsafe void OnARCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
        {
            //Reference:https://docs.unity3d.com/Packages/com.unity.xr.arfoundation@4.1/manual/cpu-camera-image.html

            if (!CameraManager.TryAcquireLatestCpuImage(out XRCpuImage image)) return;

            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(0, 0, image.width, image.height),
                outputDimensions = new Vector2Int(image.width / 2, image.height / 2),
                outputFormat = TextureFormat.RGBA32,
                transformation = XRCpuImage.Transformation.MirrorY
            };

            int size = image.GetConvertedDataSize(conversionParams);
            var buffer = new NativeArray<byte>(size, Allocator.Temp);
            image.Convert(conversionParams, new IntPtr(buffer.GetUnsafePtr()), buffer.Length);
            image.Dispose();

            if (RGB_Texture == null)
            {
                RGB_Texture = new Texture2D(conversionParams.outputDimensions.x, conversionParams.outputDimensions.y, conversionParams.outputFormat, false);
            }
            
            RGB_Texture.LoadRawTextureData(buffer);
            RGB_Texture.Apply();

            buffer.Dispose();
        }
        #endregion
    }
}

