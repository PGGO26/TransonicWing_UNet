> 使用基於開源軟體 OpenFOAM 開發的求解器 HiSA 對不同的 3D 機翼再穿音速不同攻角條件下做三維的流場模擬，使用模擬的數據建立資料庫並結合機器學習使用 UNet 架構來做訓練，預期能快速地得到不同機翼在不同來流條件下的表面壓力分佈

**ps. OpenFOAM 版本為 v2206**

![image](https://github.com/user-attachments/assets/22a943c5-24d9-4c1a-be04-3a33d040567c)

## 簡介

### HiSA 模擬
- **OpenFOAM 資料夾**
  - 主要包含三個資料夾 *0/* 控制初始條件
  - *constant/* 定義外型及物理模型
  - *system/* 包含調整計算步長等控制面
- **dataGen.py 程式碼**
  - 自動執行包含流場模擬及數據後處理
- **utils.py 函式庫**
  - 提供給 dataGen.py 運作的函式

### UNet 訓練
- utils.py 函式庫，資料處理及匯入
- UNet.py UNet 神經網路架構
- runTrain.py 訓練神經網路模型
- runTest.py 測試模型預測結果

### 流程
- *Mesh/* 資料夾中存放由 fluent 建立的網格，並在 *OpenFOAM/* 資料夾中調整 *write_surface.py* 檔案中的 res 來控制後處理需要的圖片解析度 **(上面是機翼上下表面，下面是機翼截面)**
