from typing import List
import numpy as np

def calculateMassSpectrumMzTranverse(mz: np.ndarray, delta: float) -> np.ndarray:
    """
    返回左右偏移后的质荷比序列
    """
    return np.sort(
        np.concatenate((
            mz[:, 0] + delta * mz[:, 0], 
            mz[:, 0] - delta * mz[:, 0]
        ), 
        axis=0))


def calculateIonMobilityDistance(
    peptideMatchedMassSpectrumsIonMobility: np.ndarray,
    peptideIonMobility: float,
    peptideMs2IonMobilityDistance: List[np.ndarray]
):
    """
    计算肽段淌度和实验图谱淌度的绝对值差值, 方便我们后续筛选相关实验图谱, 这个信息非常重要 
    
    `ionMobilityDistance[k]` = |peptideIonMobility - ms2IonMobility[k]|
    
    ### Input Parameters:
    -   `peptideMatchedMassSpectrumsIonMobility`: 肽段匹配的实验图谱淌度值序列
    -   `peptideIonMobility`: 肽段理论淌度值
    -   `peptideMs2IonMobilityDistance`: 存储肽段和实验图谱计算得到的淌度差值
    """
    # 肽段和实验图谱的淌度信息差值
    ionMobilityDistance = np.abs(peptideMatchedMassSpectrumsIonMobility - peptideIonMobility)
    peptideMs2IonMobilityDistance.append(ionMobilityDistance)
    
def calculateMassSpectrumsFeatures():
    """
    提高文件的可拓展性, 在这个函数中, 你可以自定义一些谱图的特征计算函数, 用于筛选匹配程度更高的实验图谱
    """
    pass

def calculateIntesityDistanceSum(
    peptideMatchedMassSpectrumsPeakIntensity: np.ndarray,
    peptideMassSpectrumPeakIntensity: np.ndarray,
    peptideMs2PeaksIntensityDistanceSum:List[np.ndarray],
) -> None:
    """
    我们这里只对实验图谱进行峰强度归一化, 我们在生成图谱的时其中的肽段参考图谱已经归一化了
    -   `k` 表示实验图谱的下标, `i` 表示峰的下标

    1. `ms2IntensityNormalize[k][i]` = intensity[k][i] / \\sum_{i = 0}^{n - 1} intensity[k][i] 
    
    (正常情况我们还需要加一个 `epsilon`, 避免峰全为 0 时出现除 0 操作)
    
    2. `distance[k][i]` = |peptideIntensityNormalize[i] - ms2IntensityNormalize[k][i]|
    3. `distanceSum[k]` = \\sum_{i = 0}^{n - 1} distance[k][i]

    将 `diatanceSum` 存入对应列表
    
    ### Input Parameters:
    -   `peptideMatchedMassSpectrumsPeakIntensity`: 肽段匹配的实验图谱峰强度序列
    -   `peptideMassSpectrumPeakIntensity`: 肽段理论图谱峰强度
    -   `peptideMs2PeaksIntensityDistanceSum`: 存储肽段和实验图谱计算得到的归一化哈密顿距离和
    """
    # 先计算出归一化的峰强度向量, 再将其和肽段参考图谱的峰强度进行相减再去绝对值, 计算哈密顿距离
    # 防止出现除 0
    ms2PeaksIntensitySum = np.sum(peptideMatchedMassSpectrumsPeakIntensity, axis=1, keepdims=True) + 1e-7
    ms2PeaksIntensityNormalize = peptideMatchedMassSpectrumsPeakIntensity / ms2PeaksIntensitySum
    ms2PeaksIntensityNormalize[ms2PeaksIntensityNormalize == 0.0] = 1.0
    intensityDistance = np.abs(peptideMassSpectrumPeakIntensity - ms2PeaksIntensityNormalize)
    intensityDistanceSum = np.sum(intensityDistance, axis=1)
    peptideMs2PeaksIntensityDistanceSum.append(intensityDistanceSum)