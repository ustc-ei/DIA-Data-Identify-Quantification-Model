import os
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import numpy as np
from ..utils.calculateFunctions import (
    calculateMassSpectrumMzTranverse,
    calculateIonMobilityDistance,
    calculateIntesityDistanceSum,
)
from ..utils.peaksProcessFunctions import (
    divideLibraryByWindows,
    peaksMatch,
    fillMassSpectrumWithZeros
)


def peptideMatchMassSpectrumByPeaks(
    libraryPath: str,
    massSpectrumFileRootPath: str,
    targetFilePath: str,
    peptideMatchMs2PeakNums: int,
    tol: int = 15,
    isDecoy: bool = False
):
    """
    读取图谱库(正库/反库), 利用图谱库中肽段的理论图谱峰去匹配实验图谱, 峰匹配成功数量大于 3 则表示肽段与实验图谱匹配成功

    我们只保留匹配成功的峰信息, 其余信息设为 0, 类似于下面

    -   [肽段对应峰质荷比, 实验图谱对应峰强度]
    -   [0, 0]

    如果某肽段一张实验图谱都没有匹配到, 则抛弃该肽段, 直接认定肽段不在样品中, 即直接打分为 0

    峰匹配操作具体情形可以参见 `peaksMatch` 函数

    最终返回下面的数据格式
    {
        -   (fileName, window): [

            -   肽段匹配成功的多张图谱,
            -   肽段理论图谱,
            -   肽段标签,
            -   (肽段名称, 带电荷数),
            -   匹配成功的图谱对应的淌读信息
        ]
    }

    ### Input Parameters:
    -   `libraryPath`: 图谱库文件路径
    -   `massSpectrumFilePathList`: 经峰合并操作后的质谱数据文件路径列表
    -   `peptideMatchMs2PeakNums`: 肽段和实验图谱峰匹配成功个数 (肽段和实验图谱匹配成功的标志)
    -   `targetFilePath`: 肽段的标签文件路径 (我们是使用 Spectronaut 的结果进行训练)
    -   `tol`: 峰匹配的上限值 (tims-TOF 仪器产生的数据为 15, TripleTOF 仪器产生的数据为 30, ppm 浓度为 1*e-6)
    -   `isDecoy`: 是否为诱饵库 (直接打 0 / 选择性打 1)
    ### Return:
    -   fileWindowPeptideMatchMassSpectrumInfo: 字典, key 值为 (fileName, window), value 为肽段匹配实验图谱信息及标签

    具体信息如上所示 
    """
    delta = tol * 1e-6
    fileWindowPeptideMatchMassSpectrumInfo: Dict[Tuple, List] = {}
    targetLabel = np.load(targetFilePath, allow_pickle=True).item()
    root = massSpectrumFileRootPath + "/"
    massSpectrumFilePathList = [root + filePath for filePath in os.listdir(
        massSpectrumFileRootPath) if filePath.endswith(".npy")]
    for msFilePath in massSpectrumFilePathList:
        fileName = msFilePath.split(".npy")[0].split("/")[-1]
        print(fileName)
        # the mass spectrum with [array([mz, intensity]), scan window, RT, index, ion mobility]
        massSpectrums = np.load(msFilePath, allow_pickle=True).item()
        windows = massSpectrums.keys()
        print("divide library by windows!")
        library = divideLibraryByWindows(libraryPath, windows)
        print("end!")
        # 由于我们图谱库和实验图谱都是按照窗口划分的, 因此我们最外面应该是窗口循环
        peptideAllNum = 0
        for window in windows:        # 匹配也只在窗口内进行匹配
            mzAfterTranverseList = []  # 存储每张质谱图左右偏移后的质荷比序列

            # 先将每张实验图谱的质荷比进行左右偏移, 将其存入 mzForMassSpectrum 中
            for ms2 in massSpectrums[window]:
                mzAfterTranverse = calculateMassSpectrumMzTranverse(
                    ms2[0], delta)
                mzAfterTranverseList.append(mzAfterTranverse)

            peptideNum = 0
            for peptideNameWithCharge, peptideInfo in tqdm(library[window].items(), f"{window}"):
                candidateMs2 = []  # 存储肽段匹配成功的实验图谱
                candidateMs2IonMobility = []  # 存储肽段匹配成功的实验图谱的离子淌度 IonMobility
                peptideMassSpectrum = peptideInfo["Spectrum"]

                for ms2_i, ms2 in enumerate(massSpectrums[window]):
                    matchedPeaks = []  # 存储匹配的峰
                    massSpectrumPeaks = ms2[0]
                    mzAfterTranverse = mzAfterTranverseList[ms2_i]
                    insertIndex, peakMatchedNum = peaksMatch(
                        peptideMassSpectrum[:, 0], mzAfterTranverse)
                    if peakMatchedNum < peptideMatchMs2PeakNums:
                        continue
                    for i, index in enumerate(insertIndex):
                        if index == -1:
                            matchedPeaks.append([0.0, 0.0])
                        else:
                            matchedPeaks.append(
                                [peptideMassSpectrum[i, 0], massSpectrumPeaks[index, 1]])
                    candidateMs2.append(matchedPeaks)
                    candidateMs2IonMobility.append(ms2[-1])
                # end for massSpectrums
                if len(candidateMs2) == 0:  # 一张实验图谱都没有匹配上
                    continue

                if (fileName, window) not in fileWindowPeptideMatchMassSpectrumInfo.keys():
                    fileWindowPeptideMatchMassSpectrumInfo[(
                        fileName, window)] = []

                if not isDecoy:
                    if peptideNameWithCharge in targetLabel[fileName].keys():
                        fileWindowPeptideMatchMassSpectrumInfo[(fileName, window)].append([
                            np.array(candidateMs2),
                            peptideMassSpectrum,
                            1,
                            peptideNameWithCharge,
                            candidateMs2IonMobility
                        ])
                else:
                    fileWindowPeptideMatchMassSpectrumInfo[(fileName, window)].append([
                        np.array(candidateMs2),
                        peptideMassSpectrum,
                        0,
                        peptideNameWithCharge,
                        candidateMs2IonMobility
                    ])
                peptideNum += 1
            # end for peptide
            print(
                f"the {window} in file, {peptideNum} peptides has matched at least one Mass Spectrum")
            peptideAllNum += peptideNum
        # end for window
        print(
            f"file, {peptideAllNum} peptides has matched at least one Mass Spectrum")
    # end for file
    return np.array(fileWindowPeptideMatchMassSpectrumInfo, dtype=object)


def filterNumsMassSpectrumByFeatures(
    libraryPath: str,
    peptideMatchedMassSpectrumsInfo: Dict[Tuple[str, Tuple[int, int]], Any],
    filterMassSpectrumNums: int,
    peptidePeakNums: int,
    mobilityDistanceThreshold: int,
    isDecoy: bool = False
):
    """
    把肽段相关信息存入对应的列表中
    -   `peptideMatchedMassSpectrumPeaks`: 肽段匹配的实验图谱峰信息
    -   `peptideMatchedMassSpectrumIonMobility`: 肽段匹配的实验图谱淌度信息
    -   `peptideMassSpectrumPeaks`: 肽段参考图谱峰信息
    -   `peptideTarget`: 肽段标签
    -   `peptideIonMobility`: 肽段淌度信息
    -   `peptideRealtiveInfo`: 肽段相关信息 `(file, window, (name, charge))`

    我们需要预先输入一个数, 表示筛选肽段最相关的匹配成功的实验图谱数量

    如果某肽段匹配成功的实验图谱数小于该数, 则对 `肽段匹配的实验图谱峰信息` 和 `肽段匹配的实验图谱淌度信息` 进行 0 填充

    后续遍历肽段以及它匹配成功的实验图谱, 计算 `归一化哈密顿距离` 和 `淌度差值`

    计算得到淌度差值之后, 根据传入的淌度差阈值进行图谱筛选

    如果一张满足淌度阈值之内的图谱都没有, 则选择淌度值差最小的保留; 否则, 将所有满足淌度差值的图谱保留

    最后进行补 0

    再之后使用哈密顿距离进行筛选, 保留哈密顿距离排序之后前 s 个图谱.

    ### Input Parameters:
    -   `libraryPath`: 图谱库路径
    -   `peptideMatchedMassSpectrumsInfo`: 肽段匹配实验图谱的相关情况, 数据格式详情可见 `peptideMatchMassSpectrumByPeaks` 函数
    -   `filterMassSpectrumNums`: 需要筛选最相关的实验图谱数量
    -   `peptidePeakNums`: 肽段参考图谱峰的个数
    -   `mobilityDistanceThreshold`: 淌度差值阈值
    -   `isDecoy`: 是否为诱饵库

    ### Return:
    -   `TrainData`: 筛选过后的肽段最相关的 s 个实验图谱数据以及其对应的淌度信息
    """
    library = np.load(libraryPath, allow_pickle=True).item()
    # 将所有信息存入对应的列表中
    peptideMatchedMassSpectrumPeaks: List[np.ndarray] = []  # 肽段匹配的实验图谱峰信息
    # 肽段匹配的实验图谱淌度信息
    peptideMatchedMassSpectrumIonMobility: List[np.ndarray] = []
    peptideMassSpectrumPeaks: List[np.ndarray] = []  # 肽段参考图谱峰信息
    peptideTarget: List[int] = []  # 肽段标签
    peptideIonMobility: List[float] = []  # 肽段淌度信息
    # 肽段相关信息 (file, window, (name, charge))
    peptideRealtiveInfo: List[Tuple[str,
                                    Tuple[int, int], Tuple[str, str]]] = []

    for fileWindow, peptideInfo in tqdm(peptideMatchedMassSpectrumsInfo.items(), "extracting information!"):
        # peptideInfo[0], peptideInfo[-1] = fillMassSpectrumWithZeros(
        #                                         np.array(peptideInfo[0]),
        #                                         filterMassSpectrumNums,
        #                                         peptidePeakNums,
        #                                         np.array(peptideInfo[-1]))
        # 信息存储对应列表中
        for peptidei in range(len(peptideInfo)):
            peptideMatchedMassSpectrumPeaks.append(
                np.array(peptideInfo[peptidei][0]))
            peptideMatchedMassSpectrumIonMobility.append(
                np.array(peptideInfo[peptidei][-1]))
            peptideMassSpectrumPeaks.append(np.array(peptideInfo[peptidei][1]))
            peptideTarget.append(peptideInfo[peptidei][2])
            peptideIonMobility.append(
                library[peptideInfo[peptidei][-2]]["IonMobility"])
            peptideRealtiveInfo.append(
                (fileWindow[0], fileWindow[1], peptideInfo[peptidei][-2]))

    # 存储肽段和它匹配成功的实验图谱的淌度差值
    peptideMs2IonMobilityDistance: List[np.ndarray] = []

    # 计算肽段和匹配成功的图谱的淌度差值
    for ms2i, ms2IonMobility in enumerate(tqdm(peptideMatchedMassSpectrumIonMobility, "calculating IonMobility distance!")):
        calculateIonMobilityDistance(
            ms2IonMobility,
            peptideIonMobility[ms2i],
            peptideMs2IonMobilityDistance)

    # 存储肽段和它匹配成功的实验图谱的归一化哈密顿距离和
    peptideMs2PeaksIntensityDistanceSum: List[np.ndarray] = []

    for peptidei, ionMobilityDistance in enumerate(tqdm(peptideMs2IonMobilityDistance, "first filting mass Spectrums by IonMobility!")):
        # 筛选出淌度差值小于设定阈值的实验图谱对应下标
        selectedIndex = (ionMobilityDistance <
                         mobilityDistanceThreshold).nonzero()
        # 如果它匹配的所有的图谱淌度差都大于阈值, 则我们只留下淌度差值最小的, 再进行填充
        if len(selectedIndex) == 0:
            index = np.argmin(ionMobilityDistance)
            peptideMatchedMassSpectrumPeaks[peptidei] = peptideMatchedMassSpectrumPeaks[peptidei][index]
            peptideMatchedMassSpectrumIonMobility = peptideMatchedMassSpectrumIonMobility[
                peptidei][index]
        else:
            peptideMatchedMassSpectrumPeaks[peptidei] = peptideMatchedMassSpectrumPeaks[peptidei][selectedIndex]
            peptideMatchedMassSpectrumIonMobility[peptidei] = peptideMatchedMassSpectrumIonMobility[peptidei][selectedIndex]

        # 填充 0
        (peptideMatchedMassSpectrumPeaks[peptidei],
            peptideMatchedMassSpectrumIonMobility[peptidei]
         ) = fillMassSpectrumWithZeros(peptideMatchedMassSpectrumPeaks[peptidei],
                                       filterMassSpectrumNums,
                                       peptidePeakNums,
                                       peptideMatchedMassSpectrumIonMobility[peptidei])
        # 计算肽段和相关图谱的哈密顿距离和
        calculateIntesityDistanceSum(
            peptideMatchedMassSpectrumPeaks[peptidei][:, :, 1],
            peptideMassSpectrumPeaks[peptidei][:, 1],
            peptideMs2PeaksIntensityDistanceSum
        )
    # end for ionMobilityDistance

    # 通过一些图谱特征筛选特定数量的图谱
    for peptidei, PeaksIntensityDistanceSum in enumerate(tqdm(peptideMs2PeaksIntensityDistanceSum, "filter Mass Spectrums by Peaks features!")):
        sortIndex = np.argsort(PeaksIntensityDistanceSum)
        filterIndex = sortIndex[:filterMassSpectrumNums]
        filterIndex = np.sort(filterIndex)
        peptideMatchedMassSpectrumPeaks[peptidei] = peptideMatchedMassSpectrumPeaks[peptidei][filterIndex]
        peptideMatchedMassSpectrumIonMobility[peptidei] = peptideMatchedMassSpectrumIonMobility[peptidei][filterIndex]
    # end for PeaksIntensityDistanceSum

    """
    trainPeptideMatchedMassSpectrumPeaks: List[np.ndarray] = [] # 肽段匹配的实验图谱峰信息
    trainPeptideMatchedMassSpectrumIonMobility:List[np.ndarray] = [] # 肽段匹配的实验图谱淌度信息
    trainPeptideMassSpectrumPeaks:List[np.ndarray] = [] # 肽段参考图谱峰信息
    trainPeptideTarget: List[int] = [] # 肽段标签
    trainPeptideIonMobility: List[float] = [] # 肽段淌度信息
    trainPeptideRealtiveInfo: List[Tuple[str, Tuple[int, int], Tuple[str, str]]] = [] # 肽段相关信息 (file, window, (name, charge))         
    """
    fileDict = {
        file: set()
        for file in set(info[0] for info in peptideRealtiveInfo)
    }
    filterIndex = []

    # 只筛选正标签的, 感觉没必要, 之前在峰匹配阶段, 正库我们只保留了正样本!
    # 避免文件出现多条相同的肽段, 但是感觉这种担忧毫不必要
    for peptidei, target in enumerate(tqdm(peptideTarget)):
        if peptideRealtiveInfo[peptidei][2] not in fileDict[peptideRealtiveInfo[peptidei][0]]:
            fileDict[peptideRealtiveInfo[peptidei][0]].add(
                peptideRealtiveInfo[peptidei][2])
            if isDecoy or target == 1:
                filterIndex.append(peptidei)
                # trainPeptideMatchedMassSpectrumPeaks.append(peptideMatchedMassSpectrumPeaks[peptidei])
                # trainPeptideMatchedMassSpectrumIonMobility.append(peptideMatchedMassSpectrumIonMobility[peptidei])
                # trainPeptideMassSpectrumPeaks.append(peptideMassSpectrumPeaks[peptidei])
                # trainPeptideTarget.append(peptideTarget[peptidei])
                # trainPeptideIonMobility.append(peptideIonMobility[peptidei])
                # trainPeptideRealtiveInfo.append(peptideRealtiveInfo[peptidei])
    # end for target

    peptideMatchedMassSpectrumPeaks = np.array(
        peptideMatchedMassSpectrumPeaks)  # type: ignore
    peptideMassSpectrumPeaks = np.array(
        peptideMassSpectrumPeaks)  # type: ignore
    peptideTarget = np.array(peptideTarget)  # type: ignore
    peptideRealtiveInfo = np.array(
        peptideRealtiveInfo, dtype=object)  # type: ignore
    peptideMatchedMassSpectrumIonMobility = np.array(
        peptideMatchedMassSpectrumIonMobility)  # type: ignore
    peptideIonMobility = np.array(peptideIonMobility)  # type: ignore
    filterIndex = np.array(filterIndex)

    return np.array([
        list(peptideMatchedMassSpectrumPeaks[filterIndex]),
        list(peptideMassSpectrumPeaks[filterIndex]),
        list(peptideTarget[filterIndex]),  # type: ignore
        list(peptideRealtiveInfo[filterIndex]),
        list(peptideMatchedMassSpectrumIonMobility[filterIndex]),
        list(peptideIonMobility[filterIndex])  # type: ignore
    ], dtype=object)


def main():
    massSpectrumFileRootPath = "/data/xp/data/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/mzml/Identify/QC"
    peptideMatchMs2PeakNums = 3
    filterMassSpectrumNums = 6
    peptidePeakNums = 6
    mobilityDistanceThreshold = 100
    rootPath = massSpectrumFileRootPath + "/testCode/"
    libraryPath = rootPath + "library/20220112_MN_plasma_DDA_library_im_norm_peak6.npy"
    decoyLibraryPath = rootPath + \
        "library/20220112_MN_plasma_DDA_library_im_decoy_params_100_norm_peak6.npy"
    targetFilePath = rootPath + \
        "spectronautLabel/plasma_1_2_3_4_6_7_8_9_Spectronaut_identifyLabel.npy"
    tempDataSavePath = rootPath + \
        "trainData/identify/targetData/plasmaPeptideMatchMassSpectrumByPeaksData.npy"
    trainDataSavePath = rootPath + \
        "trainData/identify/targetData/plasmaPeptideMatchMassSpectrumfilterData.npy"
    decoyTempDataSavePath = rootPath + \
        "trainData/identify/decoyData/plasmaPeptideMatchMassSpectrumByPeaksDataDecoy.npy"
    decoyTrainDataSavePath = rootPath + \
        "trainData/identify/decoyData/plasmaPeptideMatchMassSpectrumfilterData.npy"

    # data = peptideMatchMassSpectrumByPeaks(
    #     libraryPath,
    #     massSpectrumFileRootPath,
    #     targetFilePath,
    #     peptideMatchMs2PeakNums
    # )
    # np.save(tempDataSavePath, data)

    # data = np.load(tempDataSavePath, allow_pickle=True).item()
    # filterData = filterNumsMassSpectrumByFeatures(
    #     libraryPath,
    #     data,
    #     filterMassSpectrumNums,
    #     peptidePeakNums,
    #     mobilityDistanceThreshold
    # )
    # np.save(trainDataSavePath, filterData)

    decoyData = peptideMatchMassSpectrumByPeaks(
        decoyLibraryPath,
        massSpectrumFileRootPath,
        targetFilePath,
        peptideMatchMs2PeakNums,
        isDecoy=True
    )
    np.save(decoyTempDataSavePath, decoyData)

    data = np.load(decoyTempDataSavePath, allow_pickle=True).item()
    filterData = filterNumsMassSpectrumByFeatures(
        decoyLibraryPath,
        data,
        filterMassSpectrumNums,
        peptidePeakNums,
        mobilityDistanceThreshold,
        isDecoy=True
    )
    np.save(decoyTrainDataSavePath, filterData)


if __name__ == "__main__":
    main()
    # import sys
    # args = sys.argv
    # libraryPath = args[1]
    # decoyLibraryPath = args[2]
    # targetFilePath = args[3]
    # massSpectrumFileRootPath = args[4]
    # peptideMatchMs2PeakNums = int(args[5])
    # massSpectrumFileRootPath = "/data/xp/data/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/mzml/Identify/QC"
    # peptideMatchMs2PeakNums = 3
    # filterMassSpectrumNums = 6
    # peptidePeakNums = 6
    # mobilityDistanceThreshold = 100
    # rootPath = "../../"
    # libraryPath = rootPath + "library/20220112_MN_plasma_DDA_library_im_norm_peak6.npy"
    # decoyLibraryPath = rootPath + "library/20220112_MN_plasma_DDA_library_im_decoy_params_100_norm_peak6.npy"
    # targetFilePath = rootPath + "spectronautLabel/plasma_1_2_3_4_6_7_8_9_Spectronaut_identifyLabel.npy"
    # tempDataSavePath = rootPath + "trainData/identify/targetData/plasmaPeptideMatchMassSpectrumByPeaksData.npy"
    # trainDataSavePath = rootPath + "trainData/identify/targetData/plasmaPeptideMatchMassSpectrumfilterData.npy"
    # decoyTempDataSavePath = rootPath + "trainData/identify/decoyData/plasmaPeptideMatchMassSpectrumByPeaksDataDecoy.npy"
    # decoyTrainDataSavePath = rootPath + "trainData/identify/decoyData/plasmaPeptideMatchMassSpectrumfilterData.npy"
    # print(libraryPath)
    # print(targetFilePath)
    # print(massSpectrumFileRootPath)
    # print(peptideMatchMs2PeakNums)
    # if len(args) > 6:
    #     tol = int(args[5])

    # python -u "genTrainData.py" 20220112_MN_plasma_DDA_library_im_norm_peak6.npy 20220112_MN_plasma_DDA_library_im_decoy_params_100_norm_peak6.npy plasma_1_2_3_4_6_7_8_9_Spectronaut_identifyLabel.npy /data/xp/data/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/mzml/Identify/QC 3
    # nohup python -u "genTrainData.py" 20220112_MN_plasma_DDA_library_im_norm_peak6.npy 20220112_MN_plasma_DDA_library_im_decoy_params_100_norm_peak6.npy plasma_1_2_3_4_6_7_8_9_Spectronaut_identifyLabel.npy /data/xp/data/tims-TOF_20211002_30min_LCH_MN_144-Plasma_SPEED-DIA_300ng_mix-qc/mzml/Identify/QC 3
    # data = peptideMatchMassSpectrumByPeaks(
    #     libraryPath,
    #     massSpectrumFileRootPath,
    #     targetFilePath,
    #     peptideMatchMs2PeakNums
    # )
    # np.save(tempDataSavePath, data)
    # data = np.load(tempDataSavePath, allow_pickle=True).item()
    # filterData = filterNumsMassSpectrumByFeatures(
    #     libraryPath,
    #     data,
    #     filterMassSpectrumNums,
    #     peptidePeakNums,
    #     mobilityDistanceThreshold
    # )
    # np.save(trainDataSavePath, filterData)

    # decoyData = peptideMatchMassSpectrumByPeaks(
    #     decoyLibraryPath,
    #     massSpectrumFileRootPath,
    #     targetFilePath,
    #     peptideMatchMs2PeakNums,
    #     isDecoy=True
    # )
    # np.save(decoyTempDataSavePath, decoyData)

    # data = np.load(decoyTempDataSavePath, allow_pickle=True).item()
    # filterData = filterNumsMassSpectrumByFeatures(
    #     decoyLibraryPath,
    #     data,
    #     filterMassSpectrumNums,
    #     peptidePeakNums,
    #     mobilityDistanceThreshold,
    #     isDecoy=True
    # )
    # np.save(decoyTrainDataSavePath, filterData)
