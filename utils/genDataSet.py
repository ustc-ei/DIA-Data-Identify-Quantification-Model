import torch
from enum import Enum
from torch.utils.data import Dataset, random_split, ConcatDataset, Subset
from typing import Any, Tuple, Union
from .dataPretreatment import (
    peptideMatchedMs2IonMobilityPretreatment, 
    peptideMs2IonMobilityPretreatment
)


class DIADataSet(Dataset):
    def __init__(self,
                 peptideMatchedMassSpectrumsPeaks,
                 peptideMatchedMassSpectrumsIonMobility,
                 peptideMassSpectrumPeaks,
                 peptideMassSpectrumIonMobility,
                 peptideTarget) -> None:
        super(DIADataSet, self).__init__()

        self.peptideMatchedMassSpectrumsPeaks = peptideMatchedMassSpectrumsPeaks
        self.peptideMatchedMassSpectrumsIonMobility = peptideMatchedMassSpectrumsIonMobility
        self.peptideMassSpectrumPeaks = peptideMassSpectrumPeaks
        self.peptideMassSpectrumIonMobility = peptideMassSpectrumIonMobility
        self.peptideTarget = peptideTarget

        self.length = len(self.peptideMassSpectrumIonMobility)
    
    def __getitem__(self, index) -> Any:
        return (
            self.peptideMatchedMassSpectrumsPeaks[index],
            self.peptideMatchedMassSpectrumsIonMobility[index],
            self.peptideMassSpectrumPeaks[index],
            self.peptideMassSpectrumIonMobility[index],
            self.peptideTarget[index]
        )
    
    def __len__(self):
        return self.length

def splitTrainAndValDataSet(
    trainDataSet: Union[ConcatDataset, Subset, DIADataSet],
    lengthValDataSet: int
) -> Tuple[Union[ConcatDataset, Subset, DIADataSet],...]:
    """
    ### Input Parameters:
    -   `trainDataSet`: 训练数据
    -   `lengthValDataSet`: 验证集大小

    ### Return:
    -   `trainDataSet`: 训练集
    -   `valDataSet`: 验证集
    """

    trainDataSet, valDataSet = random_split(
        trainDataSet,
        [len(trainDataSet) - lengthValDataSet, lengthValDataSet],
        generator=torch.Generator().manual_seed(0),
    )

    return trainDataSet, valDataSet

def generateDataSet(
    data,
    peptideMs2PeakNum: int,
):
    """
        传入原始数据之后, 将其经过预处理之后, 输出 `dataset` 和肽段相关信息数据

    ### Input Parameters:
    -   `data`: 原始数据
    -   `peptideMs2PeakNum`: 峰数目

    ### Return:
    -   `DIADataSet`: 预处理过后的数据集
    -   `peptideRelativeInfo`: 肽段相关信息数据
    """
    (   
        peptideMatchedMassSpectrumPeaks,
        peptideMassSpectrumPeaks,
        peptideTarget,
        peptideRelativeInfo,
        peptideMatchedMassSpectrumIonMobility,
        peptideIonMobility
    ) = data[0], data[1], data[2], data[3], data[4], data[5]

    peptideMatchedMassSpectrumIonMobility = peptideMatchedMs2IonMobilityPretreatment(peptideMatchedMassSpectrumIonMobility, peptideMs2PeakNum)
    peptideIonMobility = peptideMs2IonMobilityPretreatment(peptideIonMobility, peptideMs2PeakNum)
    return DIADataSet(
        peptideMatchedMassSpectrumPeaks,
        peptideMatchedMassSpectrumIonMobility,
        peptideMassSpectrumPeaks,
        peptideIonMobility,
        peptideTarget
    ), peptideRelativeInfo

class TrainDataSplitMode(Enum):
    """
    -   `TRAIN_VALIDATE_SPLIT`: 正常的训练集/验证集划分
    -   `TARGET_DECOY_TRAIN_VALIDATE`: `正库训练集`、`正库验证集`、`反库训练集`、`反库验证集`划分方式
    """
    TRAIN_VALIDATE_SPLIT = 0
    TARGET_DECOY_TRAIN_VALIDATE = 1

def generateTrainDataSet(
    data,
    decoyData, 
    peptideMs2PeakNum: int,
    lengthValDataSet: int, 
    decoyLengthValDataSet: int,
    trainDataSplitMode: TrainDataSplitMode
):
    """
    ### Input Parameters: 
    -   `data`: 正库训练数据
    -   `decoyData`: 反库训练数据
    -   `peptideMs2PeakNum`: 肽段参考图谱峰数量 
    -   `lengthValDataSet`: 正库验证集数目 (当采取第一种划分时, 这个表示的是总的验证集的数目)
    -   `decoyLengthValDataSet`: 反库验证集数目
    -   `trainDataSplitMode`: 训练数据划分采取的模式
    
    ### Return:
    1. 采取训练集验证集划分方式
    -   `trainData`: 训练集
    -   `valData`: 验证集
    2. 采取`正库训练集`、`正库验证集`、`反库训练集`、`反库验证集` 划分方式
    -   `trainData`: 正库/反库数据合并后的训练集
    -   `valData`: 正库验证集
    -   `decoyValData`: 反库验证集
    """
    dataSet, _ = generateDataSet(data, peptideMs2PeakNum)
    decoyDataSet, _ = generateDataSet(decoyData, peptideMs2PeakNum)

    decoyTrainData, decoyValData = splitTrainAndValDataSet(decoyDataSet, decoyLengthValDataSet)

    if trainDataSplitMode == TrainDataSplitMode.TRAIN_VALIDATE_SPLIT:
        mergedDataSet = mergeDataSet(dataSet, decoyDataSet)
        trainData, valData = splitTrainAndValDataSet(mergedDataSet, lengthValDataSet)
        return trainData, valData, decoyValData
    
    elif trainDataSplitMode == TrainDataSplitMode.TARGET_DECOY_TRAIN_VALIDATE:
        trainData, valData = splitTrainAndValDataSet(dataSet, lengthValDataSet)
        trainData = mergeDataSet(trainData, decoyTrainData)
        return trainData, valData, decoyValData
    
def generateTestDataSet(
    data,
    decoyData, 
    peptideMs2PeakNum: int
):
    """
    ### Input Parameters: 
    -   `data`: 正库测试数据
    -   `decoyData`: 反库测试数据
    -   `peptideMs2PeakNum`: 肽段参考图谱峰数量 

    ### Return:
    -   `dataSet`: 正库测试数据集
    -   `peptideRelativeInfo`: 正库肽段相关信息数据 (file, window, (peptideName, charge))
    -   `decoyDataSet`: 反库测试数据集
    -   `decoyPeptideRelativeInfo`: 反库肽段相关信息数据 (file, window, (peptideName, charge))
    """

    testData, peptideRelativeInfo = generateDataSet(data, peptideMs2PeakNum)
    testDecoyData, decoyPeptideRelativeInfo = generateDataSet(decoyData, peptideMs2PeakNum)
    return testData, peptideRelativeInfo, testDecoyData, decoyPeptideRelativeInfo

def mergeDataSet(*args):
    """
    多个数据集进行合并
    """
    return ConcatDataset([*args])

