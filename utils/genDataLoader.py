import torch
from .genDataSet import DIADataSet, generateTrainDataSet, generateTestDataSet, TrainDataSplitMode
from torch.utils.data import DataLoader, ConcatDataset, Subset
from typing import Union


def generateDataLoader(
    batchSize: int,
    dataSet: Union[DIADataSet, ConcatDataset, Subset],
    isShuffle: bool = True
):
    """
        由输入的 `dataset` 生成对应的 `dataLoader`
    ### Input Parameters:
    -   `batchSize`: batch 大小
    -   `dataSet`: 数据集
    -   `isShuffle`: 是否随机化
    
    ### Return:
    -   对应的 `DataLoader`
    """
    return DataLoader(
            dataset= dataSet, 
            batch_size = batchSize,
            shuffle = isShuffle,
            generator=torch.Generator().manual_seed(0),
            num_workers = 8)

def generateTrainDataLoader(
    batchSize: int,
    peptideMs2PeakNum: int,
    data,
    decoyData,
    lengthValDataSet: int,
    decoyLengthValDataSet: int,
    trainDataSplitMode: TrainDataSplitMode
):
    """
        由输入的原始训练数据 `data` 生成对应的 `dataLoader`

        IF `trainDataSplitMode == TrainDataSplitMode.TRAIN_VALIDATE_SPLIT`
            使用 `trainDataLoader, `valDataLoader`, `_` 进行接收

        ELIF `trainDataSplitMode == TrainDataSplitMode.TARGET_DECOY_TRAIN_VALIDATE`
            使用 `trainData`, `valData`, `decoyValData` 进行接收
    ### Input Parameters:
    -   `batchSize`: batch 大小
    -   `peptideMs2PeakNum`: 肽段参考图谱峰的个数
    -   `data`: 正库训练数据
    -   `decoyData`: 反库训练数据
    -   `peptideMs2PeakNum`: 肽段参考图谱峰数量 
    -   `lengthValDataSet`: 正库验证集数目 (当采取第一种划分时, 这个表示的是总的验证集的数目)
    -   `decoyLengthValDataSet`: 反库验证集数目
    -   `trainDataSplitMode`: 训练数据划分采取的模式
    
    ### Return:
    -   对应的 `DataLoader`
    """
    trainData, valData, decoyValData = generateTrainDataSet(
                                            data, 
                                            decoyData,
                                            peptideMs2PeakNum,
                                            lengthValDataSet,
                                            decoyLengthValDataSet,
                                            trainDataSplitMode
                                        )
    return (generateDataLoader(batchSize, trainData), 
            generateDataLoader(batchSize, valData), 
            generateDataLoader(batchSize, decoyValData)
    )

def generateTestDataLoader(
    batchSize: int,
    peptideMs2PeakNum: int,
    data,
    decoyData
):
    """
        由输入的原始测试数据 `data` 生成对应的 `dataLoader`

    ### Input Parameters:
    -   `batchSize`: batch 大小
    -   `peptideMs2PeakNum`: 肽段参考图谱峰的个数
    -   `data`: 正库测试数据
    -   `decoyData`: 反库测试数据
    
    ### Return:
    -   正库测试数据对应的 `DataLoader`
    -   `peptideRelativeInfo`: 正库肽段相关信息数据 (file, window, (peptideName, charge))
    -   反库测试数据对应的 `DataLoader`
    -   `decoyPeptideRelativeInfo`: 反库肽段相关信息数据 (file, window, (peptideName, charge))
    """
    (
        testData, 
        peptideRelativeInfo, 
        decoyTestData, 
        decoyPeptideRelativeInfo
    ) = generateTestDataSet(data, decoyData, peptideMs2PeakNum)

    return (generateDataLoader(batchSize, testData, isShuffle = False), 
            peptideRelativeInfo, 
            generateDataLoader(batchSize, decoyTestData, isShuffle=False), 
            decoyPeptideRelativeInfo
    ) 
