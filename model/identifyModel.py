import torch
import torch.nn as nn


class IdentifyModel(nn.Module):
    def __init__(self, 
                 filterMassSpectrumNum: int,
                 peptideMassSepctrumPeakNum) -> None:
        super(IdentifyModel, self).__init__()
        """
        ### Input Parameters:
        -   filterMassSpectrumNum: 筛选过滤后的实验图谱数量
        -   peptideMassSepctrumPeakNum: 肽段理论图谱峰数量
        """
        
        self.filterMassSpectrumNum = filterMassSpectrumNum
        self.peptideMassSepctrumPeakNum = peptideMassSepctrumPeakNum

        self.dropout = nn.Dropout(0.1)

        # 定义肽段卷积层和肽段匹配的实验图谱卷积层

        self.peptideMatchMs2ConvBlock = nn.Sequential(
            nn.Conv1d(self.filterMassSpectrumNum, 64,kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            self.dropout
        )

        self.peptideMsConvBlock = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            self.dropout
        )

        self.mergeConvBlock = nn.Sequential(
            nn.Conv1d(256 * 4, 256, kernel_size=1),
            nn.ReLU(),
            self.dropout
        )

        self.linearBlock = nn.Sequential(
            nn.Linear(256 * self.peptideMassSepctrumPeakNum, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                peptideMatchedMassSpectrumsPeaks, # [batch, filterMassSpectrumNums, peptideMassSpectrumPeakNums, 2]
                peptideMatchedMassSpectrumsIonMobility, # [batch, filterMassSpectrumNums, peptideMassSpectrumPeakNums]
                peptideMassSpectrumPeaks,  # [batch, 1, peptideMassSpectrumPeakNums, 2]
                peptideMassSpectrumIonMobility): # [batch, 1, peptideMassSpectrumPeakNums]
        
        batchSize = peptideMatchedMassSpectrumsPeaks.shape[0]

        # [batch, filterMassSpectrumNums, peptideMassSpectrumPeakNums]
        peptideMatchedMassSpectrumsIntensity = peptideMatchedMassSpectrumsPeaks[:, :, :, 1]
        # 
        peptideMatchedMassSpectrumsIntensity = self.peptideMatchMs2ConvBlock(peptideMatchedMassSpectrumsIntensity)
        # peptideMatchedMassSpectrumsIntensity = self.dropout(peptideMatchedMassSpectrumsIntensity)
        # 
        peptideMatchedMassSpectrumsIonMobility = self.peptideMatchMs2ConvBlock(peptideMatchedMassSpectrumsIonMobility)
        # peptideMatchedMassSpectrumsIonMobility = self.dropout(peptideMatchedMassSpectrumsIonMobility)
        
        # [batch, 1, peptideMassSpectrumPeakNums]
        peptideMassSpectrumIntensity = peptideMassSpectrumPeaks[:, :, :, 1]
        # 
        peptideMassSpectrumIntensity = self.peptideMsConvBlock(peptideMassSpectrumIntensity)
        # peptideMassSpectrumIntensity = self.dropout(peptideMassSpectrumIntensity)
        # 
        peptideMassSpectrumIonMobility = self.peptideMsConvBlock(peptideMassSpectrumIonMobility)
        # peptideMassSpectrumIonMobility = self.dropout(peptideMassSpectrumIonMobility)

        # [batch, 1, peptideMassSpectrumPeakNums]
        x = torch.concat([peptideMassSpectrumIntensity, 
                          peptideMatchedMassSpectrumsIntensity, 
                          peptideMassSpectrumIonMobility,
                          peptideMatchedMassSpectrumsIonMobility], dim=1)
        
        x = self.mergeConvBlock(x)
        # x = self.dropout(x)
        x = x.view(batchSize, -1)
        x = self.linearBlock(x)
        return x
        