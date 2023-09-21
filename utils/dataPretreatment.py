import numpy as np


def peptideMatchedMs2IonMobilityPretreatment(
    peptideMatchedMs2IonMobility: np.ndarray,
    peptideMassSpectrumPeakNum: int
):
    """
    拓展为向量
    [[0.1, 0.2, 0.3],
     [0.2, 0.3, 0.4]]  > [[[0.1, 0.1, 0.1],
                           [0.2, 0.2, 0.2],
                           [0.3, 0.3, 0.3]],
                           [[0.2, 0.2, 0.2],
                           [0.3, 0.3, 0.3],
                           [0.4, 0.4, 0.4]]]
    """ 
    result = []
    for ionMobilitys in peptideMatchedMs2IonMobility:
        ms2IonMobilitys = []
        for ionMobility in ionMobilitys:
            ionMobilityExtension = np.repeat(ionMobility, peptideMassSpectrumPeakNum)
            ms2IonMobilitys.append(ionMobilityExtension)
        result.append(ms2IonMobilitys)
    return np.array(result)

def peptideMs2IonMobilityPretreatment(
    peptideMs2IonMobility: np.ndarray,
    peptideMassSpectrumPeakNum: int
):
    """
    拓展为向量
    [0.1, 0.2, 0.3] > [[0.1, 0.1, 0.1],
                       [0.2, 0.2, 0.2],
                       [0.3, 0.3, 0.3]]
    """
    result = []
    for ionMobility in peptideMs2IonMobility:
        ionMobilityExtension = np.repeat(ionMobility, peptideMassSpectrumPeakNum)
        result.append(ionMobilityExtension)

    return np.array(result)
