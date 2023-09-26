import generatedata.identify.genTrainData as identifyTrain
import generatedata.identify.genTestData as identifyTest
import generatedata.quant.genTrainData as quantTrain
import generatedata.quant.genTestData as quantTest

if __name__ == "__main__":
    print("start: generate identify target and decoy train data!")
    identifyTrain.main()
    print("start: generate identify target and decoy test data!")
    identifyTest.main()
    print("start: generate quant target train data!")
    # quantTrain.main()
    print("start: generate quant target test data!")
    quantTest.main()
