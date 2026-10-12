@testable import IntentCore

enum OnlinePathTestFixture {
    static func weights(identity: String, seed: String) -> OnlinePathMLPWeights {
        var value = OnlinePathMLPWeights(
            format: "online-path-mlp-f32-v1", modelVersion: OnlinePathMLPScorer.modelVersion,
            featureSchema: OnlinePathMLPScorer.featureSchema,
            numericFields: OnlinePathMLPScorer.numericFields,
            categoricalFields: OnlinePathMLPScorer.categoricalFields,
            categoricalBuckets: 512, tokenHashPerson: "OffRankV1",
            hashPerson: "SharedMLPPath1", hidden: 16,
            layer1Weights: Array(repeating: Array(repeating: 0, count: 543), count: 16),
            layer1Bias: Array(repeating: 0, count: 16),
            layer2Weights: Array(repeating: 0, count: 16), layer2Bias: 0,
            identityKeySHA256: identity, trainedThroughEventID: seed)
        value.layer1Bias[0] = 1
        value.layer2Weights[0] = 1
        value.learningRate = 0.001
        value.negativeLimit = 8
        value.seed = 20260929
        value.beta1 = 0.9
        value.beta2 = 0.999
        value.epsilon = 1e-8
        let matrix = Array(repeating: [Float](repeating: 0, count: 543), count: 16)
        let vector = [Float](repeating: 0, count: 16)
        value.optimizerState = OnlinePathMLPOptimizerState(
            layer1Weights: OnlinePathMLPMatrixMoments(expAvg: matrix, expAvgSq: matrix, step: 0),
            layer1Bias: OnlinePathMLPVectorMoments(expAvg: vector, expAvgSq: vector, step: 0),
            layer2Weights: OnlinePathMLPVectorMoments(expAvg: vector, expAvgSq: vector, step: 0),
            layer2Bias: OnlinePathMLPScalarMoments(expAvg: 0, expAvgSq: 0, step: 0))
        return value
    }

}
