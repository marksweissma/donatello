from numpy import testing as npt

from donatello.examples import (intent_simple,
                                intent_pipeline,
                                intent_line_graph)
# add full


# temp integration tests until units tests are writen

def test_simple():
    s = intent_simple.load_sculpture()
    s.fit()
    assert s.mearurements.crossValidation
    assert s.mearurements.crossValidation.roc_auc_score.mean > 0.5
    assert s.mearurements.crossValidation.roc_auc_score.std > 0
    assert s.mearurements.crossValidation.average_precision_score.mean > 0.5
    assert s.mearurements.crossValidation.average_precision_score.std > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean() > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)


def test_pipeline():
    s = intent_pipeline.load_sculpture()
    s.fit()
    assert s.mearurements.crossValidation
    assert s.mearurements.crossValidation.roc_auc_score.mean > 0.5
    assert s.mearurements.crossValidation.roc_auc_score.std > 0
    assert s.mearurements.crossValidation.average_precision_score.mean > 0.5
    assert s.mearurements.crossValidation.average_precision_score.std > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean() > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)


def test_line_graph():
    s = intent_line_graph.load_sculpture()
    s.fit()
    assert s.mearurements.crossValidation
    assert s.mearurements.crossValidation.roc_auc_score.mean > 0.5
    assert s.mearurements.crossValidation.roc_auc_score.std > 0
    assert s.mearurements.crossValidation.average_precision_score.mean > 0.5
    assert s.mearurements.crossValidation.average_precision_score.std > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean() > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)
