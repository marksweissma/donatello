from donatello.examples import intent_simple, intent_pipeline, intent_line_graph
# add full

# temp integration tests until units tests are writen


def test_simple():
    s = intent_simple.load_sculpture()
    s.fit()
    assert s.measurements.crossValidation
    assert s.measurements.crossValidation.roc_auc_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.roc_auc_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.average_precision_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.average_precision_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean()[0] > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)


def test_pipeline():
    s = intent_pipeline.load_sculpture()
    s.fit()
    assert s.measurements.crossValidation
    assert s.measurements.crossValidation.roc_auc_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.roc_auc_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.average_precision_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.average_precision_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean()[0] > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)


def test_line_graph():
    s = intent_line_graph.load_sculpture()
    s.fit()
    assert s.measurements.crossValidation
    assert s.measurements.crossValidation.roc_auc_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.roc_auc_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.average_precision_score.mean.mean()[0] > 0.5
    assert s.measurements.crossValidation.average_precision_score.std.mean()[0] > 0
    assert s.measurements.crossValidation.feature_weights.mean.notnull().mean()[0] > .9
    assert s.measurements.crossValidation.ThresholdRates.mean.shape == (101, 12)
