import tensorflow as tf

_EPS = 1e-8

# Helper functions

def _prep(y_true, y_pred):
    """Flatten and cast to float32."""
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    return y_true, y_pred

def _binarize(y_pred, threshold):
    thr = tf.cast(threshold, tf.float32)
    return tf.cast(y_pred >= thr, tf.float32)


# Performance metrics from total counts

class PrecisionMetric(tf.keras.metrics.Metric):
    def __init__(self, name="precision", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _prep(y_true, y_pred)
        y_hat = _binarize(y_pred, self.threshold)
        self.tp.assign_add(tf.reduce_sum(y_hat * y_true))
        self.fp.assign_add(tf.reduce_sum(y_hat * (1.0 - y_true)))

    def result(self):
        return self.tp / (self.tp + self.fp + _EPS)

    def reset_states(self):
        self.tp.assign(0.0); self.fp.assign(0.0)

class RecallMetric(tf.keras.metrics.Metric):
    def __init__(self, name="recall", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _prep(y_true, y_pred)
        y_hat = _binarize(y_pred, self.threshold)
        self.tp.assign_add(tf.reduce_sum(y_hat * y_true))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_hat) * y_true))

    def result(self):
        return self.tp / (self.tp + self.fn + _EPS)

    def reset_states(self):
        self.tp.assign(0.0); self.fn.assign(0.0)

class AccuracyMetric(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=tf.float32)
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _prep(y_true, y_pred)
        y_hat = _binarize(y_pred, self.threshold)
        self.tp.assign_add(tf.reduce_sum(y_hat * y_true))
        self.fp.assign_add(tf.reduce_sum(y_hat * (1.0 - y_true)))
        self.tn.assign_add(tf.reduce_sum((1.0 - y_hat) * (1.0 - y_true)))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_hat) * y_true))

    def result(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + _EPS)

    def reset_states(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.tn.assign(0.0); self.fn.assign(0.0)

class F1Metric(tf.keras.metrics.Metric):
    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _prep(y_true, y_pred)
        y_hat = _binarize(y_pred, self.threshold)
        self.tp.assign_add(tf.reduce_sum(y_hat * y_true))
        self.fp.assign_add(tf.reduce_sum(y_hat * (1.0 - y_true)))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_hat) * y_true))

    def result(self):
        return (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + _EPS)

    def reset_states(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

class MCCMetric(tf.keras.metrics.Metric):
    def __init__(self, name="mcc", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros", dtype=tf.float32)
        self.tn = self.add_weight(name="tn", shape=(), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = _prep(y_true, y_pred)
        y_hat = _binarize(y_pred, self.threshold)
        self.tp.assign_add(tf.reduce_sum(y_hat * y_true))
        self.fp.assign_add(tf.reduce_sum(y_hat * (1.0 - y_true)))
        self.tn.assign_add(tf.reduce_sum((1.0 - y_hat) * (1.0 - y_true)))
        self.fn.assign_add(tf.reduce_sum((1.0 - y_hat) * y_true))

    def result(self):
        num = (self.tp * self.tn) - (self.fp * self.fn)
        den = tf.sqrt(
            (self.tp + self.fp)
            * (self.tp + self.fn)
            * (self.tn + self.fp)
            * (self.tn + self.fn)
        )
        return num / (den + _EPS)

    def reset_states(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.tn.assign(0.0); self.fn.assign(0.0)