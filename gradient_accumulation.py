# ===================================================
# GRADIENT ACCUMULATION WRAPPER
# ===================================================
# Purpose: Enable larger effective batch sizes on memory-limited GPUs
# by accumulating gradients over multiple mini-batches before applying.
#
# Usage:
#   model = build_unet()
#   ga_model = GradientAccumulationModel(model, accumulation_steps=4)
#   ga_model.compile(optimizer=..., loss=..., metrics=...)
#   ga_model.fit(...)

import tensorflow as tf


class GradientAccumulationModel(tf.keras.Model):
    """Model wrapper for gradient accumulation.
    
    Accumulates gradients over `accumulation_steps` mini-batches,
    then applies the averaged gradient. This simulates a larger
    effective batch size without increasing memory usage.
    
    Example:
        With batch_size=16 and accumulation_steps=4,
        effective batch size = 16 * 4 = 64.
    """

    def __init__(self, inner_model, accumulation_steps=4, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.accumulation_steps = accumulation_steps
        self._step_counter = None
        self._gradient_accumulator = None

    def _init_accumulators(self):
        """Lazily initialize gradient accumulators."""
        if self._gradient_accumulator is None:
            self._gradient_accumulator = [
                tf.Variable(
                    tf.zeros_like(var),
                    trainable=False,
                    name=f"grad_accum_{i}",
                    aggregation=tf.VariableAggregation.NONE,
                )
                for i, var in enumerate(self.inner_model.trainable_variables)
            ]
            self._step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def call(self, inputs, training=None, mask=None):
        return self.inner_model(inputs, training=training, mask=mask)

    @property
    def trainable_variables(self):
        return self.inner_model.trainable_variables

    @property
    def non_trainable_variables(self):
        return self.inner_model.non_trainable_variables

    def train_step(self, data):
        x, y = data
        self._init_accumulators()

        with tf.GradientTape() as tape:
            y_pred = self.inner_model(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.inner_model.trainable_variables)

        # Accumulate gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self._gradient_accumulator[i].assign_add(grad)

        self._step_counter.assign_add(1)

        # Apply accumulated gradients when we've done enough steps
        if self._step_counter % self.accumulation_steps == 0:
            # Average the gradients
            avg_grads = [
                acc / tf.cast(self.accumulation_steps, tf.float32)
                for acc in self._gradient_accumulator
            ]
            self.optimizer.apply_gradients(
                zip(avg_grads, self.inner_model.trainable_variables)
            )
            # Reset accumulators
            for acc in self._gradient_accumulator:
                acc.assign(tf.zeros_like(acc))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self.inner_model(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super().get_config()
        config.update({
            "accumulation_steps": self.accumulation_steps,
        })
        return config


def wrap_with_gradient_accumulation(model, accumulation_steps=4):
    """Convenience function to wrap a model with gradient accumulation.
    
    Args:
        model: A compiled or uncompiled tf.keras.Model
        accumulation_steps: Number of mini-batches to accumulate before applying
        
    Returns:
        GradientAccumulationModel wrapping the input model
    """
    ga_model = GradientAccumulationModel(model, accumulation_steps=accumulation_steps)
    print(f"[OK] Gradient accumulation enabled: {accumulation_steps} steps")
    print(f"     Effective batch size multiplier: {accumulation_steps}x")
    return ga_model


print("✓ Gradient accumulation module ready.")
