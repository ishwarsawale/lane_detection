"""
Backend branch which is mainly used for binary and instance segmentation loss calculation
"""
import tensorflow as tf

from config import global_config
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class SegNetDataFeeder(cnn_basenet.CNNBaseModel):
    """
    Backend branch which is mainly used for binary and instance segmentation loss calculation
    """
    def __init__(self, phase):
        """
        init backend
        :param phase: train or test
        """
        super(SegNetDataFeeder, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)


        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )
        print(f'loss at gradients is {loss}')

        return loss

    def compute_loss(self, binary_seg_logits, binary_label,
                     name, reuse):
        """
        compute loss
        :param binary_seg_logits:
        :param binary_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_label_onehot = tf.one_hot(
                    tf.reshape(
                        tf.cast(binary_label, tf.int32),
                        shape=[binary_label.get_shape().as_list()[0],
                               binary_label.get_shape().as_list()[1],
                               binary_label.get_shape().as_list()[2]]),
                    depth=CFG.TRAIN.CLASSES_NUMS,
                    axis=-1
                )

                binary_label_plain = tf.reshape(
                    binary_label,
                    shape=[binary_label.get_shape().as_list()[0] *
                           binary_label.get_shape().as_list()[1] *
                           binary_label.get_shape().as_list()[2] *
                           binary_label.get_shape().as_list()[3]])
                unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
                counts = tf.cast(counts, tf.float32)
                inverse_weights = tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )

                binary_segmenatation_loss = self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels=binary_label_onehot,
                    logits=binary_seg_logits,
                    classes_weights=inverse_weights
                )

            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = binary_segmenatation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'binary_seg_loss': binary_segmenatation_loss
            }

        return ret

    def inference(self, binary_seg_logits, name, reuse):
        """

        :param binary_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

        return binary_seg_prediction
