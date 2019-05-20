# vim: expandtab:ts=4:sw=4
import functools
import os
import numpy as np
import scipy.io as sio
import train_app
from datasets import bdd
from datasets import util
import nets.deep_sort.network_definition as net


class BddTracking(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids = bdd.read_train_split_to_str(
            self._dataset_dir)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        return filenames, ids

    def read_validation(self):
        filenames, ids, = bdd.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        return filenames, ids

    def read_test(self):
        return bdd.read_test_split_to_str(self._dataset_dir)


def main():
    arg_parser = train_app.create_default_argument_parser("bdd")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to BDD tracking dataset directory.",
        default="data/bdd-tracking-reid")
    arg_parser.add_argument(
        "--sdk_dir", help="Path to BDD baseline evaluation software.",
        default="/path/to/bdd/sdk")
    args = arg_parser.parse_args()
    dataset = BddTracking(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y = dataset.read_train()
        print("Train set size: %d images, %d identities" % (
            len(train_x), len(np.unique(train_y))))
        
        network_factory = net.create_network_factory(
            is_training=True, num_classes=bdd.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=bdd.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y = dataset.read_validation()
        print("Validation set size: %d images, %d identities" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=bdd.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, None,
            image_shape=bdd.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        # Export one specific model.
        gallery_filenames, _, query_filenames, _, _ = dataset.read_test()

        network_factory = net.create_network_factory(
            is_training=False, num_classes=bdd.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        gallery_features = train_app.encode(
            net.preprocess, network_factory, args.restore_path,
            gallery_filenames, image_shape=bdd.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_test.mat"),
            {"features": gallery_features})

        network_factory = net.create_network_factory(
            is_training=False, num_classes=bdd.MAX_LABEL + 1,
            add_logits=False, reuse=True)
        query_features = train_app.encode(
            net.preprocess, network_factory, args.restore_path,
            query_filenames, image_shape=bdd.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_query.mat"),
            {"features": query_features})
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=bdd.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=bdd.IMAGE_SHAPE,
            output_filename="./bdd.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=bdd.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=bdd.IMAGE_SHAPE,
            output_filename="/data1/haofeng/out/cml/cml-bdd-tracking-train/bdd-tracking-val.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()
