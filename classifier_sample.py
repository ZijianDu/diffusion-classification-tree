"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from visualization.visualizer import image_visualizer
from PIL import Image
import pickle
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import unittest
from unittest import TestCase

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

class classification_tree:
    def __init__(self):
        self.image_data_path = 'imgnet_64x64_datasets/train_data_batch_1.npz'
        self.output_path = "results/"
        self.args = self.create_argparser().parse_args()

    def create_argparser(self):
        defaults = dict(
            clip_denoised=True,
            num_samples=10000,
            batch_size=16,
            use_ddim=True,
            model_path="",
            classifier_path="",
            classifier_scale=1.0,
            num_diffusion_samples = None,
            save_results = None)
        defaults.update(model_and_diffusion_defaults())
        defaults.update(classifier_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def generate_classifications(self):
        test = TestCase()
        dist_util.setup_dist()
        logger.configure()
        # sample generated should be greater or equal to number of 
        # diffusion samples to generate a square image grid
        assert int(self.args.num_samples) >= int(self.args.num_diffusion_samples)

        all_images, all_probabilities, all_classes = [], [], []
        logger.log("reading imagenet images ...")

        with np.load(self.image_data_path) as data:
            shape = data["data"].shape
            logger.log("reshape clean images ...")
            images = data["data"].reshape(shape[0], 3, self.args.image_size, self.args.image_size)
            # convert images into -1.0 ~ 1.0 range
            images = (images/127.5) - 1.0
            classes = data["labels"]
            if self.args.save_results == "True":
                out_path_labels = os.path.join(self.output_path, "labels.npz")
                logger.log(f"saving labels to {out_path_labels}")
                np.savez(out_path_labels, classes)
        images = th.tensor(images, dtype = th.float32).to("cuda")
        logger.log("creating model and diffusion ...")
    
        model, diffusion = create_model_and_diffusion(**args_to_dict(self.args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(dist_util.load_state_dict(self.args.model_path, map_location="cpu"))

        model.to(dist_util.dev())
        if self.args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("loading classifier ...")
        classifier = create_classifier(**args_to_dict(self.args, classifier_defaults().keys()))
        classifier.load_state_dict(dist_util.load_state_dict(self.args.classifier_path, map_location="cpu"))
        classifier.to(dist_util.dev())
        if self.args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        # classifier to output probability class for each input image/t combination
        # x: batchsize x 3 x imgsize x imgsize
        # probability: batchsize x 1000
        def classify(x, t):
            x_in = x.detach()
            logits = classifier(x_in, t)
            probability = F.softmax(logits, dim = -1).detach().cpu().numpy()
            assert len(probability.shape) == 2
            assert probability.shape == (self.args.batch_size, 1000)
            for i in range(self.args.batch_size):
                np.testing.assert_almost_equal(sum(probability[i, :]), 1.0, decimal=5) 
            return probability

        def cond_fn(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                # x_in: batch_size x 3 x img_size x img_size
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * self.args.classifier_scale

        def model_fn(x, t, y=None):
            assert y is not None
            return model(x, t, y if self.args.class_cond else None)
        
        def get_one_batch_probability(output_images, batch_size, num_diffusion_samples):
            # batchsize x diffusion steps x 1000 classes
            one_batch_probability_vector = np.zeros(shape = (batch_size, num_diffusion_samples, 1000))
            for time_step in range(int(num_diffusion_samples)):
                t = th.tensor([time_step] * int(batch_size), device = "cuda")
                probability = classify(output_images[:, time_step, :, :, :].to("cuda"), t)
                one_batch_probability_vector[:, time_step, :] = probability
            return one_batch_probability_vector

        logger.log("sampling ...")
        while len(all_images) * self.args.batch_size < self.args.num_samples:
            model_kwargs = {}
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(self.args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not self.args.use_ddim else diffusion.ddim_sample_loop
            )

            # output images: batch x diffusionstep x 3 x imgsize x imgsize 
            # output probability vectors: batch x diffusionsteps x num_classes
            output_images = sample_fn(
                model_fn, (self.args.batch_size, 3, self.args.image_size, self.args.image_size),
                noise = images[:self.args.batch_size, :, :, :], 
                clip_denoised=self.args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                args = self.args
            )

            output_images = output_images.contiguous()
            all_probabilities.append(get_one_batch_probability(output_images, 
                                                            int(self.args.batch_size), 
                                                            int(self.args.num_diffusion_samples)))
        
            # renormalize images to be viewable
            output_images = ((output_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
            all_images.append(output_images)
            del output_images

            logger.log(f"created {len(all_images) * self.args.batch_size} samples")
            logger.log(f"created {len(all_probabilities) * self.args.batch_size} probabilities")

        # final image arr shape: num_samples x diffusionsteps x 3 x img_size x img_size
        output_images = np.concatenate(all_images, axis=0)
        output_images = output_images[: self.args.num_samples]
        print("final output image shape: ", output_images.shape)

        # final probability vector shape: num_samples x diffusionsteps x 1000
        output_probabilities = np.concatenate(all_probabilities, axis=0)
        output_probabilities = output_probabilities[: self.args.num_samples]
        print("final output probability shape: ", output_probabilities.shape)

        if self.args.save_results == "True":
            shape_str = "x".join([str(x) for x in output_images.shape])
            out_path_images = os.path.join(self.output_path, f"{shape_str}_images.npz")
            out_path_probabilities = os.path.join(self.output_path, f"{shape_str}_probabilities.npz")
            logger.log(f"saving images to {out_path_images}")
            logger.log(f"saving probabilities to {out_path_probabilities}")
            np.savez(out_path_images, output_images)
            np.savez(out_path_probabilities, output_probabilities)
        dist.barrier()
        logger.log("data generation completed ...")

    def visualize_images_and_probabilities(self):
        logger.log("starting to visualize ...")
        # read from generated images 
        shape_str = "x".join([str(x) for x in (self.args.num_samples, self.args.num_diffusion_samples,
                                               3, self.args.image_size, self.args.image_size)])
        images = np.load(os.path.join(self.output_path, f"{shape_str}_images.npz"))["arr_0"][:int(self.args.num_diffusion_samples), :, :, :, :]
        visualizer = image_visualizer(int(self.args.num_diffusion_samples), int(self.args.num_diffusion_samples))
        #visualizer = image_visualizer(5, 5)
        # valid image should be height x width x 3
        visualizer.save_image_grid(images.transpose(0, 1, 3, 4, 2))
        logger.log("flow is completed ...")


if __name__ == "__main__":
    tree = classification_tree()
    tree.generate_classifications()
    tree.visualize_images_and_probabilities()

