"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from visualization.visualizer import *
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
            save_results = None,
            output_path = None,
            selected_class = None,
            plots_dir = None,
            image_data_path = None)
        defaults.update(model_and_diffusion_defaults())
        defaults.update(classifier_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    def generate_classifications(self):
        test = TestCase()
        dist_util.setup_dist()
        logger.configure()

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
            #assert probability.shape == (self.args.batch_size, 1000)
            np.testing.assert_almost_equal(np.sum(probability, axis = 0).all(), 1.0, decimal=5) 
            return probability
        
        # to verify the selected class indeed has right classification by the classifier
        # images: number_of_samples x 3 x img_size x img_size
        def verify_clean_image_prediction(images):
            logger.log("verifying clean images are classified correctly ...")
            logger.log("clean image shape: ", images.shape)
            t = th.tensor([0] *  images.shape[0], device = "cuda")
            probability = classify(images.to("cuda"), t)
            assert probability.shape == (images.shape[0], 1000)
            predicted_class_labels = list(np.argmax(probability, axis = 1))
            print(predicted_class_labels)
            #assert len(predicted_class_labels) == images.shape[0]
            accuracy = np.sum(predicted_class_labels == int(self.args.selected_class)) / images.shape[0]
            logger.log("prediction accuracy for clean images is: ", accuracy)
        
        all_images, all_probabilities, all_classes = [], [], []
        logger.log("reading imagenet images ...")
        with np.load(self.args.image_data_path) as data:
            shape, classes = data["data"].shape, data["labels"]
            logger.log("reshape clean images ...")
            images = data["data"].reshape(shape[0], 3, self.args.image_size, self.args.image_size)
            del data
            # convert images into -1.0 ~ 1.0 range
            images = (images/127.5) - 1.0
            print("shape of original images is: ", images.shape)
            selected_class_index = np.where(classes == int(self.args.selected_class))[0]
            assert len(selected_class_index) > int(self.args.batch_size)
            print("selected {number} images!".format(number=len(selected_class_index)))   
            # update number of samples
            self.args.num_samples = len(selected_class_index)
            if self.args.save_results == "True":
                out_path_labels = os.path.join(self.args.output_path, "labels.npz")
                logger.log(f"saving labels to {out_path_labels}")
                np.savez(out_path_labels, classes)
        images = th.tensor(images[selected_class_index[:10], :, :, :], dtype = th.float32).to("cuda")
        logger.log("images of specified class obtained  ...")
        
        logger.log("checking classifier performance on clean images ...")
        verify_clean_image_prediction(images)

        logger.log("creating model and diffusion ...")
        model, diffusion = create_model_and_diffusion(**args_to_dict(self.args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(dist_util.load_state_dict(self.args.model_path, map_location="cpu"))

        model.to(dist_util.dev())
        if self.args.use_fp16:
            model.convert_to_fp16()
        model.eval()
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
        batch_index = 0
        while len(all_images) * self.args.batch_size < self.args.num_samples:
            model_kwargs = {}
            #classes = th.randint(low=0, high=NUM_CLASSES, size=(self.args.batch_size,), 
            #device=dist_util.dev())
            # fix one single class 
            classes = th.ones(size = (self.args.batch_size,), dtype = th.int, device = dist_util.dev())
            model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not self.args.use_ddim else diffusion.ddim_sample_loop
            )

            # output images: batch x diffusionstep x 3 x imgsize x imgsize 
            # output probability vectors: batch x diffusionsteps x num_classes
            output_images = sample_fn(
                model_fn, (self.args.batch_size, 3, self.args.image_size, self.args.image_size),
                noise = images[batch_index * self.args.batch_size:(batch_index + 1) * self.args.batch_size, :, :, :], 
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
            batch_index += 1

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
            out_path_images = os.path.join(self.args.output_path, f"{shape_str}_images.npz")
            out_path_probabilities = os.path.join(self.args.output_path, f"{shape_str}_probabilities.npz")
            logger.log(f"saving images to {out_path_images}")
            logger.log(f"saving probabilities to {out_path_probabilities}")
            np.savez(out_path_images, output_images)
            np.savez(out_path_probabilities, output_probabilities)
        dist.barrier()
        logger.log("data generation completed ...")

    def visualize_images_and_probabilities(self):
        logger.log("starting to visualize ...")
        # read from generated images 
        #img_visualizer = image_visualizer(self.args)
        prob_visualizer = probability_visualizer(self.args)
        prob_visualizer.visualize_histogrm()
        visualizer = image_visualizer(self.args)
        # valid image should be height x width x 3
        visualizer.save_image_grid()
        #visualizer.display_one_image(images.transpose(0, 1, 3, 4, 2), 0, 0)
        logger.log("flow is completed ...")


if __name__ == "__main__":
    tree = classification_tree()
    tree.generate_classifications()
    tree.visualize_images_and_probabilities()

