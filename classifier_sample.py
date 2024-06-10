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
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
os.environ["WORLD_SIZE"] = "1"
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import unittest
from unittest import TestCase
import torchvision
from torchvision.transforms import v2
from torchvision import transforms
from PIL import Image
from guided_diffusion.image_datasets import *
import pickle

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
import glob

class classification_tree:
    def __init__(self):
        self.args = self.create_argparser().parse_args()
        self.folder_name_set = set()
        # dictionary of clean images, key is class
        self.clean_images = defaultdict()
        self.all_keys = None
        # dictionary of clean images with 1000 diffusion steps
        # shape: num_images x diffusion steps x 3 x imgsize x imgsize
        self.all_diffusion_steps_images = defaultdict()
        # dictionary of predicted probabilities
        # shape: num_images x diffusion steps x num_classes
        self.all_diffusion_steps_prediction = defaultdict()

    # modify once there is new args to parse
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
            image_data_path = None, 
            random_crop_per_image = None,
            random_h_flip = None,
            random_v_flip = None)
        defaults.update(model_and_diffusion_defaults())
        defaults.update(classifier_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser
    
    ## function to generate and save images of desired size with augmentation
    def create_augmented_images(self):
        image_names = os.listdir(self.args.image_data_path)
        num_images = len(image_names)
        cropper = v2.RandomCrop(size=(int(self.args.image_size), int(self.args.image_size)))
        hflipper = v2.RandomHorizontalFlip(p = float(self.args.random_h_flip))
        vflipper = v2.RandomVerticalFlip(p = float(self.args.random_v_flip))
        all_augmented_images = []
        for img_name in image_names:
            curr_image = Image.open(self.args.image_data_path + img_name)
            for crop_idx in range(int(self.args.random_crop_per_image)):
                all_augmented_images.append(cropper(curr_image))
        # do the h and v flips
        for img_index in range(len(all_augmented_images)):
            all_augmented_images[img_index] = vflipper(hflipper(all_augmented_images[img_index]))
        
        all_augmented_images[41].save("cropped.jpg")
        # reshape images to be (CropNumxImgNum) x 3 x image_size x image_size
        return np.array(all_augmented_images).reshape((int(self.args.random_crop_per_image)*num_images,
                                                       3, int(self.args.image_size), int(self.args.image_size)))
    def generate_clean_images(self):
        image_names = sorted(os.listdir(self.args.image_data_path))
        print("number of images: " + str(len(image_names)))
        for name in image_names:
            image = Image.open(self.args.image_data_path + name)
            resized_image = center_crop_arr(image, int(self.args.image_size))
            if name[:9] not in self.clean_images:
                self.clean_images[name[:9]] = []
            self.clean_images[name[:9]].append(resized_image)
        print("number of classes: " + str(len(list(self.clean_images.keys()))))
        print("saving clean images")
        with open(self.args.output_path + 'all_clean_images.pickle', 'wb') as handle:
            pickle.dump(self.clean_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # read saved clean images and reshape values into Num_Images x 3 x 64 x 64 then change to -1.0~1.0 range
    def read_clean_images(self):
        with open(self.args.output_path + 'all_clean_images.pickle', 'rb') as handle:
            self.clean_images = pickle.load(handle)
        self.all_keys = list(self.clean_images.keys())
        for i, key in enumerate(self.all_keys):
            curr_images = self.clean_images[key]
            curr_num_images = len(curr_images)
            self.clean_images[key] = np.array(curr_images).reshape((curr_num_images, 
                                                                    int(self.args.image_size), 
                                                                    int(self.args.image_size), 3))
            # sanity check the images are correct
            Image.fromarray(self.clean_images[key][0, :, :, :]).save(self.args.output_path + f"{i}th class sample.png".format(i))
            self.clean_images[key] = (self.clean_images[key].transpose(0, 3, 1, 2) / 127.5) - 1.0
            # discard last couple images to make number of images dividable by batch number
            kept_num_images = curr_num_images - curr_num_images % int(self.args.batch_size)
            self.clean_images[key] = self.clean_images[key][:kept_num_images, :, :, :]
            assert self.clean_images[key].shape == (kept_num_images, 3, int(self.args.image_size), int(self.args.image_size))
        print("finish reading clean images")
        
    def generate_classifications(self):

        # classifier to output probability class for each input image/t combination
        # x: batchsize x 3 x imgsize x imgsize probability: batchsize x 1000
        def classify(x, t):
            x_in = x.detach()
            logits = classifier(x_in, t)
            probability = F.softmax(logits, dim = -1).detach().cpu().numpy()
            assert len(probability.shape) == 2
            assert probability.shape[1] == 1000
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


        # load classifier
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
        
        logger.log("creating model and diffusion ...")
        model, diffusion = create_model_and_diffusion(**args_to_dict(self.args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(dist_util.load_state_dict(self.args.model_path, map_location="cpu"))

        model.to(dist_util.dev())
        if self.args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        
        logger.log("reading saved clean images ...")
        #self.generate_clean_images()
        self.read_clean_images()

        logger.log("checking classifier performance on clean images ...")
        sample_class_images = self.clean_images[self.all_keys[0]][:10, :, :, :]
        verify_clean_image_prediction(th.tensor(sample_class_images, dtype = th.float32, device="cuda"))

        logger.log("generating reverse DDIM samples ...")
        # two iterations: outter is different class, inner is batch for this class
        for key_idx, key in enumerate(self.all_keys):
            curr_class_num_images = self.clean_images[key].shape[0]
            batch_index = 0
            one_class_images, one_class_probabilities = [], []
            print("processing one class with key: ", key, " images shape: ", self.clean_images[key].shape, "\n")
            images = self.clean_images[key]
            while len(one_class_images) * self.args.batch_size < curr_class_num_images:
                logger.log("\n" + f"processing batch {batch_index}".format(batch_index))
                model_kwargs = {}
                classes = th.randint(low=0, high=NUM_CLASSES, size=(self.args.batch_size,), 
                device=dist_util.dev())
                model_kwargs["y"] = classes
                sample_fn = (
                    diffusion.p_sample_loop if not self.args.use_ddim else diffusion.ddim_sample_loop
                )

                # output images: batch x diffusionstep x 3 x imgsize x imgsize 
                # output probability vectors: batch x diffusionsteps x num_classes
                output_images = sample_fn(
                    model_fn, (self.args.batch_size, 3, self.args.image_size, self.args.image_size),
                    noise = th.tensor(images[batch_index*self.args.batch_size:(batch_index + 1)*self.args.batch_size, :, :, :], dtype = th.float32).to("cuda"), 
                    clip_denoised=self.args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                    args = self.args
                    )
                output_images = output_images.contiguous()
                one_class_probabilities.append(get_one_batch_probability(output_images, 
                                                                int(self.args.batch_size), 
                                                                int(self.args.num_diffusion_samples)))
            
                # renormalize images to be viewable
                output_images = ((output_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
                one_class_images.append(output_images)
                del output_images
                logger.log(f"processed batch {batch_index}".format(batch_index))
                logger.log(f"created {len(one_class_images) * self.args.batch_size} samples")
                logger.log(f"created {len(one_class_probabilities) * self.args.batch_size} probabilities")
                batch_index += 1
                break

            # save results for one class
            # final image arr shape: num_samples x diffusionsteps x 3 x img_size x img_size
            output_images = np.concatenate(one_class_images, axis=0)
            print("current class output image shape: ", output_images.shape)
            self.all_diffusion_steps_images[key] = output_images

            # final probability vector shape: num_samples x diffusionsteps x 1000
            output_probabilities = np.concatenate(one_class_probabilities, axis=0)
            print("current class output probability shape: ", output_probabilities.shape)
            self.all_diffusion_steps_prediction[key] = output_probabilities
            break

        if self.args.save_results == "True":
            shape_str = "x".join([str(x) for x in output_images.shape])
            logger.log("saving images ...")
            logger.log("saving probabilities ...")
            with open(self.args.output_path + 'all_diffusion_steps_images.pickle', 'wb') as handle:
                pickle.dump(self.all_diffusion_steps_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.args.output_path + 'all_diffusion_steps_probabilities.pickle', 'wb') as handle:
                pickle.dump(self.all_diffusion_steps_prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dist.barrier()
        logger.log("data generated and probabilities calculated ...")

    def visualize_images_and_probabilities(self):
        logger.log("starting to visualize ...")
        # read from generated images 
        #img_visualizer = image_visualizer(self.args)
        prob_visualizer = probability_visualizer(self.args)
        prob_visualizer.visualize_histogrm()
        visualizer = image_visualizer(self.args)
        # valid image should be height x width x 3
        #visualizer.save_image_grid()
        #visualizer.display_one_image(images.transpose(0, 1, 3, 4, 2), 0, 0)
        logger.log("flow is completed ...")


if __name__ == "__main__":
    tree = classification_tree()
    tree.generate_classifications()
    tree.visualize_images_and_probabilities()

