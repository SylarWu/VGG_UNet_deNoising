import torch
import scipy.io as scio


benchmark = scio.loadmat('./BenchmarkNoisyBlocksSrgb.mat')

benchmark = benchmark["BenchmarkNoisyBlocksSrgb"]
