---
layout: post
title: Training AlexNet on VOC2007 in PyTorch
published: true
---

The goal of this post is to introduce data loading and model training through PyTorch. We will use the PASCAL 2007 dataset and Alexnet to train an image detection network.

This tutorial assumes that you are running Ubuntu 16.04

Go through the Development Setup and read the Alexnet paper while you're waiting for the setup steps:

<https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>

## Development Setup

We will use Docker for a development environment. Docker prevents conflicts with your local machine configuration while avoiding the issues of GPU access through a VM. This will save you a lot of trouble when switching between different deep learning frameworks, and will prevent conflicts with library versions (Protobuf must be a different version for Pytorch and ROS for example.)

Clone the tutorials repo:

{% highlight bash %}
cd ~
git clone https://github.com/grudsby/tutorials
{% endhighlight %}

Setup the Docker image:

{% highlight bash %}
cd ~/tutorials/alexnet-voc2007-pytorch
sudo chmod +x install.sh
sudo ./install.sh
{% endhighlight %}

The Docker install will take a bit of time, so it would be a good idea to leave it running in a terminal and move onto the cloud setup section.

When completed, the setup script will launch into bash for the Docker container. Make sure to navigate to the tutorial folder when first launching the Container.

{% highlight bash %}
cd ~/storage/tutorials/alexnet-voc2007-pytorch
{% endhighlight %}

Here are some useful commands:

{% highlight bash %}
# Close the Docker container 
exit
# Launch the Docker container 
./launch_container.sh
{% endhighlight %}

The current Docker container is configured for Python 2.7 and Pytorch. For different configurations, replace the contents of 'Dockerfile' with any of the options from https://github.com/ufoym/deepo/tree/master/docker.

After modifying 'Dockerfile', recompile the container:

{% highlight bash %}
./03_setup_container.sh
{% endhighlight %}

Note that docker containers are typically "stateless". This means you are launching a docker container as an execution environment, not for data storage. This means you will either write data to a folder structure shared on your local drive, or you will write data to a sql server (in the case of web applications running docker.)

The following code docker option in launch_container.sh takes care of mapping the tutorial folder on your local machine to the docker container. 

{% highlight bash %}
-v $(pwd)/../:/root/storage/tutorials
{% endhighlight %}

My typical develpment method is to launch the container and use it to run scripts, but edit the code directly on my machine rather than editing through the docker container. Updates to files are propagated instantly.

## Cloud Setup

The cloud setup is optional, but is recommended to speed up training time. It is also good to get some exposure to working with AWS. The majority of develpment time should be spent on your local machine, as cloud instances for deep learning can be expensive.

Register for an AWS account: https://aws.amazon.com/

Sign into the AWS console: https://aws.amazon.com/console/

Go to 'Limits' and request a limit increase to 1 for the p2.xlarge instances.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/limit_increase.png)

When you have access to a p2.xlarge instance, proceed with the following setup steps in the AWS Console. 

Access to the cloud compute will be through ssh, so we will need to generate an encryption key. Click on 'Key Pairs' under the 'Network & Security' section on the left of the AWS Console. 

Create a new key pair named 'aws'.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/create_key.png)

Key creation will automatically download a file named 'aws.pem. Move this file to the /tutorials/alexnet-voc2007-pytorch/ folder.

{% highlight bash %}
mv ~/Downloads/aws.pem ~/tutorials/alexnet-voc2007-pytorch/aws.pem
chmod 400 ~/tutorials/alexnet-voc2007-pytorch/aws.pem 
{% endhighlight %}

AWS instances do not have reliable data storage, so an exernal volume must be created for your data and code. Click on 'Volumes' under the 'Elastic Block Store' section on the left of the AWS Console.

Click on 'Create Volume'. Create a General Purpose SSD with 50 GiB of storage. Choose an availability zone, but note that the zone must be consistent with your AWS instance. 

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/create_volume.png)

Now we can launch an AWS instance! Click on 'Instances' under the 'Instances' section on the left of the AWS Console. 

Choose the "Deep learning AMI (Ubuntu)" image. 

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/deep_learning_ami.png)

For Instance Type, choose the "p2.xlarge" platform. Click Next: Configure Instance Details.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/p2_xlarge.png)

For 'Subnet', make sure to select the same availability zone as your SSD volume. Click 'Next: Add Storage'.
 
![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/subnet_zone.png)

Note that the storage added here is for the instance itself. The volume we previously created will need to linked to the instance later. Click 'Review and Launch'.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/add_storage.png)

Review the summary and click 'Launch'. Choose the key pair we generated previously.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/launch_assign_key.png)

Click on 'View Instances' from the next screen, or just click on 'Instances' in the 'Instances' section in the AWS Console. Copy the 'Instance ID' of your p2.xlarge instance, we will need to use it to attach our storage volume to this instance. Click on 'Volumes' under 'Elastic Block Store' on the left of the AWS Console and right click on our 50 GiB volume. Click 'Attach Volume' and past the instance ID into the instance field. Name the device '/dev/sda2'. Note that volumes can only be attached to a single instance.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/attach_volume.png)


Go back to 'Instances' and click 'Connect' and follow the example command to log into your AWS instance. Note that you may need to change 'root' to a different user name depending on how AWS is configuring instances. If this change is needed you will get a message when logging in as root.

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/ssh_login.png)

Format and mount the volume using the following commands. In this example, the device name was /dev/xvdb, this may be different depending on the return of lsblk:

{% highlight bash %}
lsblk                        # Show the available devices
sudo file -s /dev/xvdb       # Show whether the device is formatted
                             # This will return 'data' if the device is not formatted
sudo mkfs -t ext4 /dev/xvdb  # Format the device with an ext4 file system
{% endhighlight %}

Enter 'y' if prompted to overwrite existing formatting. Proceed with device mounting below:

{% highlight bash %}
sudo mkdir storage            # The mounting point for our volume
sudo mount /dev/xvdb storage
sudo chmod 777 -R storage
cd storage 
git clone https://github.com/grudsby/tutorials
cd ~/storage/tutorials/alexnet-voc2007-pytorch
{% endhighlight %}

Enter the Pytorch environment and install a few extra dependencies:

{% highlight bash %}
source activate pytorch_p27
pip install cython visdom Pillow Tensorboard
{% endhighlight %}

You will need to source the environment again when you restart your AWS instance.

Make sure to stop your instance when you are finished using it. It can cost a dollar or two every hour running!

## Implementing Alexnet

From this point on, it is assumed that you can follow along on, Docker, AWS, or both. I would recommend that you develop and test your code in Docker and train in AWS. 

### Download the PASCAL 2007 dataset:

{% highlight bash %}
mkdir code, data
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar xf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
cd ../
{% endhighlight %}

Create a new file called 'alexnet.py' with the following code:

{% highlight python %}
import sys
import argparse
import pdb


def load_pascal(data_directory, split='train'): # train/test/val
    return (images, labels, weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Alexnet in Pytorch.')
    parser.add_argument(
        'data_directory', type=str, default='data/VOC2007',
        help='Path to PASCAL data set')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    pdb.set_trace()


if __name__ == "__main__":
    main()
{% endhighlight %}

To run the code:

{% highlight bash %}
python code/alexnet.py ~/storage/tutorials/alexnet-voc2007-pytorch/data 5 20
{% endhighlight %}

This is a very basic structure, but will be instructive for implementing Alexnet from scratch. We will go through the following steps to implement the model:

1) Write the data loader

2) Implement the model

3) Setup the optimizer and batch randomization

4) Define the loss function

5) Create evaluation metrics

6) Visualize model inputs, outputs, and intermediate layers

7) Train and test the model

8) Load a pre-trained network

### Data Loading:

The data is divided into three sets: train, test, and validation. The loader will need to get a list of images for each set, load the images, read in ground truth labels, and parse label weights.

Take a look at the VOC2007 data structure and find where images are and where to extract label data. The label for each class should be a vector as large as the number of classes with ones or zeros that indicate which class is present in each image. The weight for each class indicates uncertainty. If the label is uncertain, then the weight should be zero.

Implement the data loader.

<details>
    <summary>Hint #1</summary>
    <br/>
    Images are located in data/VOCdevkit/VOC2007/JPEGImages/ <br/>
    Image names are located in data/VOCdevkit/VOC2007/ImageSets/Main/"test/train/val".txt <br/>
    Ground truth labels are defined in data/VOCdevkit/VOC2007/ImageSets/Main/"class"_"test/train/val".txt <br/>
    <br/>
    Labels are defined as follows: <br/>
    1: The image contains the class <br/>
    0: The class definition is uncertain <br/>
    -1: The image does not contain the class  <br/>
    <br/>
</details>

<details>
    <summary>Hint #2</summary>

Here are the classes in PASCAL 2007:

{% highlight python %}
CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
{% endhighlight %}
</details>

<details>
    <summary>Hint #3</summary>
Here is one way to load the test/train/val image names:

{% highlight python %}
imageFile = open(data_directory + "/VOCdevkit/VOC2007/ImageSets/Main/" + split + ".txt","r").read()
splitImageFile = imageFile.split('\n')[:-1]
numImages = len(splitImageFile)
{% endhighlight %}
</details>

<details>
    <summary>Hint #4</summary>
Alexnet takes in RGB images with a resolution of 224x224. Our initial images, labels, and weights should be formatted as numpy arrays with the right number of classes.
{% highlight python %}
import numpy as np
...
images = np.empty([numImages,224,224,3],dtype=np.float32)
labels = np.empty([numImages,20],dtype=np.int32)
weights = np.empty([numImages,20],dtype=np.int32)
{% endhighlight %}
</details>


<details>
    <summary>Hint #5</summary>
Use Pillow to load images.
{% highlight python %}
from PIL import Image
...
for i in range(num_images):
    image_name  = split_image_file[i]
    input_img   = Image.open(data_directory + "/VOCdevkit/VOC2007/JPEGImages/" + image_name + ".jpg")
    scaled_img  = np.asarray(input_img.resize([224,224]))
    images[i,:,:,:] = scaled_img[np.newaxis,:,:,:]
{% endhighlight %}
</details>
 
<details>
    <summary>Code after writing data loader</summary>

{% highlight python %}
import sys
import argparse
import pdb
from PIL import Image
import numpy as np
CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

NUM_CLASSES = 20


def load_pascal(data_directory, split='train'): # train/test/val
    image_file = open(data_directory + "/VOCdevkit/VOC2007/ImageSets/Main/" + split + ".txt","r").read()
    split_image_file = image_file.split('\n')[:-1]
    num_images = len(split_image_file)

    images = np.empty([num_images,224,224,3],dtype=np.float32)
    labels = np.empty([num_images,NUM_CLASSES],dtype=np.int32)
    weights = np.empty([num_images,NUM_CLASSES],dtype=np.int32)

    for i in range(num_images):
        image_name  = split_image_file[i]
        input_img   = Image.open(data_directory + "/VOCdevkit/VOC2007/JPEGImages/" + image_name + ".jpg")
        scaled_img  = np.asarray(input_img.resize([224,224]))
        images[i,:,:,:] = scaled_img[np.newaxis,:,:,:]
        image_label     = np.empty([0])
        image_weight    = np.empty([0])
        for class_name in CLASS_NAMES:
            split_file = open(data_directory + "/VOCdevkit/VOC2007/ImageSets/Main/" + class_name + "_" + split + ".txt","r").read()
            img_pos = split_file.find(image_name)+len(image_name)+1
            img = split_file[img_pos:img_pos+2]
            if img == " 1":
                image_label  = np.append(image_label,[1],axis=0)
                image_weight = np.append(image_weight,[1],axis=0)
            else:
                image_label  = np.append(image_label,[0],axis=0)
                if img == " 0":
                    image_weight = np.append(image_weight,[0],axis=0)
                else:
                    image_weight = np.append(image_weight,[1],axis=0)
        labels[i,:] = image_label
        weights[i,:] = image_weight
    return (images, labels, weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Alexnet in Pytorch.')
    parser.add_argument(
        'data_directory', type=str, default='data/VOC2007',
        help='Path to PASCAL data set')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    (train_images, train_labels, train_weights) = load_pascal(args.data_directory, split='train')
    (test_images, test_labels, test_weights) = load_pascal(args.data_directory, split='test')
    (val_images, val_labels, val_weights) = load_pascal(args.data_directory, split='val')



if __name__ == "__main__":
    main()

{% endhighlight %}
</details>

Run the code and make sure you don't get any errors. If you have any issues, use pdb.set_trace() before the line throwing an error and use the interactive debugger to test commands and fix the issue.

{% highlight bash %}
python code/alexnet.py ~/storage/tutorials/alexnet-voc2007-pytorch/data
{% endhighlight %}

The data can take up to 70 seconds to parse, which is awful for debugging. We can use numpy's save function to save out the parsed data so that loading goes quicker. No code hints are provided, but the speed improvements should be a great incentive to finish out the implementation. Loading will take less than a few seconds when implemented. The saved data will take 6.0 GB of space on your hard drive. 

### Model Implementation:

It's time to talk about tensors. Tensors are multidimensional arrays. A vector is a first order tensor and a matrix is a second order tensor. In Pytorch, it is simple to convert from Numpy vectors to Pytorch tensors and back. Another important distinction is that Pytorch tensors can be stored on CPU Ram or in the GPU. Training a network on the GPU while pulling data out of CPU Ram would be too slow, so all current training data should be held in GPU memory. This is one of the reasons why we train in batches when computing gradients for back propogation (more on this later.) Here are example commands for conversions: 

{% highlight python %}
import torch
import numpy as np
numpy_vec = np.ones([50,20,20,3], dtype=np.float32)
tensor_cpu = torch.from_numpy(numpy_vec)
tensor_gpu = tensor_cpu.cuda()
tensor_cpu_2 = tensor_gpu.cpu()
numpy_vec_2 = tensor_cpu_2.numpy()
{% endhighlight %}

Implementing the Alexnet model will give Pytorch the information needed to calculate the output of the network, backpropogate error through the model, and track layer weights for gradient descent. The model does not contain a way to compute an error, this will be done through a criterion, and a Pytorch optimizer will handle gradient descent. 

The easiest way to implement a pytorch model is to build a torch.nn.Sequential model. This will take different Pytorch layers as an input and stack the layers, with the output from the first going into the second, etc. Here is a simple model definition as a starting point:

{% highlight python %}
model = torch.nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=7, stride=3),
    nn.Dropout(p=0.2),
    nn.Linear(9600, 4096)
)
{% endhighlight %}

Let's see how this network would behave if we feed images through it. Starting with a tensor of 10 images, size 256x256 with 3 color channels, we would have a tensor input of shape [10,256,256,3]. 

The model will first perform 16 convolutions on each image. The first parameter in nn.Conv2d is the channel size of the input images. Another way to think of this parameter is that it is the size of the last dimension of the input tensor. Kernel size is the size of the convolution, stride is how many pixels are skipped on each convolution, and padding is the number of pixels to add on all borders of the image. The image size of the output can be found through the following formula:

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/output_formula.png)

O is the output height/width <br/> 
W is the input height/width <br/> 
K is the kernel size  <br/> 
P is the padding <br/> 
S is the stride  <br/> 

Another useful formula would be to set the padding so that the input and output heights/widths are the same. To do this, set the stride to 1 and set the padding as follows:

![_config.yml]({{ site.baseurl }}/images/alexnet-voc2007-pytorch/zero_padding.png)

Given these two formulas, the output of the first convolution would be a tensor of shape:
 
<details>
    <summary>Click to reveal answer</summary>
{% highlight python %}
 [10,64,64,16] 
{% endhighlight %}
</details>

The second layer in our example is a Rectified Linear Unit, which introduces non-linearity into the model. Inputs and outputs have the same dimensionality

The third layer performs Max Pooling, which takes a kernel and extracts the maximum value from the kernel shape and provides that as the output. Max pooling is similar to convolution, except the number of channels does not change. 

The output of the third convolution would be a tensor of shape:
 
<details>
    <summary>Click to reveal answer</summary>
{% highlight python %}
 [10,20,20,16] 
{% endhighlight %}
</details>

One of the interesting effects of Max Pooling is that it only allows gradients to propogate back through the maximum element from the kernel. This reduces the effect of vanishing gradients, concentrating gradient propagation in the most important elements rather than spreading them out across many elements. This also introduces a non-linearity into the model which helps to compensate for the linear nature of convolutions and fully connected layers.

The fourth layer performs Dropout, which randomly turns weights off in the network. The parameter p is the probability that an element will be zeroed. Dropout allows the network to become more robust by forcing different areas within the network to learn redundant tasks. For example, a network trained with dropout would potentially have two hand detection neurons instead of a single activation for hands.

Finally, the fifth layer is a fully connected layer. The input to a fully connect layer in Pytorch is a concatenation of the image width, height, and channel depth. These dimensions can be multiplied together to produce the correct number of inputs to the network, and the output value is equal to the number of neurons in the network. In the example model I intentionally entered an incorrect value for the input layer. The true number of inputs would be: 
 
<details>
    <summary>Click to reveal answer</summary>
{% highlight python %}
16*20*20 = 6400
{% endhighlight %}
</details>

Note that in each of these operations, the dimension of the tensor corresponding to the number of images is unaffected, so each operation is performed on a per-image basis, preservering the number of images in the output.

Build the model for Alexnet using the elements contained in the sample model.

<details>
    <summary>Hint: Full Alexnet Model Definition</summary>
{% highlight python %}
    model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, NUM_CLASSES)
    ).cuda()
{% endhighlight %}
</details>

### Optimizer/Batch Randomization

Stochastic gradient descent with momentum is a typical optimizer to use for debugging a network or checking baseline performance.  

{% highlight python %}
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
{% endhighlight %}

A better optimizer is the Adam optimizer (<https://arxiv.org/abs/1412.6980>.) Implement the Pytorch version of Adam from torch.optim.

<details>
    <summary>Hint: Adam Optimizer</summary>
{% highlight python %}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.005, amsgrad=False)
{% endhighlight %}
</details>

Batch randomization groups your training set into batches for processing on the GPU. With very large models you will come close to using up all of your GPU Ram, so it is important to make batch sizes appropriate for available Ram.

For each epoch you will run as many batches as needed to evaluate the entire training set of images.



### Loss Function:

A loss function gives your network a measure of distance from the ground truth label. For 'One-hot' image classification, only one label is valid for each image. In this case it is best to use a softmax cross entropy loss. The softmax function combines each of the class outputs from the model and votes for the strongest output as the most likely class, zeroing the others. The cross entropy component of this loss function measures the distance between the output distribution of classes and the ground truth distribution.

For PASCAL 2007 we may have multiple labels per image, so we will use a Binary Cross Entropy (BCE) loss function. Binary cross entropy uses a sigmoid at each class output and allows the outputs to determine their label likelihood independently. The output is a probability measuring how likely each label is based on the input image. This allows multiple classes to indicate a high likelihood at the output, but is a weaker assumption bias, making the network more difficult to train. The cross entropy component operates separately on each binary distribution output, and the resulting cross entropies can be summed or averaged. Averaging will tend to normalize data sets that have many images of one type of class and few of another.

Pytorch has two options for BCE loss. In our model we have not included a sigmoid layer at the output, so the model outputs raw 'logits'. These logits, ranging from -infinity to infinity, must be converted to probabilities through a sigmoid function, but this is better done within the loss criterion for numerical stability. Implement the BCEWithLogitsLoss function: 

<details>
    <summary>Hint: Loss Function</summary>
{% highlight python %}
criterion = torch.nn.BCEWithLogitsLoss(weight=weights_train,size_average=True)
{% endhighlight %}
<br/>
Note that we are passing in the full weights vector. When we implement batch randomization we will input only the vector of weights for a given batch.
</details>

### Evaluation Metrics:

### Model Visualization:

### Test/Train 

### Pretrained Model 
