---
layout: post
title: Training AlexNet on VOC2007 in PyTorch
published: true
---

The goal of this post is to introduce data loading and model training through PyTorch. We will use the VOC2007 dataset and Alexnet to train an image detection network.

## Development Setup

We will use Docker for a development environment. Docker prevents conflicts with your local machine configuration while avoiding the issues of GPU access through a VM. This will save you a lot of trouble when switching between different deep learning frameworks, and will prevent conflicts with library versions (Protobuf must be a different version for Pytorch and ROS for example.)

Clone the tutorials repo:

{% highlight javascript %}
cd ~
git clone https://github.com/grudsby/tutorials

{% endhighlight %}

Setup the Docker image:

{% highlight javascript %}
cd ~/tutorials/alexnet-voc2007-pytorch
sudo chmod +x install.sh
sudo ./install.sh

{% endhighlight %}

The setup script will launch into bash for the Docker container. Here are some useful commands:

{% highlight javascript %}
# Close the Docker container
exit
# Launch the Docker container
./launch_container.sh
{% endhighlight %}

The current Docker container is configured for Python 2.7 and Pytorch. For different configurations, replace the contents of 'Dockerfile' with any of the options from https://github.com/ufoym/deepo/tree/master/docker.

After modifying 'Dockerfile', recompile the container:

{% highlight javascript %}
./03_setup_container.sh
{% endhighlight %}

## Cloud Setup

The cloud setup is optional, but is recommended to speed up training time. It is also good to get some exposure to working with AWS. The majority of develpment time should be spent on your local machine, as cloud instances for deep learning can be expensive.

Register for an AWS account: https://aws.amazon.com/

Sign into the AWS console: https://aws.amazon.com/console/

Go to 'Limits' and request a limit increase to 1 for the p2.xlarge instances.

![_config.yml]({{ site.baseurl }}/images/limit_increase.png)

When you have access to a p2.xlarge instance, go to the EC2 Dashboard. 

## Implementing Alexnet

Download the VOC2007 dataset:

{% highlight javascript %}

cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar xf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
cd ../

{% endhighlight %}

Take a look at the VOC2007 dataset structure.
