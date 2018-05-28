---
layout: post
title: Training AlexNet on VOC2007 in PyTorch
published: true
---

The goal of this post is to introduce data loading and model training through PyTorch. We will use the VOC2007 dataset and Alexnet to train an image detection network.

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

## Implementing Alexnet

Download the VOC2007 dataset:

{% highlight bash %}

cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar xf VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
cd ../

{% endhighlight %}

Take a look at the VOC2007 dataset structure.
