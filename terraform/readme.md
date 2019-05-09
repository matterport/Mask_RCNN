# Azure Setup Instructions
Portion of the following README and terraform scripts come from Andreas Offenhaeuser's excellent [Automated dev workflow for using Data Science VM on Azure](https://medium.com/@an0xff/automated-dev-workflow-for-using-data-science-vm-on-azure-13c1a5b56f91). Before following the instructions below to provision an Azure cluster, you'll need an Azure account and [Azure CLI tools](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) installed on your local machine. Then run

```sh
# have your account username and password handy
az login
```

Next, login to the portal and create a permanent storage account under it's own resource group, meaning a seperate resource group from what you choose for the resource group that is called by terraform (Otherwise you could end up deleting your storage with `terraform destroy`). Edit your azure_configs_template.yaml to populate all the variables necessary to set up a Machine Learning Workspace. The Machine Learning Workspace will use whatever existing resource group is specified for the `perm_resource_group` key in azure_configs.yaml. Rename the template and place azure_configs.yaml in your home directory. Then, from the repo root, run

```sh
python cropmask/azuresetup/create_workspace.py
```
Your workspace should be setup, allowing you to profile models, keep track of model versions, etc.

Next, we will set up our VM infrstructure with terraform, which will also use the azure_configs.yaml as well as other files in the terraform folder to define what sort of vm or cluster we want to spin up. This infrastructure can be destroyed and created whenever, since the resource group that terraform references is seperate from the permanent resource group used by your ML workspace (which tracks your modeling history) and your storage (which contains model inputs, etc.).

As a part of the terraform setup, extra packages will be installed into a conda environment called `cropmask`. It will include `mrcnn`, the python geospatial stack, and [Landsat Surface Reflectance Utils](https://github.com/loicdtx/lsru). You will also need to follow the instructions on the [lsru page](https://github.com/loicdtx/lsru) to provide credentials to order Landsat imagery. These credentials should be placed in the `work/` folder next to the azure config file.


# manage deep learning VM with GPU on Azure ☁️

<!-- TOC depthFrom:2 -->

- [Installation 📦](#installation-)
    - [Prerequisites 🛠](#prerequisites-🛠)
    - [Sign the terms of service ⚖️](#sign-the-terms-of-service-)
    - [Initialize Terraform 🌏](#initialize-terraform-)
- [Configuration ⚙️](#configuration-)
- [Usage 📖](#usage-)
    - [Create or **permanently** delete the Virtual Machine 🆙 🚫](#create-or-permanently-delete-the-virtual-machine--)
    - [Work with the machine 👩‍💻](#work-with-the-machine-‍)
- [Install cuDNN](#install-cudnn)

<!-- /TOC -->

## Installation 📦

First copy the content of this repository into the folder from where you want to manage the VM.

```sh
# download the file and unzip in current directory (under vm-automation)
curl -s https://codeload.github.com/anoff/vm-automation/zip/master | tar -xz --exclude "assets/"
# link the Makefile into the working directory
ln -s vm-automation-master/Makefile Makefile
```

### Prerequisites 🛠

First make sure you have some prerequisites installed:

- [ ] [Terraform](https://www.terraform.io/downloads.html) for infrastructure provisioning
- [ ] [azure cli 2.0](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) as provider for terraform and to interact with the VM
- [ ] [make](http://gnuwin32.sourceforge.net/packages/make.htm) as simple cross-platform scripting solution

### Sign the terms of service ⚖️

The Data Science VM on Azure is offered via the Marketplace and therefore has specific terms of service. Before this offering can be automatically deployed via Terraform you need to accept the license agreement for your subscription. This can be done via **PowerShell**. Easiest way to use powershell is open the Cloudshell on the [Azure Portal](http://portal.azure.com)

<img src="./assets/azure_cloudshell.png" width="80%"><br>
_Open the Cloudshell by clicking the `>_` icon in the top right_

<img src="./assets/azure_powershell.png" width="50%"><br>
_Once open select `PowerShell` as environment_

```powershell
# Use this command to view the current license agreement
Get-AzureRmMarketplaceTerms -Publisher "microsoft-ads" -Product "linux-data-science-vm-ubuntu" -Name "linuxdsvmubuntu"

# If you feel confident to agree to the agreement use the following command to enable the offering for your subscription
Get-AzureRmMarketplaceTerms -Publisher "microsoft-ads" -Product "linux-data-science-vm-ubuntu" -Name "linuxdsvmubuntu" | Set-AzureRmMarketplaceTerms -Accept
```
<img src="./assets/azure_sign_terms.png" width="80%"><br>
_Final output should look like this_

### Initialize Terraform 🌏

Before you can use the Terraform recipe you need to initialize it by running

```sh
terraform init
```

## Configuration ⚙️

To customize the VM deployment you should edit the `config.auto.tfvars` file in this directory. The only mandatory variable you need to provide is `admin_key` which should be a publich SSH key that will be used to connect to the Virtual Machine. See [this explanation](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/) on how to create a SSH Key pair if you do not have one.

If you want to add another user to have ssh access to the vm, share the .vm-id and .vm-ip files with them (and place them in the working directory, the root of this repo), hav ethem generate a key pair by following the instructions above, and have them sen you their `id_rsa.pub` which should be in their .ssh folder. Then,

```sh
#for example. the instance addresses will differ
ssh-copy-id -f -i ~/Downloads/id_rsa.pub ryan@103.44.5.162
```
This person will need to follow the same setup instructions above and will have just as much access as you, since you are using the same Azure account.

For the terraform (.tf) files, just uncomment the variables you want to overwrite. If you want to customize other things feel free to [submit an issue](https://github.com/anoff/vm-automation/issues/new) or look into the way [variables work in terraform](https://www.terraform.io/docs/configuration/variables.html).

## Usage 📖

### Create or **permanently** delete the Virtual Machine 🆙 🚫

```sh
# create a new data scientist VM on a GPU machine
terraform apply

# kill the entire VM
terraform destroy -force
```

When running the terraform command, two new files will be created. `.vm-id` will hold the unique Azure Resource ID of the VM that is used to start/stop it as well as `.vm-ip` that is the public IP address of the VM. The IP is static which means it will not change if you start/stop the machine. 

> Note: When stopping the VM either use `make stop` or `az vm deallocate`. `az vm stop` wil **NOT** deallocate the machine, that means you still have to pay for the compute resources.

### Work with the machine 👩‍💻

```sh
# link the Makefile to your main directory and then run the following commands

make start # to start the VM

make stop # to deallocate it (no more costs for the compute resource)

make ssh # SSH into the machine and port forward 8888 so you can just run 'jupyter notebook' on the VM and open it on your local machine. You may need to first manually connect once from the graphical console before this command will work.

make syncup # copy your local directory to the VM

make syncdown # copy any changes you made on the remote system over to your local directory 🚨 WARNING: OVERWRITES LOCAL CHANGES
```

## Install cuDNN

> 🚨 Note: I think the download is unnecessary as the cuDNN directory already exists under `usr/local/cuda-8-cuddn-5` but is not correctly linked.

The Data Science VM might lack the Cuda Deep Neural Net framework. To install it download it from the [nVidia website](https://developer.nvidia.com/rdp/cudnn-download) (needs a free dev account) for your Cuda version (`nvcc --version`) and follow [this blogpost](https://aboustati.github.io/How-to-Setup-a-VM-in-Azure-for-Deep-Learning/) for the installation. You might need cUDNN 5.0.


## How to setup cropmask conda environment for Jupyter if requirements change

On the Azure VM do the following
```
conda activate cropmask
conda install nb_conda ipykernel # required for jupyter
python -m ipykernel install --user --name cropmask # tells jupyter where to find env
```

Now you can run `jupyter lab` or `jupyter notebook` and the environment will be accessible.

## Mounting Azure blob storage
From [these instructions](https://blogs.msdn.microsoft.com/uk_faculty_connection/2019/02/20/blobfuse-is-an-open-source-project-developed-to-provide-a-virtual-filesystem-backed-by-the-azure-blob-storage/)
Do the following

```
sudo nano /opt/blobfuse.cfg
```

enter in the following information

```
accountName #account name here#
accountKey #account key here#
containerName #container name here#
```

then

```
sudo mkdir /az-ml-container
sudo mkdir /mnt/blobfusecache
chown -R <your user account> /az-ml-container
chown -R <your user account> /mnt/blobfusecache/
# mounts the blob container at az-ml-container, for one time only (becomes inactive on deallocation
blobfuse /images --tmp-path=/mnt/blobfusecache -o big_writes -o max_read=131072 -o max_write=131072 -o attr_timeout=240 -o fsname=blobfuse -o entry_timeout=240 -o negative_timeout=120 --config-file=/opt/blobfuse.cfg
```

# use the custom mount scripts after starting and sshing into the vm to connect the vm to storage. Terraform will mount the vms for you one time but if you start and stop the vms you will need to remount. From the Azure portal you can also try to use the connection script provided by clicking "Connect" on the file share, but this has failed in the past due to bugs on the Azure side of things.
