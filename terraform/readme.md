# Azure Setup Instructions
The following README and terraform scripts come from Andreas Offenhaeuser's [Automated dev workflow for using Data Science VM on Azure](https://medium.com/@an0xff/automated-dev-workflow-for-using-data-science-vm-on-azure-13c1a5b56f91). Before following the instructions below to provision an Azure cluster, you'll need an Azure account and [Azure CLI tools](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest) installed on your local machine. Then run

```sh
# have your account username and password handy
az login
```

After following these instructions, you can run the configure.sh script manually once it has been uploaded to the cluster, to install extra packages that do not come with the Data Science VM, like MaskRCNN, the python geospatial stack, and [Landsat Surface Reflectance Utils](https://github.com/loicdtx/lsru). These packages will be installed to the `py36` conda environment on the Data Science VM, so you should use that environment/kernel when testing with Jupyter. You will also need to follow the instructions on the [lsru page](https://github.com/loicdtx/lsru) to provide credentials to order Landsat imagery. These credentials should be placed in the `work/` folder next to the azure config file. The WBD_GDB_National.gdb folder/geodatabase should also go in the `work` directory on the azure instance.


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


## Setup cropmask conda environment for Jupyter

On the Azure VM do the following
```
conda activate cropmask
conda install nb_conda ipykernel
python -m ipykernel install --user --name cropmask
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

# The following does not work yet, waiting on a github issue with blobfuse
to persist the mounting, add the following to `/etc/fstab`

```
# mounts the blob container at az-ml-container
blobfuse /az-ml-container --tmp-path=/mnt/blobfusetmp -o big_writes -o max_read=131072 -o max_write=131072 -o attr_timeout=240 -o fsname=blobfuse -o entry_timeout=240 -o negative_timeout=120 --config-file
=/opt/blobfuse.cfg fuse _netdev
```

# use the custom mount script in this folder after starting and sshing into the vm. requires /az-ml-container to already be a directory
`bash mount.sh`
