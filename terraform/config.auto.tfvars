# Path to Public SSH key of the admin user (required)
admin_public_key = "~/.ssh/id_rsa.pub"

# Path to Private SSH key of the admin user (required)
admin_private_key = "~/.ssh/id_rsa"

# Datacenter location to deploy the VM into (default: westeurope)
location    = "westus2"

# Name of the virtual machine (acts as prefix for all generated resources, default: dsvm)"
vm_name     = "cropmask-dev"

# Admin username (default: root)
admin_user = "ryan"

# Type of VM to deploy (default: Standard_NC6 - GPU instance)
vm_type = "Standard_NC6s_v2"

# Name of your repo
repo_name = "CropMask_RCNN"

