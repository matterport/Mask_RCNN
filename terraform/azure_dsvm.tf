variable "location" {
  description = "Datacenter location to deploy the VM into"
  default     = "westus"
}

variable "vm_name" {
  description = "Name of the virtual machine (acts as prefix for all generated resources)"
  default     = "dsvm"
}

variable "vm_type" {
  description = "The type of VM to deploy"
  default     = "Standard_NC6"
}

variable "admin_user" {
  description = "Admin username"
  default     = "root"
}

variable "admin_public_key" {
  description = "Path to Public SSH key of the admin user"
  default     = "~/.ssh/id_rsa.pub"
}

variable "admin_private_key" {
  description = "Path to Private SSH key of the admin user"
  default     = "~/.ssh/id_rsa"
}

variable "repo_name" {
  description = "The name of the repo, or folder you sync from"
  default     = "CropMask_RCNN"
}

variable "account_name" {
  description = "Name of the azure account"
  default     = ""
}

variable "account_key" {
  description = "Key for the account"
  default     = ""
}

variable "storage_name" {
  description = "Name of the azure storage account"
  default     = ""
}

variable "storage_key" {
  description = "Key for the storage account"
  default     = ""
}

variable "container_name" {
  description = "Name of the blob container in the storage account"
  default     = ""
}

variable "fileshare_name" {
  description = "Name of the file share in the storage account"
  default     = ""
}

variable "lsru_config" {
  description = "Path to the .lsru config gile on your local machine"
  default     = ""
}

resource "azurerm_resource_group" "ds" {
  name     = var.vm_name
  location = var.location
}

resource "azurerm_virtual_network" "ds" {
  name                = "${var.vm_name}-network"
  address_space       = ["10.0.0.0/15"] # was 16 instead of 15
  location            = azurerm_resource_group.ds.location
  resource_group_name = azurerm_resource_group.ds.name
}

resource "azurerm_subnet" "ds" {
  name                 = "${var.vm_name}-subnet"
  resource_group_name  = azurerm_resource_group.ds.name
  virtual_network_name = azurerm_virtual_network.ds.name
  address_prefix       = "10.0.2.0/23" # was 24 instead of 23
}

resource "azurerm_network_interface" "ds" {
  name                = "${var.vm_name}-ni"
  location            = azurerm_resource_group.ds.location
  resource_group_name = azurerm_resource_group.ds.name

  ip_configuration {
    name                          = "${var.vm_name}-cfg"
    subnet_id                     = azurerm_subnet.ds.id
    private_ip_address_allocation = "dynamic"
    public_ip_address_id          = azurerm_public_ip.ds.id
  }
}

resource "azurerm_virtual_machine" "ds" {
  name                             = "${var.vm_name}-vm"
  location                         = azurerm_resource_group.ds.location
  resource_group_name              = azurerm_resource_group.ds.name
  network_interface_ids            = [azurerm_network_interface.ds.id]
  vm_size                          = var.vm_type
  delete_os_disk_on_termination    = true
  delete_data_disks_on_termination = true

  plan {
    name      = "linuxdsvmubuntu"
    publisher = "microsoft-ads"
    product   = "linux-data-science-vm-ubuntu"
  }

  storage_image_reference {
    publisher = "microsoft-ads"
    offer     = "linux-data-science-vm-ubuntu"
    sku       = "linuxdsvmubuntu"
    version   = "latest"
  }

  storage_os_disk {
    name              = "${var.vm_name}-osdisk"
    caching           = "ReadWrite"
    create_option     = "FromImage"
    managed_disk_type = "Standard_LRS"
  }

  # Optional data disks  provisioner "file"{
  storage_data_disk {
    name              = "${var.vm_name}-data"
    managed_disk_type = "Standard_LRS"
    create_option     = "FromImage"
    lun               = 0
    disk_size_gb      = "120"
  }

  os_profile {
    computer_name  = var.vm_name
    admin_username = var.admin_user
  }

  os_profile_linux_config {
    disable_password_authentication = true

    ssh_keys {
      path     = "/home/${var.admin_user}/.ssh/authorized_keys"
      key_data = file(var.admin_public_key)
    }
  }

  tags = {
    environment = "datascience-vm, ${var.vm_name}"
  }

  provisioner "remote-exec" {
    inline = [
      "mkdir -p work",
    ]

    connection {
      type        = "ssh"
      user        = var.admin_user
      private_key = file(var.admin_private_key)
      host        = azurerm_public_ip.ds.ip_address
    }
  }
}

# create a public IP to bind against the VM
resource "azurerm_public_ip" "ds" {
  name                         = "${var.vm_name}-ip"
  location                     = azurerm_resource_group.ds.location
  resource_group_name          = azurerm_resource_group.ds.name
  public_ip_address_allocation = "static"

  tags = {
    environment = "datascience-vm, ${var.vm_name}"
  }
}

# dump reference to public IP and VM ID to local files; if anything changes just re-run terraform apply to re-generate the files locally
resource "null_resource" "ds" {
  triggers = {
    vm_id      = azurerm_virtual_machine.ds.id
    ip_address = azurerm_public_ip.ds.ip_address
  }

  provisioner "local-exec" {
    command = "echo ${azurerm_virtual_machine.ds.id} > .vm-id"
  }

  provisioner "local-exec" {
    command = "echo ${var.admin_user}@${azurerm_public_ip.ds.ip_address} > .vm-ip"
  }

  provisioner "local-exec" {
    command     = "make syncup"
    working_dir = "../"
  }

  # mounts the blob container with blobfuse on the vm. used for saving out landsat
  # setting secrets
  # mounts the blob container with blobfuse on the vm. used for saving out landsat
  # setting secrets
  provisioner "remote-exec" {
    inline = [
      "export ACCOUNT_NAME=${var.account_name}",
      "export ACCOUNT_KEY=${var.account_key}",
      "export STORAGE_NAME=${var.storage_name}",
      "export STORAGE_KEY=${var.storage_key}",
      "export FILESHARE_NAME=${var.fileshare_name}",
      "export CONTAINER_NAME=${var.container_name}",
      "echo \"accountName $STORAGE_NAME\" > /home/${var.admin_user}/work/blobfuse.cfg",
      "echo \"accountKey $STORAGE_KEY\" >> /home/${var.admin_user}/work/blobfuse.cfg",
      "echo \"containerName $CONTAINER_NAME\" >> /home/${var.admin_user}/work/blobfuse.cfg",
      "bash /home/${var.admin_user}/work/${var.repo_name}/bash-scripts/mount-fileshare.sh $STORAGE_NAME $FILESHARE_NAME $STORAGE_KEY",
      "bash /home/${var.admin_user}/work/${var.repo_name}/bash-scripts/mount-blobfuse.sh",
      "bash /home/${var.admin_user}/work/${var.repo_name}/bash-scripts/setup_env.sh ${var.repo_name}",
    ]

    connection {
      type        = "ssh"
      user        = var.admin_user
      private_key = file(var.admin_private_key)
      host        = azurerm_public_ip.ds.ip_address
    }
  }

  provisioner "file" {
    source      = var.lsru_config
    destination = "/home/${var.admin_user}/.lsru"

    connection {
      type        = "ssh"
      user        = var.admin_user
      private_key = file(var.admin_private_key)
      host        = azurerm_public_ip.ds.ip_address
    }
  }
}

