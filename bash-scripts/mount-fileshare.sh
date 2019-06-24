# once off case below
sudo mkdir -p /permmnt/$2
sudo chown -R ryan /permmnt/$2
if [ ! -d "/etc/smbcredentials" ]; then
sudo mkdir /etc/smbcredentials
fi
if [ ! -f "/etc/smbcredentials/$1.cred" ]; then
sudo bash -c 'echo "username=$1" >> /etc/smbcredentials/$1.cred'
sudo bash -c 'echo "password=$3" >> /etc/smbcredentials/$1.cred'
fi

sudo chmod 600 /etc/smbcredentials/$1.cred

sudo bash -c 'echo "//$1.file.core.windows.net/$2 /permmnt/$2 cifs nofail,vers=3.0,credentials=/etc/smbcredentials/$1.cred,dir_mode=0777,file_mode=0777,serverino" >> /etc/fstab'

sudo mount /permmnt/$2
# sudo mount -t cifs //$1.file.core.windows.net/$2 /permmnt/$2 -o vers=3.0,username=$1,password=$3,dir_mode=0777,file_mode=0777,serverino

#1 is the storage name, 2 is the share name and 3 is the account key
