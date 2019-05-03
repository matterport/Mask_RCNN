sudo mount -t cifs //$1.file.core.windows.net/$2 /mnt/fileshare -o vers=3.0,username=$1,password=$3,dir_mode=0777,file_mode=0777,serverin
#1 is the storage name, 2 is the share name and 3 is the account key
