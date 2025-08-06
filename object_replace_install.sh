#!/bin/bash

# Everything needs to be install manually. So, save your precious time!
# make a directory
apt update -y
apt upgrade -y

cd
cd /home/evobits/

git clone https://github.com/bitmorpherserver/Foocus_ObjectReplace.git
cd Foocus_ObjectReplace/

python -m venv .venv
source .venv/bin/activate


cd /home/evobits/Foocus_ObjectReplace
pip install -r requirements.txt 

cp fooocus_object_replace.service /etc/systemd/system/
cp fooocus_object_replace.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable fooocus_object_replace
service fooocus_object_replace start



cp fooocus_object_replace_nginx.conf /etc/nginx/sites-available/
ln -s /etc/nginx/sites-available/fooocus_object_replace_nginx.conf /etc/nginx/sites-enabled/
service nginx restart

# python main.py 
# make a directory at /var/log/ named foocus_object_replace 
# clear port before loading project 
# do not have same file name. 
