# How you use CloudLab for FVN setup or FVN with flower framework.

* Atfirst choose a CloudLab profile based on your hardware requirement. Hardware specification you can find from this link(https://docs.cloudlab.us/hardware.html). I have chosen m400 node from CloudLab Utah cluster for my experiment. You can change node based on node availability and RAM, CPU core, CPU speed etc. 

* Install putty and puttygen in your local device. Using puttygen generate public and private keys. Add the public key in "Manage SSH Keys" page of CloudLab. Add the private key in putty for authentication.

* Use SSH command of the corresponding node as the Host Name in putty (e.g., @ms0644.utah.cloudlab.us)
 ```bash
 ssh Parthaw@ms0644.utah.cloudlab.us

* Now in the shell, update the environment (sudo apt update) and install pip (sudo apt install python3-pip)

Step 5: Now you can install other dependencies (e.g., numpy, torch, matplotlib etc) or these can be installed later on jupyter notebook.

Step 6: Install the notebook (pip install notebook) and test the SSH connection (ssh -L 8888:localhost:8888 Parthaw@ms0644.utah.cloudlab.us).

Step 7: Permission can be denied due to not having public key. Debug for getting more information. (ssh -v -L 8888:localhost:8888 Parthaw@ms0644.utah.cloudlab.us)

Step 8: On your local machine, display the content of your public key (e.g., cat ~/.ssh/id_ed25519.pub). If no file or directory found, gennerate one public key (ssh-keygen -t ed25519 -C "parthawgoswami555@gmail.com") 

Step 9: Show the public key and add in the "Manage SSH Keys" page of CloudLab. (cat ~/.ssh/id_ed25519.pub)

Step 10: Now, test the SSH connection again (ssh -L 8888:localhost:8888 Parthaw@ms0644.utah.cloudlab.us).

Step 11:  run Jupyter (jupyter notebook --no-browser --ip=0.0.0.0 --port=8888) and 
collect the URL (e.g., http://127.0.0.1:8889/tree?token=6e226c54143026e4279041e299b7d566d0c997b2e13c37d9)

Step 11: Replace 127.0.0.1 with the remote IP of the node (e.g., http://128.110.216.235:8889/tree?token=6e226c54143026e4279041e299b7d566d0c997b2e13c37d9
128.110.216.235) and paste it to the browser.

Step 12: Now in the Jupyter notebook you can run your experiment. How to check if jupyter notebook is using remote node for experiment? Run the following command to check which IP addresses Jupyter is bound to:

sudo netstat -tuln | grep 8888
