# Running MLProject experiments on kubernetes

1. Create a docker image for your experiments by:
   1. Modify the [Dockerfile](../Dockerfile) to build an image that can succesfully run your experiment.
   2. Build the images by running `docker build . -t image-name:version` from within the minimal-ml-template directory.
   3. Go to your [Github token generation page](https://github.com/settings/tokens/new) and generate a token that allows package management push permissions. Copy the new token.
   4. Then to push your image to the github image registry, run:
        ```bash
        export CR_PAT=YOUR_TOKEN ; echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
        docker tag SOURCE_IMAGE_NAME:VERSION ghcr.io/TARGET_OWNER/TARGET_IMAGE_NAME:VERSION
        docker push ghcr.io/OWNER/IMAGE_NAME:VERSION
        ```
        Replacing YOUR_TOKEN with your github api key, USERNAME with your github username, SOURCE_IMAGE_NAME:VERSION with the local name and version of the image you build, and TARGET_OWNER/TARGET_IMAGE_NAME:VERSION with the target github owner (TARGET_OWNER) under which you want to push the package, along with a package name (TARGET_IMAGE_NAME) and a version (VERSION)
 2. Once your image has been pushed to a relevant image registry, you should fill in the variables in the file [setup_variables.sh](setup_variables.sh) file with their respective values, and then export the variables to your local VM and the kubernetes cluster by running:
    ```bash
    source kubernetes/setup_variables.sh
    bash kubernetes/setup_kubernetes_variables.sh
    bash kubernetes/setup_secrets.sh
    ```
 3. Modify the [runner script](run_kube_experiments.py) to generate all the experiment commands you'd like to be launched in the kubernetes cluster by modifying the `get_scripts()` method.
 4. Run the runner script to launch your experiments:
    ```bash
    python kubernetes/run_kube_experiments.py
    ```