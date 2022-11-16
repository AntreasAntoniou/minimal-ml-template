# Tutorial Notes

## Use repository

1. Clone repository
2. Use vscode editor to launch on remote GCP machine
3. Build container automatically using VS code
4. run huggingface-cli login
5. copy token from website
6. mamba install -c conda-forge git-lfs
7. git lfs install

## Dealing with secret passcodes/tokens in docker

1. Create a docker secret

   ```bash
   docker secret create my_secret_code password1234 
   ```

2. When building/running an image, use

   ```bash
   docker run --secret my_secret_code
   ```

3. Your secret is now under /run/secrets/my_secret_code
