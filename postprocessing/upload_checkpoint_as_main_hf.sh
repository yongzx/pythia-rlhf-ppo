#TODO: currently uploads the checkpoint from LOCAL instead of remote branch to main
SRC_MODEL_DIR="/mnt/ssd-1/pythia-rlhf/checkpoints/sft_hh/pythia-70m"
CKPT="$SRC_MODEL_DIR/checkpoint_14000" 

GIT_FOLDER="/mnt/ssd-1/pythia-rlhf/gh/pythia-70m-sft-hh"
BRANCH_NAME="main"
cd $GIT_FOLDER

###### huggingface login and config
# huggingface-cli login
# git config user.email "usaiprashanth2018@gmail.com"
# git config user.name "usvsnsp"
git config user.email "zheng_xin_yong@brown.edu"
git config user.name "yongzx"

###### copy files
git checkout $BRANCH_NAME # main branch
echo ">>>>> processing $CKPT"
cp -r "${CKPT}/". "${GIT_FOLDER}/"

#### upload to HF
huggingface-cli lfs-enable-largefiles "${GIT_FOLDER}" # Enable lfs for large files
echo "===== Uploading to branch $BRANCH_NAME"
git pull origin $BRANCH_NAME
git add .
git commit -m "upload checkpoint $(basename $CKPT) $BRANCH_NAME"
git push origin $BRANCH_NAME

# huggingface-cli logout