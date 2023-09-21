SRC_MODEL_DIR="/mnt/ssd-1/pythia-rlhf/checkpoints/sft_hh/pythia-6.9b"

# huggingface-cli login
GIT_FOLDER="/mnt/ssd-1/pythia-rlhf/gh/pythia-6.9b-sft-hh"
cd $GIT_FOLDER
git config user.email "usaiprashanth2018@gmail.com"
git config user.name "usvsnsp"


for CKPT in $SRC_MODEL_DIR/*/ ; do

    BRANCH_NAME=$(basename $CKPT)

    if [[ "$BRANCH_NAME" == "checkpoint_01000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_02000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_03000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_04000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_05000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_06000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_07000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_08000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_09000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_10000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_11000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_12000" ]]; then
        continue
    fi
    if [[ "$BRANCH_NAME" == "checkpoint_13000" ]]; then
        continue
    fi
    ### set up branch
    # only checks for local git branch for now (only uploading, doesn't care about remote.)
    # TODO: extends this to handle remote branch.
    if git show-ref --quiet refs/heads/$BRANCH_NAME;
    then
        echo "✅ Branch named $BRANCH_NAME already exists"
        git checkout $BRANCH_NAME
    else
        echo "⏪ Branch named $BRANCH_NAME does not exist"
        git checkout -b $BRANCH_NAME
    fi
    
    ### copy over files
    echo ">>>>> processing $CKPT"
    cp -r "${CKPT}"* "${GIT_FOLDER}/"

    ### Enable lfs for large files
    huggingface-cli lfs-enable-largefiles "${GIT_FOLDER}"

    #### git push
    echo "===== Uploading to branch $BRANCH_NAME"
    git add .
    git commit -m "upload checkpoint $BRANCH_NAME"
    git push origin $BRANCH_NAME

    if [ $? -eq 0 ]; then
        echo "🔥 success uploading $BRANCH_NAME" 
    else
        echo FAIL
        break
    fi
done

# huggingface-cli logout