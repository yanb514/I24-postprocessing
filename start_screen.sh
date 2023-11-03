#!/bin/bash

echo User:
whoami

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/remote/i24_common/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/remote/i24_common/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/remote/i24_common/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/remote/i24_common/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate postproc

echo Conda:
python -V
conda info --env | grep '*'

export USER_CONFIG_DIRECTORY=/remote/i24_config/$HOSTNAME
export USER_CONFIG_SECTION=DEBUG

if [ -d "$USER_CONFIG_DIRECTORY" ]; then
    # config dir is OK

    echo "USER_CONFIG_DIRECTORY is at $USER_CONFIG_DIRECTORY"
else
    # config dir is missing

    echo -e "\033[0;31m---- USER_CONFIG_DIRECTORY=$USER_CONFIG_DIRECTORY is MISSING!! ----\033[0m"
fi

TIME=`date +"%Y-%m-%d_%H-%M-%S"`

#echo $TIME

config="logfile /local/postproc_${TIME}.log
logfile flush 5
log on
logtstamp after 20
logtstamp string \"[ %t %Y-%m-%d %c:%s ]\012\"
logtstamp on";

echo "$config" > /tmp/log.conf
screen -c /tmp/log.conf -SL postproc-manual
rm /tmp/log.conf

echo Service Started